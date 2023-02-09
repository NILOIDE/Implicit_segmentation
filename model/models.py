import abc
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Any, Dict, Optional
import numpy as np
import pytorch_lightning as pl
from monai.losses import DiceLoss
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import nibabel as nib

import logging

from tqdm import tqdm

# Prevent unnecessary warning prints
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib
import matplotlib.pyplot as plt
# Prevent pyplot from opening popup windows when generating figures
plt.ioff()  # Turn interactive mode to off
matplotlib.use('Agg')  # Changing non-gui backend to Agg

import os
# Stop OpenMP from complaining about multiple instances (caused by Lightning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from data_loading.data_loader import Seg3DWholeImage_SAX_test, Seg4DWholeImage_SAX_test
from model.activation_functions import Sine, WIRE
from model.mlp import MLP, ResMLP, ResMLPHiddenCoords, MLPHiddenCoords, ReconstructionHead, SegmentationHead
from model.pos_encoding import PosEncodingNeRFOptimized, PosEncodingGaussian, PosEncodingNone
from utils import ValProgressBar, draw_mask_to_image, to_1hot


def process_params(params: Dict[str, Any]):
    if params["activation"] == "periodic":
        activation = Sine
    elif params["activation"] == "relu":
        activation = nn.ReLU
    elif params["activation"] == "wire":
        activation = WIRE
    else:
        raise ValueError

    if params["pos_encoding"] == "none":
        pos_encoding = PosEncodingNone
    elif params["pos_encoding"] == "nerf":
        pos_encoding = PosEncodingNeRFOptimized
    elif params["pos_encoding"] == "gaussian":
        pos_encoding = PosEncodingGaussian
    else:
        raise ValueError

    if params["skip_connections"]:
        if params["input_coord_to_all_layers"]:
            model = ResMLPHiddenCoords
        else:
            model = ResMLP
    else:
        if params["input_coord_to_all_layers"]:
            model = MLPHiddenCoords
        else:
            model = MLP
    print(f"Architecture: {params['model_type']}, MLP_type: {model.__name__}, Activation_func: {activation.__name__}, Pos_enc: {str(pos_encoding(**params))}, Image_shape: {params['side_length']}")
    return activation, pos_encoding, model


class AbstractPrior(pl.LightningModule):
    def __init__(self, *args, aug_num_parameters: int = 0, **kwargs):
        super(AbstractPrior, self).__init__()
        self.save_hyperparameters(kwargs)
        self.activation_class, self.pos_encoding_class, self.backbone_class = process_params(kwargs)
        self.latent_size = kwargs.get("latent_size")
        self.lr = kwargs.get("lr")
        self.num_train_samples = kwargs.get("num_train_samples")
        self.num_coord_dims = kwargs.get("coord_dimensions")
        self.side_length = kwargs.get("side_length")
        self.h = nn.Parameter(torch.normal(0., 0.1**2, (self.num_train_samples, self.latent_size - aug_num_parameters,)), requires_grad=True)

        self.latent_reg = kwargs.get("latent_reg")
        self.weight_reg = kwargs.get("weight_reg")
        self.backbone_reg = self.weight_reg > 0.0

        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        self.l2_loss = nn.MSELoss()
        self.dice_loss = None
        self.seg_class_weights = None

        self.initial_val = kwargs.get("initial_val", False)
        self.train_log_interval = kwargs.get("train_log_interval")
        self.val_epochs = kwargs.get("val_max_epochs")
        self.val_interval = kwargs.get("val_interval")
        self.params = kwargs
        self.split_name = 'train'

    @abc.abstractmethod
    def val_dataset(self) -> torch.utils.data.Dataset:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_progress_bar_dict(self):
        """ Remove 'v_num' from progress bar """
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    def get_normalized_dist_map(self, mask, dist_cutoff=.5):
        """
        :param mask: Background mask (background = 1, foreground = 0)
        :param dist_cutoff: Proportion in terms of image height that will be used as cutoff distance (clip
        # """
        # TODO fix edge slices
        dist_map = distance_transform_edt((mask==0).cpu().numpy())
        dist_map = torch.tensor(dist_map, dtype=torch.float32, device=mask.device)
        dist_map = dist_map / (dist_cutoff * mask.shape[1])
        dist_map = dist_map.clip(min=0.01)
        return dist_map

    def reconstruction_criterion(self, pred_im, gt_im, background_mask=None, log=True):
        loss = self.bce_loss(pred_im, gt_im)
        if background_mask is not None:
            loss = loss * self.get_normalized_dist_map(background_mask)
        loss = loss.mean()
        if log:
            self.log(f"{self.split_name}/pixel_L2", loss.item(), prog_bar=True)
        return loss

    def regularization_criterion(self, h, log=True):
        loss = self.l2_loss(h, torch.zeros_like(h))
        if log:
            self.log(f"{self.split_name}/latent_L2", loss.item())
        loss *= self.latent_reg
        if self.backbone_reg:
            backbone_l2 = sum((p*p).sum() for p in self.backbone.parameters())
            if log:
                self.log(f"{self.split_name}/backbone_L2", loss.item())
            loss += backbone_l2 * self.weight_reg
        return loss

    def segmentation_criterion(self, pred_seg, gt_seg, log=True):
        gt_seg_1hot = to_1hot(gt_seg)
        non_class_dims = (0, *tuple(range(2, len(pred_seg.shape))))
        loss_seg_bce = self.bce_loss(pred_seg, gt_seg_1hot).mean(non_class_dims)
        loss_seg_bce_weighted = (loss_seg_bce * self.seg_class_weights).mean()
        loss_seg_dice = self.dice_loss(pred_seg, gt_seg_1hot).mean(non_class_dims)
        loss_seg_dice_weighted = (loss_seg_dice * self.seg_class_weights).mean()

        if log:
            self.log(f"{self.split_name}/seg_bce_loss", loss_seg_bce.mean().item())
            self.log(f"{self.split_name}/seg_dice_loss", loss_seg_dice.mean().item())

            with torch.no_grad():
                dice = 1 - self.dice_loss(pred_seg.round(), gt_seg_1hot).mean(non_class_dims)
                dice = dice.cpu().tolist()
            self.log(f"{self.split_name}/seg_dice_BG", dice[0], prog_bar=True)
            self.log(f"{self.split_name}/seg_dice_LV_Pool", dice[1], prog_bar=True)
            self.log(f"{self.split_name}/seg_dice_LV_Myo", dice[2], prog_bar=True)
            self.log(f"{self.split_name}/seg_dice_RV_Pool", dice[3], prog_bar=True)

        return loss_seg_dice_weighted + loss_seg_bce_weighted

    def do_validation(self, epoch):
        self.eval()
        params = deepcopy(self.params)
        if "dropout" in params:
            params["dropout"] = 0.0
        val_model = self.val_model_class(parent_logger=self.logger,
                                         train_epoch=epoch,
                                         sample_img_path=self.val_dataset.im_paths[0],
                                         sample_seg_path=self.val_dataset.seg_paths[0],
                                         **params)
        state_dict = deepcopy(self.state_dict())
        del state_dict['h']
        miss_keys = val_model.load_state_dict(state_dict, strict=False)
        assert 'h' in miss_keys.missing_keys and len(miss_keys.missing_keys) == 1
        val_model.h = nn.Parameter(torch.normal(0., 1e-4 ** 2, val_model.h.shape))
        val_trainer = pl.Trainer(max_epochs=self.val_epochs, accelerator="gpu",
                                 enable_model_summary=False, logger=False,
                                 enable_checkpointing=False, callbacks=[ValProgressBar()])
        val_trainer.fit(val_model, train_dataloaders=DataLoader(self.val_dataset, batch_size=1, shuffle=False))
        del val_trainer, val_model
        self.train()

    def on_fit_start(self):
        if self.initial_val:
            self.do_validation(self.current_epoch)

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval == 0:
            self.do_validation(self.current_epoch + 1)


class AbstractLatent(pl.LightningModule):
    def __init__(self, parent_logger=None, train_epoch=None, sample_img_path=None, sample_seg_path=None, *args, **kwargs):
        super(AbstractLatent, self).__init__(*args, **kwargs)
        self.num_coord_dims = kwargs.get("coord_dimensions")
        self.max_epochs = kwargs.get("val_max_epochs")
        self.h = nn.Parameter(torch.normal(0., 1e-4**2, (1, self.latent_size)), requires_grad=True)
        self.parent_logger = parent_logger
        self.train_epoch = train_epoch
        self.split_name = 'val'
        self.z_holdout_rate = kwargs.get("val_z_holdout_rate", 1)
        self.t_holdout_rate = kwargs.get("val_t_holdout_rate", 1)
        self.sample_img_path = sample_img_path
        self.sample_seg_path = sample_seg_path

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)

    def on_fit_start(self):
        return

    def on_train_epoch_end(self):
        return

    def on_fit_end(self):
        if self.parent_logger is not None:
            self.draw_fit_plot()
            if self.num_coord_dims == 4 and self.sample_img_path is not None and self.sample_seg_path is not None:
                vid = self.draw_time_video(self.sample_img_path, self.sample_seg_path)
                self.parent_logger.experiment.add_video(f"val_end_video", vid[None], fps=10, global_step=self.train_epoch)

    def val_logging(self,
                    step: int,
                    image: torch.Tensor,
                    seg: Optional[torch.Tensor] = None,
                    loss_pixel: Optional[float] = None,
                    loss_seg: Optional[float] = None,
                    dice_BG: Optional[float] = None,
                    dice_LV_Pool: Optional[float] = None,
                    dice_LV_Myo: Optional[float] = None,
                    dice_RV_Pool: Optional[float] = None,
                    loss_reg: Optional[float] = None):

        # Log to progress bar
        if loss_pixel is not None:
            self.log(f"val_pixel_L2", loss_pixel, prog_bar=True, on_step=True)
        if loss_seg is not None:
            self.log(f"val_seg_loss", loss_seg, prog_bar=True, on_step=True)
        if dice_BG is not None:
            self.log(f"val_dice_BG", dice_BG, prog_bar=True, on_step=True)
        if dice_LV_Pool is not None:
            self.log(f"val_dice_LV_Pool", dice_LV_Pool, prog_bar=True, on_step=True)
        if dice_LV_Myo is not None:
            self.log(f"val_dice_LV_Myo", dice_LV_Myo, prog_bar=True, on_step=True)
        if dice_RV_Pool is not None:
            self.log(f"val_dice_RV_Pool", dice_RV_Pool, prog_bar=True, on_step=True)

        if self.parent_logger is not None and self.current_epoch % self.val_log_interval == 0:
            # Log to training trainer's logger
            if loss_pixel is not None:
                self.parent_logger.experiment.add_scalar(f"val_pixel_L2/{step}", loss_pixel,
                                                         global_step=self.train_epoch)
            if loss_seg is not None:
                self.parent_logger.experiment.add_scalar(f"val_seg_loss/{step}", loss_seg,
                                                         global_step=self.train_epoch)
            if dice_BG is not None:
                self.parent_logger.experiment.add_scalar(f"val_dice_BG/{step}", dice_BG,
                                                         global_step=self.train_epoch)
            if dice_LV_Pool is not None:
                self.parent_logger.experiment.add_scalar(f"val_dice_LV_Pool/{step}", dice_LV_Pool,
                                                         global_step=self.train_epoch)
            if dice_LV_Myo is not None:
                self.parent_logger.experiment.add_scalar(f"val_dice_LV_Myo/{step}", dice_LV_Myo,
                                                         global_step=self.train_epoch)
            if dice_RV_Pool is not None:
                self.parent_logger.experiment.add_scalar(f"val_dice_RV_Pool/{step}", dice_RV_Pool,
                                                         global_step=self.train_epoch)
            if loss_reg is not None:
                self.parent_logger.experiment.add_scalar(f"val_reg_L2/{step}", loss_reg,
                                                         global_step=self.train_epoch)
            # Draw images and save them to logger
            im = self.draw_image(image, seg)
            self.parent_logger.experiment.add_image(f"val_images_{step}", im, global_step=self.train_epoch)

    def draw_image(self, *args):
        raise NotImplementedError

    def draw_fit_plot(self):
        print("Generating fit plot...")
        fig, ax = plt.subplots(1, 3, )
        fig.set_figwidth(fig.get_figheight() * 3)
        ax[0].plot(list(range(len(self.train_loss))), self.train_loss, label="Image L2 Loss")
        ax[0].set_title("Image Reconstruction Loss")
        ax[0].legend(loc="upper right")
        ax[1].plot(list(range(len(self.dice))), [i[0] for i in self.dice], label="Background")
        ax[1].plot(list(range(len(self.dice))), [i[1] for i in self.dice], label="LV Pool")
        ax[1].plot(list(range(len(self.dice))), [i[2] for i in self.dice], label="LV Myo")
        ax[1].plot(list(range(len(self.dice))), [i[3] for i in self.dice], label="RV Pool")
        ax[1].set_ylim(0.5, 1.0)
        ax[1].yaxis.set_ticks(np.arange(0.5, 1.0, 0.025))
        ax[1].grid("x")
        ax[1].set_title("Segmentation Dice")
        ax[1].legend(loc="upper right")
        ax[2].plot(list(range(len(self.latent_reg_loss))), self.latent_reg_loss, label="Latent L2")
        ax[2].set_title("Latent L2 Loss")
        ax[2].legend(loc="upper right")
        self.parent_logger.experiment.add_figure("val_loss_curve", fig, global_step=self.train_epoch)
        del fig

    def draw_time_video(self, gt_im_path, gt_seg_path, z_values=(0.1, 0.3, 0.5, 0.7, 0.9)) -> torch.Tensor:
        nii_img = nib.load(gt_im_path)
        nii_seg = nib.load(gt_seg_path)
        raw_shape = nii_img.shape
        z_indices = [int(raw_shape[-2] * z_prop) for z_prop in z_values]

        # Take only one time frame of the series
        gt_im_vol = nii_img.dataobj[:]
        gt_seg_vol = nii_seg.dataobj[:].astype(np.uint8)
        # Crop the image into a square based on which side is the longest
        max_dim = np.argmax(gt_im_vol.shape)
        if max_dim == 0:
            start_idx = (gt_im_vol.shape[0] - gt_im_vol.shape[1]) // 2
            gt_im_vol = gt_im_vol[start_idx: start_idx + gt_im_vol.shape[1]]
            gt_seg_vol = gt_seg_vol[start_idx: start_idx + gt_seg_vol.shape[1]]
        elif max_dim == 1:
            start_idx = (gt_im_vol.shape[1] - gt_im_vol.shape[0]) // 2
            gt_im_vol = gt_im_vol[:, start_idx: start_idx + gt_im_vol.shape[0]]
            gt_seg_vol = gt_seg_vol[:, start_idx: start_idx + gt_seg_vol.shape[0]]
        else:
            raise ValueError
        assert gt_im_vol.shape[0] == gt_im_vol.shape[1]
        min_, max_ = gt_im_vol.min(), gt_im_vol.max()
        gt_im_vol = (gt_im_vol - min_) / (max_ - min_)
        frames = []
        for t in tqdm(range(0, gt_im_vol.shape[-1]), desc="Generating video"):
            rows = []
            for z_idx in z_indices:
                gt_im = gt_im_vol[:, :, z_idx, t]
                gt_seg = gt_seg_vol[:, :, z_idx, t]
                gt_im = cv2.resize(gt_im, self.side_length[:2])
                gt_seg = cv2.resize(gt_seg, self.side_length[:2], interpolation=cv2.INTER_NEAREST)

                gt_im = np.stack([gt_im]*3, axis=-1)
                gt_seg = draw_mask_to_image(gt_im, gt_seg)

                z_coord = z_idx/raw_shape[-2]
                t_coord = t/raw_shape[-1]
                pred_im, pred_seg = self.evaluate_2D_from_constant(gt_im.shape[:2], dim3_value=z_coord, time_value=t_coord)
                pred_im = np.stack([pred_im]*3, axis=-1)
                pred_seg = np.argmax(pred_seg, axis=0)
                pred_seg = draw_mask_to_image(pred_im, pred_seg)
                rows.append(np.concatenate((gt_im, pred_im, gt_seg, pred_seg), axis=1))
            frames.append(np.concatenate(rows, axis=0))
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).moveaxis(-1, 1)
        return frames

    @torch.no_grad()
    def evaluate(self, coord_arr: torch.Tensor, h_vector: torch.Tensor) -> np.array:
        assert coord_arr.shape[-1] == self.num_coord_dims
        assert len(h_vector.shape) == 1
        coord_arr = coord_arr.to(self.device)
        pred_im, pred_seg = self.forward(coord_arr[None], h_vector[None])
        return pred_im[0].cpu().numpy(), pred_seg[0].cpu().numpy()

    def evaluate_volume(self, out_shape: Tuple[int, ...], t=0.0):
        """ NOTE: torch.meshgrid has a different behaviour than np.meshgrid,
        using both interchangeably will produce transposed images. """
        coords = torch.meshgrid(torch.arange(out_shape[0], dtype=torch.float32),
                                torch.arange(out_shape[1], dtype=torch.float32),
                                torch.arange(out_shape[2], dtype=torch.float32))
        coords = [(c / out_shape[i]) for i, c in enumerate(coords)]
        if self.num_coord_dims == 4:
            t = torch.full_like(coords[0], t)
            coords.append(t)
        coord_arr = torch.stack(coords, dim=-1)
        return self.evaluate(coord_arr, self.h[0])

    def evaluate_2D_from_constant(self,
                                  out_shape: Tuple[int, ...],
                                  dim1_value: float = None,
                                  dim2_value: float = None,
                                  dim3_value: float = None,
                                  time_value: float = None,
                                  ) -> np.array:
        # Create meshgrid (coordinates same for every dimension)
        assert len(out_shape) == 2
        coords = torch.meshgrid(torch.arange(out_shape[0], dtype=torch.float32), torch.arange(out_shape[1], dtype=torch.float32))

        # Figure out which dimension the user wants filled with a constant
        count = 0
        if dim1_value is None:
            dim1 = coords[count] / out_shape[count]
            count += 1
        else:
            dim1 = torch.full_like(coords[0], dim1_value)
        if dim2_value is None:
            dim2 = coords[count] / out_shape[count]
            count += 1
        else:
            dim2 = torch.full_like(coords[0], dim2_value)
        if dim3_value is None:
            dim3 = coords[count] / out_shape[count]
            count += 1
        else:
            dim3 = torch.full_like(coords[0], dim3_value)
        if time_value is None:
            dimt = coords[count] / out_shape[count]
            count += 1
        else:
            dimt = torch.full_like(coords[0], time_value)

        if self.num_coord_dims == 4:
            dims = (dim1, dim2, dim3, dimt)
        elif self.num_coord_dimss == 3:
            if time_value is None:
                dims = (dim1, dim2, dim3)
            else:
                dims = (dim1, dim2, dimt)
        else:
            raise ValueError("There should be at least 3 dims")
        coord_arr = torch.stack(dims, dim=-1)
        return self.evaluate(coord_arr, self.h[0])


class ImplicitNetSegPrior(AbstractPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetSegPrior, self).__init__(*args, **kwargs)
        self.num_classes = 4
        self.pos_enc = self.pos_encoding_class(**kwargs)
        self.backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.recon_head = ReconstructionHead(self.backbone.out_size, **kwargs)
        self.seg_head = SegmentationHead(self.backbone.out_size, self.num_classes)
        self.dice_loss = DiceLoss(reduction="none")
        self.val_model_class = ImplicitNetSegLatent
        self.seg_class_weights = torch.tensor(kwargs.get("seg_class_weights"), dtype=torch.float32, device="cuda")

    @property
    def val_dataset(self) -> torch.utils.data.Dataset:
        return Seg4DWholeImage_SAX_test(**self.params)

    def forward(self, coord, h):
        coord_ = coord.view((-1, coord.shape[-1]))
        # Tile h vector in a manner to allow for batch size > 1
        h_ = torch.tile(h, (1, coord_.shape[0] // coord.shape[0])).view((-1, h.shape[1]))

        c_enc = self.pos_enc(coord_)
        out = self.backbone((c_enc, h_))
        pred_recon_ = self.recon_head(out)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        pred_seg_ = self.seg_head(out)
        pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1])).moveaxis(-1, 1)
        return pred_recon, pred_seg

    def training_step(self, batch):
        coord, image, sample_idx, seg, aug_params = batch
        h = self.h[sample_idx].cuda()
        h = torch.cat((h, aug_params), dim=1)
        pred_im, pred_seg = self.forward(coord, h)
        loss = 0
        rec_loss = self.reconstruction_criterion(pred_im, image)#, seg == 0)
        loss += rec_loss
        seg_loss = self.segmentation_criterion(pred_seg, seg)
        loss += seg_loss
        reg_loss = self.regularization_criterion(self.h[sample_idx])
        loss += reg_loss
        if self.current_epoch % self.train_log_interval == 0 and sample_idx.item() < 3:
            self.logger.experiment.add_image(f"train_images_{sample_idx.item()}",
                                             self.draw_training_image(image[0], pred_im[0], seg[0], pred_seg[0]),
                                             global_step=self.current_epoch)
        return loss

    @torch.no_grad()
    def draw_training_image(self, gt_im, pred_im, gt_seg, pred_seg):
        gt_im, pred_im, gt_seg, pred_seg = gt_im.cpu(), pred_im.cpu(), gt_seg.cpu(), pred_seg.cpu()
        gt_im_row = [np.stack([gt_im[..., i].numpy()]*3, axis=-1) for i in range(gt_im.shape[-1])]
        gt_seg_row = [draw_mask_to_image(gt_im[..., i].numpy(), gt_seg[..., i].numpy()) for i in range(gt_seg.shape[-1])]
        pred_im_row = [np.stack([pred_im[..., i].numpy()]*3, axis=-1) for i in range(pred_im.shape[-1])]
        pred_seg_row = [draw_mask_to_image(pred_im[..., i],
                                           np.argmax(pred_seg[..., i], axis=0)) for i in range(pred_im.shape[-1])]
        pred_im_row = np.concatenate(pred_im_row, axis=1)
        pred_seg_row = np.concatenate(pred_seg_row, axis=1)
        gt_im_row = np.concatenate(gt_im_row, axis=1)
        gt_seg_row = np.concatenate(gt_seg_row, axis=1)

        im = np.concatenate((gt_im_row, pred_im_row, pred_seg_row, gt_seg_row), axis=0)
        return im.transpose((2, 0, 1))


class ImplicitNetSegLatent(AbstractLatent, ImplicitNetSegPrior):
    def __init__(self, parent_logger=None, train_epoch=None, *args, **kwargs):
        super(ImplicitNetSegLatent, self).__init__(*args, **kwargs)
        self.parent_logger = parent_logger
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_epoch = train_epoch
        self.train_loss = []
        self.dice = []
        self.latent_reg_loss = []
        self.batch = None

    def training_step(self, batch):
        coord, image, sample_idx, seg, _ = batch
        coord_ = coord[..., torch.arange(0, coord.shape[-2], self.z_holdout_rate), :]
        image_ = image[..., torch.arange(0, image.shape[-1], self.z_holdout_rate)]
        seg_ = seg[..., torch.arange(0, seg.shape[-1], self.z_holdout_rate)]
        pred_im, pred_seg = self.forward(coord_, self.h)

        loss = 0
        # Pixel regression loss
        background = pred_seg[:, 0].detach() > 0.5
        loss_pixel = self.reconstruction_criterion(pred_im, image_, log=False)#, background_mask=background)
        self.train_loss.append(loss_pixel.item())
        loss += loss_pixel

        # Segmentation dice (not trained on, assume not available at test time). Not added to overall loss here.
        with torch.no_grad():
            dice = (1 - self.dice_loss(pred_seg.round(), to_1hot(seg_))).mean(0).squeeze().detach().cpu().tolist()
            self.dice.append(dice)

        # Regularization loss for h (and optionally network weights)
        loss_reg = None
        loss_reg = self.regularization_criterion(self.h[sample_idx], log=False)
        self.latent_reg_loss.append(loss_reg.item())
        loss += loss_reg

        progress = dict(loss_pixel=loss_pixel.item(),
                        dice_BG=dice[0],
                        dice_LV_Pool=dice[1],
                        dice_LV_Myo=dice[2],
                        dice_RV_Pool=dice[3],
                        loss_reg=loss_reg.item())
        del loss_pixel, dice, loss_reg
        self.val_logging(self.current_epoch, image[0], seg=seg[0], **progress)
        return loss

    def draw_image(self, gt_im, gt_seg):
        gt_2D_ims = [gt_im[..., i].cpu().numpy() for i in range(gt_im.shape[-1])]
        gt_2D_segs = [gt_seg[..., i].cpu().numpy() for i in range(gt_seg.shape[-1])]
        gt_im_row = []
        gt_seg_row = []
        for i, (og_im, og_seg) in enumerate(zip(gt_2D_ims, gt_2D_segs)):
            og_im_rgb = np.stack((og_im, og_im, og_im), axis=-1)
            if i % self.z_holdout_rate == 0:
                og_im_rgb[[0]*og_im_rgb.shape[1], list(range(og_im_rgb.shape[1]))] = np.array((0, 1, 0))
            gt_im_row.append(og_im_rgb)
            gt_im_row.append(np.zeros_like(og_im_rgb))
            masked_im = draw_mask_to_image(og_im, og_seg)
            gt_seg_row.append(masked_im)
            gt_seg_row.append(np.zeros_like(masked_im))

        pred_im, pred_seg = self.evaluate_volume((*gt_im.shape[:2], gt_im.shape[2] * 2))
        pred_im_row = [np.stack([pred_im[..., i]]*3, axis=-1) for i in range(pred_im.shape[-1])]
        pred_seg_row = [draw_mask_to_image(pred_im[..., i],
                                           np.argmax(pred_seg[..., i], axis=0)) for i in range(pred_im.shape[-1])]
        pred_im_row = np.concatenate(pred_im_row, axis=1)
        pred_seg_row = np.concatenate(pred_seg_row, axis=1)
        gt_im_row = np.concatenate(gt_im_row, axis=1)
        gt_seg_row = np.concatenate(gt_seg_row, axis=1)

        im = np.concatenate((gt_im_row, pred_im_row, pred_seg_row, gt_seg_row), axis=0)
        return im.transpose((2, 0, 1))


class ImplicitNetSeparateSegPrior(ImplicitNetSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetSeparateSegPrior, self).__init__(*args, **kwargs)
        self.seg_backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.val_model_class = ImplicitNetSeparateSegLatent

    def forward(self, coord, h):
        coord_ = coord.view((-1, coord.shape[-1]))
        # Tile h vector in a manner to allow for batch size > 1
        h_ = torch.tile(h, (1, coord_.shape[0] // coord.shape[0])).view((-1, h.shape[1]))

        c_enc = self.pos_enc(coord_)
        out_recon_ = self.backbone((c_enc, h_))
        pred_recon_ = self.recon_head(out_recon_)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        out_seg_ = self.seg_backbone((c_enc, h_))
        pred_seg_ = self.seg_head(out_seg_)
        pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1])).moveaxis(-1, 1)
        return pred_recon, pred_seg


class ImplicitNetSeparateSegLatent(ImplicitNetSegLatent, ImplicitNetSeparateSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetSeparateSegLatent, self).__init__(*args, **kwargs)
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_loss = []
        self.seg_loss = []
        self.latent_reg_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)
        # return torch.optim.Adam([self.h, *list(self.backbone.parameters())[:2]], lr=self.lr)


class ImplicitNetMountedSegPrior(ImplicitNetSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetMountedSegPrior, self).__init__(*args, **kwargs)
        self.seg_backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.val_model_class = ImplicitNetMountedSegLatent

    def forward(self, coord, h):
        coord_ = coord.view((-1, coord.shape[-1]))
        # Tile h vector in a manner to allow for batch size > 1
        h_ = torch.tile(h, (1, coord_.shape[0] // coord.shape[0])).view((-1, h.shape[1]))

        c_enc = self.pos_enc(coord_)
        out_ = self.backbone((c_enc, h_))
        pred_recon_ = self.recon_head(out_)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        # out_ = self.seg_backbone((c_enc, out_.detach()))  TODO
        # pred_seg_ = self.seg_head(out_)
        # pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1])).moveaxis(-1, 1)
        pred_seg = torch.zeros((*coord.shape[:-1], 4)).moveaxis(-1, 1).cuda()
        # pred_recon = torch.zeros(coord.shape[:-1]).cuda()
        return pred_recon, pred_seg


class ImplicitNetMountedSegLatent(ImplicitNetSegLatent, ImplicitNetMountedSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetMountedSegLatent, self).__init__(*args, **kwargs)
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_loss = []
        self.seg_loss = []
        self.latent_reg_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)
        # return torch.optim.Adam([self.h, *list(self.backbone.parameters())[:2]], lr=self.lr)
