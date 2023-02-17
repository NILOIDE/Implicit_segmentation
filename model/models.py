import abc
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Any, Dict, Optional, List, Union
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

from data_loading.data_loader import AbstractDataset
from model.activation_functions import Sine, WIRE, Relu
from model.mlp import MLP, ResMLP, ResMLPHiddenCoords, MLPHiddenCoords, ReconstructionHead, SegmentationHead
from model.pos_encoding import PosEncodingNeRFOptimized, PosEncodingGaussian, PosEncodingNone
from utils import ValProgressBar, draw_mask_to_image, to_1hot, square_image, normalize_image


def process_params(**params: Dict[str, Any]):
    if params["activation"] == "periodic":
        activation = Sine
    elif params["activation"] == "relu":
        activation = Relu
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


class Abstract(pl.LightningModule):
    def __init__(self, *args, aug_num_parameters: int = 0, split_name="", **kwargs):
        super(Abstract, self).__init__()
        self.split_name = split_name
        self.latent_size = kwargs.get("latent_size")
        self.lr = float(kwargs.get("lr"))
        self.max_epochs = kwargs.get("max_epochs")

        self.dataset = kwargs.get("dataset")
        self.num_train_samples = 1
        self.num_coord_dims = kwargs.get("coord_dimensions")
        if self.dataset is not None:
            self.num_train_samples = len(self.dataset)
            self.num_coord_dims = self.dataset.sample_coords.shape[-1]
        if "coord_dimensions" not in kwargs:
            kwargs["num_coord_dims"] = self.num_coord_dims
        self.activation_class, self.pos_encoding_class, self.backbone_class = process_params(**kwargs)
        self.side_length = kwargs.get("side_length")
        self.h_init_std = kwargs.get("h_init_std", 0.1**4)
        self.h = nn.Parameter(torch.normal(0., self.h_init_std,
                                           (self.num_train_samples, self.latent_size - aug_num_parameters,)),
                              requires_grad=True)
        self.num_classes = 4
        self.pos_enc = self.pos_encoding_class(**kwargs)

        self.latent_reg = float(kwargs.get("latent_reg", 0.0))
        self.weight_reg = float(kwargs.get("weight_reg", 0.0))
        self.backbone_reg = self.weight_reg > 0.0

        self.x_holdout_rate = kwargs.get("x_holdout_rate", 1)
        self.y_holdout_rate = kwargs.get("y_holdout_rate", 1)
        self.z_holdout_rate = kwargs.get("z_holdout_rate", 1)
        self.t_holdout_rate = kwargs.get("t_holdout_rate", 1)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        self.l2_loss = nn.MSELoss()
        self.params = kwargs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reconstruction_criterion(self, pred_im, gt_im, background_mask=None, log=True):
        loss = self.bce_loss(pred_im, gt_im)
        if background_mask is not None:
            loss = loss * self.get_normalized_dist_map(background_mask)
        loss = loss.mean()
        if log:
            self.log(f"{self.split_name}/pixel_bce", loss.item(), prog_bar=True)
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
            self.log(f"{self.split_name}/bce_loss", loss_seg_bce.mean().item())
            self.log(f"{self.split_name}/dice_loss", loss_seg_dice.mean().item())

            with torch.no_grad():
                dice = 1 - self.dice_loss(pred_seg.round(), gt_seg_1hot).mean(non_class_dims)
                dice = dice.cpu().tolist()
            self.log(f"{self.split_name}/dice_BG", dice[0], prog_bar=True)
            self.log(f"{self.split_name}/dice_LV_Pool", dice[1], prog_bar=True)
            self.log(f"{self.split_name}/dice_LV_Myo", dice[2], prog_bar=True)
            self.log(f"{self.split_name}/dice_RV_Pool", dice[3], prog_bar=True)

        return loss_seg_dice_weighted + loss_seg_bce_weighted


class AbstractPrior(Abstract):
    def __init__(self, *args, val_dataset=None, **kwargs):
        kwargs = deepcopy(kwargs)
        kwargs["split_name"] = kwargs.get("split_name", "train")
        super(AbstractPrior, self).__init__(*args, **kwargs)
        self.save_hyperparameters(kwargs)

        self.dice_loss = DiceLoss(reduction="none")
        self.seg_class_weights = torch.tensor(kwargs.get("seg_class_weights"), dtype=torch.float32, device="cuda")

        self.val_dataset: AbstractDataset = val_dataset
        self.initial_val = kwargs.get("initial_val", False)
        self.log_interval = kwargs.get("log_interval")
        self.val_max_epochs = kwargs.get("fine_tune_max_epochs")
        self.val_interval = kwargs.get("val_interval")

        self.overall_best_score = 0
        self.overall_best_num_fine_tune_epochs = 0
        self.best_checkpoint_path = kwargs.get("best_checkpoint_path")

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

    def do_validation(self, epoch):
        self.eval()
        params = deepcopy(self.params)
        params["dataset"] = self.val_dataset
        params["dropout"] = 0.0
        val_model = self.val_model_class(parent_logger=self.logger,
                                         train_epoch=epoch,
                                         **params)
        state_dict = deepcopy(self.state_dict())
        del state_dict['h']
        miss_keys = val_model.load_state_dict(state_dict, strict=False)
        assert 'h' in miss_keys.missing_keys and len(miss_keys.missing_keys) == 1
        val_trainer = pl.Trainer(max_epochs=self.val_max_epochs, accelerator="gpu",
                                 enable_model_summary=False, logger=False,
                                 enable_checkpointing=False, callbacks=[ValProgressBar()])
        val_trainer.fit(val_model, train_dataloaders=DataLoader(self.val_dataset, batch_size=1, shuffle=False))
        scores_per_epoch = [(float(np.mean(val_model.history_dice_LV_Pool[i])) +
                            float(np.mean(val_model.history_dice_LV_Myo[i])) +
                            float(np.mean(val_model.history_dice_RV_Pool[i]))) / 3
                            for i in range(len(val_model.history_dice_LV_Pool))]
        best_num_fine_tune_epochs = int(np.argmax(scores_per_epoch))
        best_score = scores_per_epoch[best_num_fine_tune_epochs]
        self.log(f"val_best_score", best_score, prog_bar=True)
        self.log(f"val_best_num_fine_tune_epochs", float(best_num_fine_tune_epochs), prog_bar=True)
        if best_score > self.overall_best_score:
            self.overall_best_score = best_score
            self.overall_best_num_fine_tune_epochs = best_num_fine_tune_epochs
            torch.save(self.state_dict(), self.best_checkpoint_path)
        del val_trainer, val_model
        self.train()

    def on_fit_start(self):
        if self.initial_val:
            self.do_validation(self.current_epoch)

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval == 0:
            self.do_validation(self.current_epoch + 1)


class AbstractLatent(Abstract):
    def __init__(self, parent_logger=None, train_epoch=None, *args, **kwargs):
        kwargs = deepcopy(kwargs)
        kwargs["split_name"] = kwargs.get("split_name", "val")
        kwargs["lr"] = kwargs.get("fine_tune_lr")
        kwargs["log_interval"] = kwargs.get("fine_tune_log_interval")
        if kwargs["fine_tune_optimal_epochs"] > 0:
            kwargs["max_epochs"] = kwargs["fine_tune_optimal_epochs"]
        else:
            kwargs["max_epochs"] = kwargs["fine_tune_max_epochs"]
        super(AbstractLatent, self).__init__(*args, **kwargs)
        self.parent_logger = parent_logger
        self.train_epoch = train_epoch

        self.history_loss = {i: [] for i in range(self.max_epochs)}
        self.history_pixel_loss = {i: [] for i in range(self.max_epochs)}
        self.history_seg_loss = {i: [] for i in range(self.max_epochs)}
        self.history_reg_loss = {i: [] for i in range(self.max_epochs)}
        self.history_dice_BG = {i: [] for i in range(self.max_epochs)}
        self.history_dice_LV_Pool = {i: [] for i in range(self.max_epochs)}
        self.history_dice_LV_Myo = {i: [] for i in range(self.max_epochs)}
        self.history_dice_RV_Pool = {i: [] for i in range(self.max_epochs)}

    def on_fit_start(self):
        return

    def on_train_epoch_end(self):
        logger = self.parent_logger if self.parent_logger is not None else self.logger
        if self.current_epoch % self.log_interval == 0:
            # Logged to training trainer's logger
            if self.current_epoch in self.history_loss and self.history_loss[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_loss/{self.current_epoch}",
                                             np.mean(self.history_loss[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_pixel_loss and self.history_pixel_loss[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_pixel_loss/{self.current_epoch}",
                                             np.mean(self.history_pixel_loss[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_seg_loss and self.history_seg_loss[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_seg_loss/{self.current_epoch}",
                                             np.mean(self.history_seg_loss[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_reg_loss and self.history_reg_loss[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_reg_L2/{self.current_epoch}",
                                             np.mean(self.history_reg_loss[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_dice_BG and self.history_dice_BG[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_dice_BG/{self.current_epoch}",
                                             np.mean(self.history_dice_BG[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_dice_LV_Pool and self.history_dice_LV_Pool[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_dice_LV_Pool/{self.current_epoch}",
                                             np.mean(self.history_dice_LV_Pool[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_dice_LV_Myo and self.history_dice_LV_Myo[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_dice_LV_Myo/{self.current_epoch}",
                                             np.mean(self.history_dice_LV_Myo[self.current_epoch]),
                                             global_step=self.train_epoch)
            if self.current_epoch in self.history_dice_RV_Pool and self.history_dice_RV_Pool[self.current_epoch]:
                logger.experiment.add_scalar(f"{self.split_name}_dice_RV_Pool/{self.current_epoch}",
                                             np.mean(self.history_dice_RV_Pool[self.current_epoch]),
                                             global_step=self.train_epoch)

            # Draw images and save them to logger
            for sample_idx in range(len(self.dataset)):
                im = self.draw_image(sample_idx)
                logger.experiment.add_image(f"{self.split_name}_images_{self.current_epoch}",
                                            im,
                                            global_step=self.train_epoch)

    def on_fit_end(self):
        logger = self.parent_logger if self.parent_logger is not None else self.logger
        fig = self.draw_fit_plot(
            pixel_loss=[float(np.mean(self.history_pixel_loss[i])) for i in range(len(self.history_pixel_loss))],
            dice_BG=[float(np.mean(self.history_dice_BG[i])) for i in range(len(self.history_dice_BG))],
            dice_LV_Pool=[float(np.mean(self.history_dice_LV_Pool[i])) for i in range(len(self.history_dice_LV_Pool))],
            dice_LV_Myo=[float(np.mean(self.history_dice_LV_Myo[i])) for i in range(len(self.history_dice_LV_Myo))],
            dice_RV_Pool=[float(np.mean(self.history_dice_RV_Pool[i])) for i in range(len(self.history_dice_RV_Pool))],
            reg_loss=[float(np.mean(self.history_reg_loss[i])) for i in range(len(self.history_reg_loss))],
            title_prefix="Average ")
        logger.experiment.add_figure(f"{self.split_name}_loss_curve_average", fig, global_step=self.train_epoch)
        del fig

        if self.split_name == "test":
            for i in range(len(self.history_dice_BG[0])):
                fig = self.draw_fit_plot(
                    pixel_loss=[self.history_pixel_loss[t][i] for t in range(len(self.history_pixel_loss))],
                    dice_BG=[self.history_dice_BG[t][i] for t in range(len(self.history_dice_BG))],
                    dice_LV_Pool=[self.history_dice_LV_Pool[t][i] for t in range(len(self.history_dice_LV_Pool))],
                    dice_LV_Myo=[self.history_dice_LV_Myo[t][i] for t in range(len(self.history_dice_LV_Myo))],
                    dice_RV_Pool=[self.history_dice_RV_Pool[t][i] for t in range(len(self.history_dice_RV_Pool))],
                    reg_loss=[self.history_reg_loss[t][i] for t in range(len(self.history_reg_loss))],
                    title_prefix=f"Sample {i}'s ")
                logger.experiment.add_figure(f"{self.split_name}_loss_curve_sample_{i}", fig, global_step=self.train_epoch)
                if self.num_coord_dims == 4 and self.dataset is not None:
                    vid = self.draw_time_video(self.dataset.im_paths[i], self.dataset.seg_paths[i])
                    logger.experiment.add_video(f"{self.split_name}_end_video_sample_{i}", vid[None], fps=10, global_step=self.train_epoch)
                    del vid

        else:
            if self.num_coord_dims == 4 and self.dataset is not None:
                vid = self.draw_time_video(self.dataset.im_paths[0], self.dataset.seg_paths[0])
                logger.experiment.add_video(f"{self.split_name}_end_video", vid[None], fps=10,
                                            global_step=self.train_epoch)
                del vid

    def do_logging(self,
                   loss_pixel: Optional[float] = None,
                   loss_seg: Optional[float] = None,
                   dice_BG: Optional[float] = None,
                   dice_LV_Pool: Optional[float] = None,
                   dice_LV_Myo: Optional[float] = None,
                   dice_RV_Pool: Optional[float] = None,
                   loss_reg: Optional[float] = None,
                   **kwargs):

        # Logged only to progress bar
        if loss_pixel is not None:
            self.log(f"{self.split_name}_pixel_L2", loss_pixel, prog_bar=True, on_step=True)
            self.history_pixel_loss[self.current_epoch].append(loss_pixel)
        if loss_seg is not None:
            self.log(f"{self.split_name}_seg_loss", loss_seg, prog_bar=True, on_step=True)
            self.history_seg_loss[self.current_epoch].append(loss_seg)
        if loss_reg is not None:
            self.log(f"{self.split_name}_reg_loss", loss_reg, prog_bar=True, on_step=True)
            self.history_reg_loss[self.current_epoch].append(loss_reg)
        if dice_BG is not None:
            self.log(f"{self.split_name}_dice_BG", dice_BG, prog_bar=True, on_step=True)
            self.history_dice_BG[self.current_epoch].append(dice_BG)
        if dice_LV_Pool is not None:
            self.log(f"{self.split_name}_dice_LV_Pool", dice_LV_Pool, prog_bar=True, on_step=True)
            self.history_dice_LV_Pool[self.current_epoch].append(dice_LV_Pool)
        if dice_LV_Myo is not None:
            self.log(f"{self.split_name}_dice_LV_Myo", dice_LV_Myo, prog_bar=True, on_step=True)
            self.history_dice_LV_Myo[self.current_epoch].append(dice_LV_Myo)
        if dice_RV_Pool is not None:
            self.log(f"{self.split_name}_dice_RV_Pool", dice_RV_Pool, prog_bar=True, on_step=True)
            self.history_dice_RV_Pool[self.current_epoch].append(dice_RV_Pool)

    def draw_image(self, sample_idx):
        t = 0  # ED frame is t index 0
        gt_im = nib.load(self.dataset.im_paths[sample_idx]).dataobj[..., t]
        gt_seg = nib.load(self.dataset.seg_paths[sample_idx]).dataobj[..., t]
        gt_im, gt_seg = square_image(gt_im, gt_seg)
        gt_im = normalize_image(gt_im)
        gt_im = cv2.resize(gt_im, self.side_length[:2])
        gt_seg = cv2.resize(gt_seg, self.side_length[:2], interpolation=cv2.INTER_NEAREST)

        gt_2D_ims = [gt_im[..., i] for i in range(gt_im.shape[-1])]
        gt_2D_segs = [gt_seg[..., i] for i in range(gt_seg.shape[-1])]
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

    def draw_fit_plot(self,
                      pixel_loss: List[float],
                      dice_BG: List[float],
                      dice_LV_Pool: List[float],
                      dice_LV_Myo: List[float],
                      dice_RV_Pool: List[float],
                      reg_loss: List[float],
                      title_prefix: str = "",
                      ):
        print("Generating fit plot...")
        fig, ax = plt.subplots(1, 3, )
        fig.set_figwidth(fig.get_figheight() * 3)
        ax[0].plot(list(range(len(pixel_loss))), pixel_loss, label="Image L2 Loss")
        ax[0].set_title(title_prefix + "Image Reconstruction Loss")
        ax[0].legend(loc="upper right")
        ax[1].plot(list(range(len(dice_BG))), dice_BG, label="Background")
        ax[1].plot(list(range(len(dice_LV_Pool))), dice_LV_Pool, label="LV Pool")
        ax[1].plot(list(range(len(dice_LV_Myo))), dice_LV_Myo, label="LV Myo")
        ax[1].plot(list(range(len(dice_RV_Pool))), dice_RV_Pool, label="RV Pool")
        ax[1].set_ylim(0.5, 1.0)
        ax[1].yaxis.set_ticks(np.arange(0.5, 1.0, 0.025))
        ax[1].grid("x")
        ax[1].set_title(title_prefix + "Segmentation Dice")
        ax[1].legend(loc="upper right")
        ax[2].plot(list(range(len(reg_loss))), reg_loss, label="Latent L2")
        ax[2].set_title(title_prefix + "Latent L2 Loss")
        ax[2].legend(loc="upper right")
        return fig

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
    def evaluate(self, coord_arr: torch.Tensor, h_vector: torch.Tensor, as_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        assert coord_arr.shape[-1] == self.num_coord_dims
        assert len(h_vector.shape) == 1
        coord_arr = coord_arr.to(self.device)
        pred_im, pred_seg = self.forward(coord_arr[None], h_vector[None])
        if as_numpy:
            pred_im, pred_seg = pred_im.cpu().numpy(), pred_seg.cpu().numpy()
        return pred_im[0], pred_seg[0]

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
        self.backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.recon_head = ReconstructionHead(self.backbone.out_size, **kwargs)
        self.seg_head = SegmentationHead(self.backbone.out_size, self.num_classes)
        self.val_model_class = ImplicitNetSegLatent

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
        if self.current_epoch % self.log_interval == 0 and sample_idx.item() < 3:
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

    def training_step(self, batch):
        coord, image, sample_idx, seg, _ = batch
        coord_ = coord[..., torch.arange(0, coord.shape[-2], self.z_holdout_rate), :]
        image_ = image[..., torch.arange(0, image.shape[-1], self.z_holdout_rate)]
        seg_ = seg[..., torch.arange(0, seg.shape[-1], self.z_holdout_rate)]
        pred_im, pred_seg = self.forward(coord_, self.h[sample_idx])

        loss = 0
        # Pixel regression loss
        background = pred_seg[:, 0].detach() > 0.5
        loss_pixel = self.reconstruction_criterion(pred_im, image_, log=False)#, background_mask=background)
        loss += loss_pixel

        # Segmentation dice (not trained on, assume not available at test time). Not added to overall loss here.
        with torch.no_grad():
            dice = (1 - self.dice_loss(pred_seg.round(), to_1hot(seg_))).mean(0).squeeze().detach().cpu().tolist()

        # Regularization loss for h (and optionally network weights)
        loss_reg = None
        loss_reg = self.regularization_criterion(self.h[sample_idx], log=False)
        loss += loss_reg

        progress = dict(loss_pixel=loss_pixel.item(),
                        dice_BG=dice[0],
                        dice_LV_Pool=dice[1],
                        dice_LV_Myo=dice[2],
                        dice_RV_Pool=dice[3],
                        loss_reg=loss_reg.item(),
                        )
        del loss_pixel, dice, loss_reg
        self.do_logging(**progress)
        return loss


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

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)
        # return torch.optim.Adam([self.h, *list(self.backbone.parameters())[:2]], lr=self.lr)
