import abc
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Any, Dict, Optional
import numpy as np
import pytorch_lightning as pl
from monai.losses import DiceLoss

import logging
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

from data_loading.data_loader import SAX3D_test, SAX3D_Seg_test
from model.activation_functions import Sine
from model.mlp import MLP, ResMLP, ResMLPHiddenCoords, MLPHiddenCoords, ReconstructionHead, SegmentationHead
from model.pos_encoding import PosEncodingNeRFOptimized, PosEncodingGaussian, PosEncodingNone
from utils import ValProgressBar, draw_mask_to_image, to_1hot


def process_params(params: Dict[str, Any]):
    if params["activation"] == "periodic":
        activation = Sine
    elif params["activation"] == "relu":
        activation = nn.ReLU
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
    print(f"Model: {model.__name__}, Activation_func: {activation.__name__}, Pos_enc: {str(pos_encoding(**params))}")
    return activation, pos_encoding, model


class AbstractPrior(pl.LightningModule):
    def __init__(self, *args, num_train_samples: int = 0, aug_num_parameters: int = 0, **kwargs):
        super(AbstractPrior, self).__init__()
        self.save_hyperparameters(kwargs)
        self.activation_class, self.pos_encoding_class, self.backbone_class = process_params(kwargs)
        self.latent_size = kwargs.get("latent_size")
        self.lr = kwargs.get("lr")
        self.num_train_samples = num_train_samples
        self.h = nn.Parameter(torch.normal(0., 0.1**2, (self.num_train_samples, self.latent_size - aug_num_parameters,)), requires_grad=True)

        self.val_epochs = kwargs.get("val_max_epochs")
        self.val_interval = kwargs.get("val_interval")
        self.params = kwargs
        self.split_name = 'train'
        self.dice_loss = None
        self.bce_loss = None

    @abc.abstractmethod
    def val_dataset(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reconstruction_criterion(self, pred_im, gt_im, log=True):
        loss = self.mse_loss(pred_im, gt_im).mean()
        if log:
            self.log(f"{self.split_name}/pixel_L2", loss.item())
        return loss

    def regularization_criterion(self, h, log=True):
        loss = self.l2_loss(h, torch.zeros_like(h))
        if log:
            self.log(f"{self.split_name}/latent_L2", loss.item())
        if self.backbone_reg:
            backbone_l2 = sum((p*p).sum() for p in self.backbone.parameters())
            if log:
                self.log(f"{self.split_name}/backbone_L2", loss.item())
            loss += backbone_l2
        return loss * self.weight_reg

    def segmentation_criterion(self, pred_seg, gt_seg, log=True):
        gt_seg_1hot = to_1hot(gt_seg)
        loss_seg_bce = self.bce_loss(pred_seg, gt_seg_1hot).mean()
        loss_seg_dice = self.dice_loss(pred_seg, gt_seg_1hot).mean()
        if log:
            self.log(f"{self.split_name}/seg_bce", loss_seg_bce.item())
            self.log(f"{self.split_name}/seg_dice", loss_seg_dice.item())
        return loss_seg_bce + loss_seg_dice

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.val_interval == 0:
            self.eval()
            params = deepcopy(self.params)
            if "dropout" in params:
                params["dropout"] = 0.0
            val_model = self.val_model_class(parent_logger=self.logger, train_epoch=self.current_epoch + 1,
                                             **params)
            state_dict = deepcopy(self.state_dict())
            del state_dict['h']
            miss_keys = val_model.load_state_dict(state_dict, strict=False)
            assert 'h' in miss_keys.missing_keys and len(miss_keys.missing_keys) == 1
            val_model.h = nn.Parameter(torch.normal(0., 1e-4**2, val_model.h.shape))
            val_trainer = pl.Trainer(max_epochs=self.val_epochs, accelerator="gpu",
                                     enable_model_summary=False, logger=False,
                                     enable_checkpointing=False, callbacks=[ValProgressBar()])
            val_trainer.fit(val_model, train_dataloaders=DataLoader(self.val_dataset, batch_size=1, shuffle=True))
            self.train()


class AbstractLatent(pl.LightningModule):
    def __init__(self, parent_logger=None, train_epoch=None, *args, **kwargs):
        super(AbstractLatent, self).__init__(*args, **kwargs)
        self.h = nn.Parameter(torch.normal(0., 1e-4**2, (1, self.latent_size)), requires_grad=True)
        self.parent_logger = parent_logger
        self.train_epoch = train_epoch
        self.split_name = 'val'
        self.z_holdout_rate = 2
        self.t_holdout_rate = 3

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)

    def on_train_epoch_end(self):
        return

    def val_logging(self, image: torch.Tensor,
                    seg: Optional[torch.Tensor] = None,
                    loss_pixel: Optional[float] = None,
                    loss_seg: Optional[float] = None,
                    loss_reg: Optional[float] = None):
        if self.parent_logger is None or self.global_step % self.val_log_interval != 0:
            return
        if loss_pixel is not None:
            self.parent_logger.experiment.add_scalar(f"val_pixel_L2/{self.global_step}", loss_pixel,
                                                     global_step=self.train_epoch)
        if loss_seg is not None:
            self.parent_logger.experiment.add_scalar(f"val_seg_loss/{self.global_step}", loss_seg,
                                                     global_step=self.train_epoch)
        if loss_reg is not None:
            self.parent_logger.experiment.add_scalar(f"val_reg_L2/{self.global_step}", loss_reg,
                                                     global_step=self.train_epoch)
        im = self.draw_image(image, seg)
        self.parent_logger.experiment.add_image(f"val_images_{self.global_step}", im, global_step=self.train_epoch)

    def draw_image(self, *args):
        pass

    def evaluate_volume_slices(self, out_shape: Tuple[int, ...]):
        coords = torch.meshgrid(torch.arange(out_shape[0], dtype=torch.float32),
                                torch.arange(out_shape[1], dtype=torch.float32),
                                torch.arange(out_shape[2], dtype=torch.float32))
        coord_arr = torch.stack([c / out_shape[i] - .5 for i, c in enumerate(coords)], dim=-1)
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
            dim1 = coords[count] / out_shape[count] - .5
            count += 1
        else:
            dim1 = torch.full_like(coords[0], dim1_value)
        if dim2_value is None:
            dim2 = coords[count] / out_shape[count] - .5
            count += 1
        else:
            dim2 = torch.full_like(coords[1], dim2_value)
        if dim3_value is None:
            dim3 = coords[count] / out_shape[count] - .5
            count += 1
        else:
            dim3 = torch.full_like(coords[2], dim3_value)
        if time_value is None:
            dimt = coords[count] / out_shape[count]
            count += 1
        else:
            dimt = torch.full_like(coords[3], time_value)

        if self.pos_encoding_class.in_features == 4:
            dims = (dim1, dim2, dim3, dimt)
        elif self.pos_encoding_class.in_features == 3:
            if time_value is None:
                dims = (dim1, dim2, dim3)
            else:
                dims = (dim1, dim2, dimt)
        else:
            raise ValueError("There should be at least 3 dims")
        coord_arr = torch.stack(dims, dim=-1)
        return self.evaluate(coord_arr, self.h[0])


class ImplicitNetPrior(AbstractPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetPrior, self).__init__(**kwargs)
        self.save_hyperparameters(kwargs)
        self.pos_enc = self.pos_encoding_class(**kwargs)
        self.backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.recon_head = ReconstructionHead(self.backbone.out_size, **kwargs)
        self.mse_loss = nn.MSELoss(reduction="none")
        self.l2_loss = nn.MSELoss()
        self.backbone_reg = False
        self.weight_reg = kwargs.get("weight_reg")

        self.val_dataset = SAX3D_test(**kwargs)
        self.val_epochs = kwargs.get("val_max_epochs")
        self.val_interval = kwargs.get("val_interval")
        self.val_model_class = ImplicitNetLatent
        self.params = kwargs

    def forward(self, coord, h):
        coord_ = coord.view((-1, 3))
        h = torch.tile(h, (coord_.shape[0], 1))

        c_enc = self.pos_enc(coord_)
        out = self.backbone((c_enc, h))
        pred_ = self.recon_head(out)
        pred = pred_.view(coord.shape[:-1])
        return pred

    def training_step(self, batch):
        coord, image, sample_idx = batch
        pred_im = self.forward(coord, sample_idx)
        loss = self.reconstruction_criterion(pred_im, image)
        loss += self.regularization_criterion(self.h[sample_idx])
        return loss

    @torch.no_grad()
    def evaluate(self, coord_arr: torch.Tensor, h_vector: torch.Tensor) -> np.array:
        assert coord_arr.shape[-1] == 3
        assert len(h_vector.shape) == 1
        pred = self.forward(coord_arr[None].cuda(), h_vector[None].cuda())
        return pred[0].cpu().numpy()


class ImplicitNetLatent(AbstractLatent, ImplicitNetPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetLatent, self).__init__(*args, **kwargs)
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_loss = []

    def on_fit_end(self):
        if self.parent_logger is not None:
            fig = plt.figure()
            fig.add_subplot()
            fig.axes[0].plot(list(range(len(self.train_loss))), self.train_loss)
            self.parent_logger.experiment.add_figure("val_loss_curve", fig, global_step=self.train_epoch)
            del fig

    def training_step(self, batch):
        coord, image, sample_idx = batch
        coord_ = coord[..., torch.tensor(list(range(0, coord.shape[-2], self.z_holdout_rate)), dtype=torch.long), :]
        image_ = image[..., torch.tensor(list(range(0, image.shape[-1], self.z_holdout_rate)), dtype=torch.long)]
        pred_im = self.forward(coord_, sample_idx)

        loss_pixel = self.reconstruction_criterion(pred_im, image_, log=False)
        self.train_loss.append(loss_pixel.item())
        loss = loss_pixel
        loss_reg = self.regularization_criterion(self.h[sample_idx], log=False)
        loss += loss_reg

        self.val_logging(image[0], loss_pixel=loss_pixel.item(), loss_reg=loss_reg.item())
        return loss.mean()

    def draw_image(self, gt_im, *args, **kwargs):
        gt_2D_ims = [gt_im[..., i].cpu().numpy() for i in range(gt_im.shape[-1])]
        gt_im_row = []
        for i, og_im in enumerate(gt_2D_ims):
            og_im_rgb = np.stack((og_im, og_im, og_im), axis=-1)
            if i % self.z_holdout_rate == 0:
                og_im_rgb[[0]*og_im_rgb.shape[0], list(range(og_im_rgb.shape[1]))] = np.array((0, 1, 0))
            gt_im_row.append(og_im_rgb)
            gt_im_row.append(np.zeros_like(og_im_rgb))

        pred_im = self.evaluate_volume_slices((*gt_im.shape[:2], gt_im.shape[2] * 2 - 1))
        pred_im_row = [np.stack([pred_im[..., i]] * 3, axis=-1) for i in range(pred_im.shape[-1])]

        pred_im_row = np.concatenate(pred_im_row, axis=1)
        gt_im_row = np.concatenate(gt_im_row, axis=1)
        im = np.concatenate((gt_im_row, pred_im_row), axis=0)
        return im.transpose((2, 0, 1))


class ImplicitNetSegPrior(ImplicitNetPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetSegPrior, self).__init__(*args, **kwargs)
        self.num_classes = 4
        self.seg_head = SegmentationHead(self.backbone.out_size, self.num_classes)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.val_dataset = SAX3D_Seg_test(**kwargs)
        self.val_model_class = ImplicitNetSegLatent

    def forward(self, coord, h):
        coord_ = coord.view((-1, 3))
        h = torch.tile(h, (coord_.shape[0], 1))

        c_enc = self.pos_enc(coord_)
        out = self.backbone((c_enc, h))
        pred_recon_ = self.recon_head(out)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        pred_seg_ = self.seg_head(out)
        pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1]))
        return pred_recon, pred_seg

    def training_step(self, batch):
        coord, image, sample_idx, seg, aug_params = batch
        h = torch.cat((self.h[sample_idx], aug_params), dim=1)
        pred_im, pred_seg = self.forward(coord, h)
        loss = self.reconstruction_criterion(pred_im, image)
        loss += self.segmentation_criterion(pred_seg, seg)
        loss += self.regularization_criterion(self.h[sample_idx])
        return loss

    @torch.no_grad()
    def evaluate(self, coord_arr: torch.Tensor, h_vector: torch.Tensor) -> np.array:
        assert coord_arr.shape[-1] == 3
        assert len(h_vector.shape) == 1
        coord_arr = coord_arr.to(self.device)
        pred_im, pred_seg = self.forward(coord_arr[None], h_vector[None])
        return pred_im[0].cpu().numpy(), pred_seg[0].cpu().numpy()


class ImplicitNetSegLatent(AbstractLatent, ImplicitNetSegPrior):
    def __init__(self, parent_logger=None, train_epoch=None, *args, **kwargs):
        super(ImplicitNetSegLatent, self).__init__(*args, **kwargs)
        self.parent_logger = parent_logger
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_epoch = train_epoch
        self.train_loss = []
        self.seg_loss = []
        self.latent_reg_loss = []
        self.batch = None

    def on_fit_end(self):
        if self.parent_logger is not None:
            fig, ax = plt.subplots(1, 3,)
            fig.set_figwidth(fig.get_figheight()*3)
            ax[0].plot(list(range(len(self.train_loss))), self.train_loss, label="Image L2 Loss")
            ax[0].legend(loc="upper right")
            ax[1].plot(list(range(len(self.seg_loss))), self.seg_loss, label="Seg Dice Loss")
            ax[1].legend(loc="upper right")
            ax[2].plot(list(range(len(self.latent_reg_loss))), self.latent_reg_loss, label="Latent L2 loss")
            ax[2].legend(loc="upper right")
            self.parent_logger.experiment.add_figure("val_loss_curve", fig, global_step=self.train_epoch)
            del fig

    def training_step(self, batch):
        coord, image, sample_idx, seg, _ = batch
        coord_ = coord[..., torch.arange(0, coord.shape[-2], self.z_holdout_rate), :]
        image_ = image[..., torch.arange(0, image.shape[-1], self.z_holdout_rate)]
        seg_ = seg[..., torch.arange(0, seg.shape[-1], self.z_holdout_rate)]
        pred_im, pred_seg = self.forward(coord_, self.h)

        # Pixel regression loss
        loss_pixel = self.reconstruction_criterion(pred_im, image_, log=False)
        self.train_loss.append(loss_pixel.item())
        loss = loss_pixel

        # Segmentation loss (not trained on, assume not available at test time). Not added to overall loss here.
        loss_seg = self.dice_loss(pred_seg, to_1hot(seg_))
        self.seg_loss.append(loss_seg.item())
        # loss += loss_seg

        # Regularization loss for h (and optionally network weights)
        loss_reg = None
        loss_reg = self.regularization_criterion(self.h[sample_idx], log=False)
        self.latent_reg_loss.append(loss_reg.item())
        loss += loss_reg

        self.val_logging(image[0], seg=seg[0], loss_pixel=loss_pixel.item(), loss_seg=loss_seg.item(), loss_reg=loss_reg.item())
        return loss.mean()

    def draw_image(self, gt_im, gt_seg):
        gt_2D_ims = [gt_im[..., i].cpu().numpy() for i in range(gt_im.shape[-1])]
        gt_2D_segs = [gt_seg[..., i].cpu().numpy() for i in range(gt_seg.shape[-1])]
        gt_im_row = []
        gt_seg_row = []
        for i, (og_im, og_seg) in enumerate(zip(gt_2D_ims, gt_2D_segs)):
            og_im_rgb = np.stack((og_im, og_im, og_im), axis=-1)
            if i % self.z_holdout_rate == 0:
                og_im_rgb[[0]*og_im_rgb.shape[0], list(range(og_im_rgb.shape[1]))] = np.array((0, 1, 0))
            gt_im_row.append(og_im_rgb)
            gt_im_row.append(np.zeros_like(og_im_rgb))
            masked_im = draw_mask_to_image(og_im, og_seg)
            gt_seg_row.append(masked_im)
            gt_seg_row.append(np.zeros_like(masked_im))

        pred_im, pred_seg = self.evaluate_volume_slices((*gt_im.shape[:2], gt_im.shape[2]*2))
        pred_im_row = [np.stack([pred_im[..., i]]*3, axis=-1) for i in range(pred_im.shape[-1])]
        pred_seg_row = [draw_mask_to_image(pred_im[..., i],
                                           np.argmax(pred_seg[..., i, :], axis=-1))
                        for i in range(pred_im.shape[-1])]

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

    def forward(self, coord, sample_idx=None):
        if sample_idx is None:
            sample_idx = torch.zeros((coord.shape[0],), dtype=torch.long)
        coord_ = coord.view((-1, 3))
        h = torch.tile(self.h[sample_idx], (coord_.shape[0], 1))

        c_enc = self.pos_enc(coord_)
        out_recon_ = self.backbone((c_enc, h))
        pred_recon_ = self.recon_head(out_recon_)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        out_seg_ = self.seg_backbone((c_enc, h))
        pred_seg_ = self.seg_head(out_seg_)
        pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1]))
        return pred_recon, pred_seg


class ImplicitNetSeparateSegLatent(ImplicitNetSegLatent, ImplicitNetSeparateSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetSeparateSegLatent, self).__init__(*args, **kwargs)
        assert self.num_train_samples == 1
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
        kwargs = deepcopy(kwargs)
        kwargs["num_hidden_layers"] = 4
        self.backbone = self.backbone_class(self.pos_enc.out_dim, self.latent_size, self.activation_class, **kwargs)
        self.val_model_class = ImplicitNetMountedSegLatent

    def forward(self, coord, sample_idx=None):
        if sample_idx is None:
            sample_idx = torch.zeros((coord.shape[0],), dtype=torch.long)
        coord_ = coord.view((-1, 3))
        h = torch.tile(self.h[sample_idx], (coord_.shape[0], 1))

        c_enc = self.pos_enc(coord_)
        out_ = self.seg_backbone((c_enc, h))
        pred_seg_ = self.seg_head(out_)
        pred_seg = pred_seg_.view((*coord.shape[:-1], pred_seg_.shape[-1]))
        out_ = self.backbone((c_enc, out_))
        pred_recon_ = self.recon_head(out_)
        pred_recon = pred_recon_.view(coord.shape[:-1])
        return pred_recon, pred_seg


class ImplicitNetMountedSegLatent(ImplicitNetSegLatent, ImplicitNetMountedSegPrior):
    def __init__(self, *args, **kwargs):
        super(ImplicitNetMountedSegLatent, self).__init__(*args, **kwargs)
        assert self.num_train_samples == 1
        self.val_log_interval = kwargs.get("val_log_interval")
        self.lr = kwargs.get("fine_tune_lr")
        self.train_loss = []
        self.seg_loss = []
        self.latent_reg_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam([self.h], lr=self.lr)
        # return torch.optim.Adam([self.h, *list(self.backbone.parameters())[:2]], lr=self.lr)
