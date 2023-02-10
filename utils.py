from typing import Union, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBarBase, TQDMProgressBar
from monai.losses import ContrastiveLoss
import os
from pathlib import Path
import nibabel as nib


class ValProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, *_) -> None:
        self.main_progress_bar.initial = 0
        self.main_progress_bar.set_description(f"Validation epoch {trainer.current_epoch} / {trainer.max_epochs}")


def square_image(im: np.ndarray, seg: np.ndarray):
    """ Crop image such that height and weight are equal. Based on which dimension of the two is largest. """
    max_dim = np.argmax(im.shape[:2])
    if max_dim == 0:
        start_idx = (im.shape[0] - im.shape[1]) // 2
        cropped_im = im[start_idx: start_idx + im.shape[1]]
        cropped_seg = seg[start_idx: start_idx + seg.shape[1]]
    elif max_dim == 1:
        start_idx = (im.shape[1] - im.shape[0]) // 2
        cropped_im = im[:, start_idx: start_idx + im.shape[0]]
        cropped_seg = seg[:, start_idx: start_idx + seg.shape[0]]
    else:
        raise ValueError(f"The largest two dimensions should be the first two. Got {im.shape}")
    # assert cropped_im.shape[0] == cropped_im.shape[1]
    return cropped_im, cropped_seg


def normalize_image(im):
    """ Normalize image to range [0, 1] """
    min_, max_ = im.min(), im.max()
    im_ = (im - min_) / (max_ - min_)
    return im_


def draw_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Takes in a 2D image and a mask. Overlays the mask over the image.
    Mask is assumed to be a 2D array containing class indices (not a 1-hot vector).
    Background class is left transparent.
    If a 3D image is passed, it is assumed the last dimensions is the RGB dimension. """
    colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if len(image.shape) == 3:
        assert image.shape[-1] == 3
        image_rgb = image
    elif len(image.shape) == 2:
        image_rgb = np.stack((image, image, image), axis=-1)
    else:
        raise ValueError
    assert len(mask.shape) == 2
    mask = np.stack((mask, mask, mask), axis=-1)
    # Background not drawn
    for i in range(1, 4):
        c_arr = np.full_like(image_rgb, np.array(colors[i]))
        image_rgb = np.where(mask == i, c_arr, image_rgb)
    return image_rgb


def to_1hot(class_indices: torch.Tensor, num_class=4) -> torch.FloatTensor:
    seg = class_indices.to(torch.long).reshape((-1))
    seg_1hot = torch.zeros((*seg.shape, num_class), dtype=torch.float32, device=class_indices.device)
    seg_1hot[torch.arange(0, seg.shape[0], dtype=torch.long), seg] = 1
    seg_1hot = seg_1hot.reshape((*class_indices.shape, num_class)).moveaxis(-1, 1)
    return seg_1hot


def find_sax_ED_images(load_dir: Union[str, Path], num_cases: int = -1, get_bbox: bool = False, **kwargs):
    ims = []
    segs = []
    bboxes = []
    count = 0
    img_file = "sa_ED.nii.gz"
    seg_file = "seg_sa_ED.nii.gz"
    for parent, subdir, files in os.walk(str(load_dir)):
        if num_cases > 0 and count >= num_cases:
            break
        im_path = Path(parent) / img_file
        seg_path = Path(parent) / seg_file
        if not os.path.exists(im_path):
            continue
        if not os.path.exists(seg_path):
            continue
        ims.append(im_path)
        segs.append(seg_path)
        if get_bbox:
            seg = nib.load(seg_path).get_data()
            arg = np.argwhere(seg > 0)
            bbox = (arg.min(0), arg.max(0))
            bboxes.append(bbox)
        count += 1
    if num_cases > 0 and count != num_cases:
        raise ValueError(f"Did not find required amount of cases ({num_cases}) in directory: {load_dir}")
    return ims, segs, bboxes if get_bbox else None


def find_sax_images(load_dir: Union[str, Path], num_cases: int = -1, get_bbox: bool = False, **kwargs):
    ims = []
    segs = []
    bboxes = []
    count = 0
    img_file = "sa.nii.gz"
    seg_file = "seg_sa.nii.gz"
    for parent, subdir, files in os.walk(str(load_dir)):
        if num_cases > 0 and count >= num_cases:
            break
        im_path = Path(parent) / img_file
        seg_path = Path(parent) / seg_file
        if not os.path.exists(im_path):
            continue
        if not os.path.exists(seg_path):
            continue
        ims.append(im_path)
        segs.append(seg_path)
        if get_bbox:
            seg = nib.load(seg_path).get_data()
            arg = np.argwhere(seg > 0)
            bbox = (arg.min(0), arg.max(0))
            bboxes.append(bbox)
        count += 1
    if num_cases > 0 and count != num_cases:
        raise ValueError(f"Did not find required amount of cases ({num_cases}) in directory: {load_dir}")
    return ims, segs, bboxes if get_bbox else None


def contrastive_loss_all_elements(latents: torch.Tensor, neg_weight: float = 1.0, pos_weight: float = 1.0) \
        -> torch.Tensor:
    """ Pull elements within the same batch together, push other elements in the batch dimension away.
    :argument latents: Tensor of the shape [batch, positive_samples, vector_size]
    """
    assert len(latents.shape) == 3
    contr_loss = ContrastiveLoss()
    neg_samples, pos_samples, vec_size = latents.shape
    ref_idx = pos_samples//2
    target = latents[:, ref_idx]
    loss = torch.zeros((neg_samples, pos_samples))
    for i in range(neg_samples):
        for j in range(pos_samples):
            loss[i, j] = contr_loss(latents[i, j], target)
    return loss


# def contrastive_loss_random(latents: torch.Tensor, neg_weight: float = 1.0, pos_weight: float = 1.0, samples: int = 5) \
#         -> torch.Tensor:
#     """ Pull elements within the same batch together, push other elements in the batch dimension away.
#     :argument latents: Tensor of the shape [batch, positive_samples, vector_size]
#     """
#     assert len(latents.shape) == 3
#     contr_loss = ContrastiveLoss()
#     neg_samples, pos_samples, vec_size = latents.shape
#     ref_idx = pos_samples//2
#     target = latents[:, ref_idx]
#     loss = torch.zeros((neg_samples, pos_samples))
#     for i in range(neg_samples):
#         for j in range(pos_samples):
#             loss[i, j] = contr_loss(latents[i, j], target)
#     return loss
