import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBarBase, TQDMProgressBar
from monai.losses import ContrastiveLoss


class ValProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, *_) -> None:
        self.main_progress_bar.initial = 0
        self.main_progress_bar.set_description(f"Validation epoch {trainer.current_epoch}")


def draw_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    seg_1hot = seg_1hot.reshape((*class_indices.shape, num_class))
    return seg_1hot


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
