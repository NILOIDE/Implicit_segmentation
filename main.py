from typing import Tuple, Optional
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetPrior, ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior
from data_loading.data_loader import SAX3D, SAX3D_Seg, SAX3D_Seg_test, SAX3D_test, SAX3D_Seg_padded, \
    SAX3D_Seg_WholeImage


@dataclass
class Params:
    initial_val: bool = False
    pos_encoding: str = "nerf"  # nerf, none, gaussian
    num_frequencies: Tuple[int, ...] = (4, 4, 4)
    # num_frequencies: Tuple[int, ...] = (128,)
    freq_scale: float = 1.
    latent_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.00
    num_hidden_layers: int = 8
    side_length: Tuple[int, int, Optional[int]] = (100, 100, -1)
    heart_pad: int = 10
    batch_size: int = 1
    max_epochs: int = 2001
    train_log_interval: int = 1
    val_max_epochs: int = 201  # Lightning is faster doing (2000 epochs x 1 batch) than (1 epoch x 2000 batches)
    val_interval: int = 1
    val_log_interval: int = 20
    val_x_holdout_rate: int = 1  # Height
    val_y_holdout_rate: int = 1  # Width
    val_z_holdout_rate: int = 1  # Slices
    val_t_holdout_rate: int = 3  # Time
    seg_class_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    lr: float = 1e-4
    fine_tune_lr: float = 1e-3
    latent_reg: float = 1e-4
    weight_reg: float = 1e-4
    activation: str = "wire"  # periodic, relu
    skip_connections: bool = True
    input_coord_to_all_layers: bool = False
    model_type: str = "shared"  # image_only, shared, separate, mounted


def main(params, exp_name=''):
    # dataset = SAX3D_Seg(**params.__dict__)
    dataset = SAX3D_Seg_WholeImage(**params.__dict__)
    if params.model_type == "separate":
        model = ImplicitNetSeparateSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "shared":
        model = ImplicitNetSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "mounted":
        model = ImplicitNetMountedSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "image_only":
        dataset = SAX3D(**params.__dict__)
        model = ImplicitNetPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    else:
        raise NotImplementedError
    # model = model.load_from_checkpoint(r"C:\Users\nilst\Documents\Implicit_segmentation\logs\ModelTest\20230123-182038_SAX3D_Seg_WholeImage_IMG_ONLY\checkpoint\epoch=79-step=6720.ckpt")
    train_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    root_dir = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\logs") / "test"
    if exp_name:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(dataset.__class__.__name__)}'
    print(root_dir)
    os.makedirs(str(root_dir), exist_ok=True)
    ckpt_saver = ModelCheckpoint(save_top_k=1, dirpath=root_dir / "checkpoint", monitor="step", mode="max")
    trainer = pl.Trainer(max_epochs=params.max_epochs, log_every_n_steps=50, accelerator="gpu", default_root_dir=root_dir, callbacks=[ckpt_saver])
    start = datetime.now()
    trainer.fit(model, train_dataloaders=train_dataloader)
    print("Elapsed time:", datetime.now() - start)


if __name__ == '__main__':
    # for enc, scale in [("gaussian", 5.0), ("gaussian", 0.2)]:
    #     print("Testing:", "Enc:", enc, "Scale", scale)
    #     params = Params(pos_encoding=enc, freq_scale=scale)
    #     main(params, exp_name=f"NumLayers_{params.num_hidden_layers}_encType_{params.pos_encoding}_freqs_{params.num_frequencies}_GausScale_{params.freq_scale}_actFunc_{params.activation}_modelType_{params.model_type}")
    #     for scale in [0.5, 1.0, 5.]:
    #         for freqs in [(64,), (128,), (256,)]:
    #             print("Testing: Layers:", lays, "Gauss scale:", scale, "Freqs:", freqs)
    #             start = datetime.now()
    #             params = Params(num_hidden_layers=lays, freq_scale=scale, num_frequencies=freqs)
    #             main(params, exp_name=f"NumLayers_{lays}_GaussScale_{scale}_GaussFeats_{freqs}")
    #             print("Elapsed time:", datetime.now() - start)
    params = Params()
    main(params, exp_name=f"NumLayers_{params.num_hidden_layers}_hidSize_{params.hidden_size}_enc_{params.pos_encoding}_hidCoords_{params.input_coord_to_all_layers}_modelType_{params.model_type}")
    # main(p, exp_name=f"SkipCon_{p.skip_connections}_NumLayers_{p.num_hidden_layers}_posEnc_{p.pos_encoding}_hiddenCoords_{p.input_coord_to_all_layers}")
    # optimize_latent(**Params().__dict__)
