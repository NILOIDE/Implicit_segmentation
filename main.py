from typing import Tuple, Optional
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetPrior, ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior
from data_loading.data_loader import SAX3D, SAX3D_Seg, SAX3D_Seg_test, SAX3D_test, SAX3D_Seg_padded


@dataclass
class Params:
    initial_val: bool = False
    pos_encoding: str = "gaussian"  # nerf, none, gaussian
    # num_frequencies: Tuple[int, ...] = (16,16,3)
    num_frequencies: Tuple[int, ...] = (64,)
    freq_scale: float = .1
    latent_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.00
    num_hidden_layers: int = 4
    side_length: Tuple[int, int, Optional[int]] = (64, 64, -1)
    heart_pad: int = 10
    batch_size: int = 1
    max_epochs: int = 200
    val_max_epochs: int = 2001
    val_interval: int = 20
    val_log_interval: int = 200
    lr: float = 1e-4
    fine_tune_lr: float = 1e-3
    weight_reg: float = 1e-5
    activation: str = "periodic"  # periodic, relu
    skip_connections: bool = True
    input_coord_to_all_layers: bool = True
    model_type: str = "shared"  # image_only, shared, separate, mounted


def main(params, exp_name=''):
    dataset = SAX3D_Seg(**params.__dict__)
    if params.model_type == "separate":
        model = ImplicitNetSeparateSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params **params.__dict__)
    elif params.model_type == "shared":
        model = ImplicitNetSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "mounted":
        model = ImplicitNetMountedSegPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "image_only":
        dataset = SAX3D(**params.__dict__)
        model = ImplicitNetPrior(num_train_samples=len(dataset), aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    else:
        raise NotImplementedError
    # model = model.load_from_checkpoint(r"C:\Users\nilst\Documents\Implicit_segmentation\logs\test\20230110-204524_SAX3D_Seg\checkpoint\epoch=79-step=1680.ckpt")
    train_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    root_dir = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\logs") / "test"
    if exp_name:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(dataset.__class__.__name__)}'
    print(root_dir)
    os.makedirs(str(root_dir), exist_ok=True)
    ckpt_saver = ModelCheckpoint(save_top_k=1, dirpath=root_dir / "checkpoint", monitor="step", mode="max")
    trainer = pl.Trainer(max_epochs=params.max_epochs, log_every_n_steps=50, accelerator="gpu", default_root_dir=root_dir, callbacks=[ckpt_saver])
    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    # for lays in [4, 8]:
    #     for hid_size in [512, 256]:
    #         for freq in [(256,), (128,), (64,)]:
    #             params = Params(num_hidden_layers=lays, hidden_size=hid_size, latent_size=hid_size, num_frequencies=freq)
    #             main(params, exp_name=f"NumLayers_{params.num_hidden_layers}_GaussScale_{params.freq_scale}_GaussFeats_{params.num_frequencies}_hidSize_{params.hidden_size}")

    #     for scale in [0.5, 1.0, 5.]:
    #         for freqs in [(64,), (128,), (256,)]:
    #             print("Testing: Layers:", lays, "Gauss scale:", scale, "Freqs:", freqs)
    #             start = datetime.now()
    #             params = Params(num_hidden_layers=lays, freq_scale=scale, num_frequencies=freqs)
    #             main(params, exp_name=f"NumLayers_{lays}_GaussScale_{scale}_GaussFeats_{freqs}")
    #             print("Elapsed time:", datetime.now() - start)
    params = Params()
    main(params, exp_name=f"NumLayers_{params.num_hidden_layers}_posEnc_{params.pos_encoding}_hidSize_{params.hidden_size}")
    # main(p, exp_name=f"SkipCon_{p.skip_connections}_NumLayers_{p.num_hidden_layers}_posEnc_{p.pos_encoding}_hiddenCoords_{p.input_coord_to_all_layers}")
    # optimize_latent(**Params().__dict__)
