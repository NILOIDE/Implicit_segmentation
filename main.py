from typing import Tuple, Optional
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetPrior, ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior
from data_loading.data_loader import SAX3D, SAX3D_Seg, SAX3D_Seg_test, SAX3D_test


@dataclass
class Params:
    pos_encoding: str = "nerf"  # nerf, none, gaussian
    num_frequencies: Tuple[int, ...] = (16,16,3)
    latent_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.00
    num_hidden_layers: int = 8
    side_length: Tuple[int, int, Optional[int]] = (64, 64, None)
    batch_size: int = 1  # TODO: Allow for varying Z dim sizes across batch to allow >1 batch size
    max_epochs: int = 201
    val_max_epochs: int = 2001
    val_interval: int = 20
    val_log_interval: int = 100
    lr: float = 1e-4
    fine_tune_lr: float = 1e-3
    weight_reg: float = 1e-5
    activation: str = "periodic"  # periodic, relu
    skip_connections: bool = True
    input_coord_to_all_layers: bool = False
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
    # model = model.load_from_checkpoint(r"C:\Users\nilst\Documents\Implicit_segmentation\logs\test\20230107-141934_SAX3D_Seg\checkpoint\epoch=200-step=16884.ckpt")
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
    # for skip in [True, False]:
    #     for coord in [True, False]:
    #         for model_type in ["separate", "shared"]:
    #             for lays in [4,6,8]:
    #                 print("Testing: Skip", skip, "Hid Coord:", coord, "Model Type:", model_type, "Layers:", lays)
    #                 start = datetime.now()
    #                 params = Params(skip_connections=skip, input_coord_to_all_layers=coord, model_type=model_type, num_hidden_layers=lays)
    #                 main(params, exp_name=f"ModelType_{model_type}_SkipCon_{skip}_HidCoord_{coord}_NumLayers_{lays}")
    #                 print("Elapsed time:", datetime.now() - datetime.now())
    # for f in [16, 8, 4]:
    #     for fz in [4,3,2]:
    #         for lays in [2,4,6]:
    #             print("Testing: Freq", (f,f,fz), "Layers:", lays)
    #             start = datetime.now()
    #             params = Params(num_frequencies=(f,f,fz), num_hidden_layers=lays)
    #             main(params, exp_name=f"SFs_{(f,f,fz)}_NumLayers_{lays}")
    #             print("Elapsed time:", datetime.now() - datetime.now())

    main(Params())
    # optimize_latent(**Params().__dict__)
