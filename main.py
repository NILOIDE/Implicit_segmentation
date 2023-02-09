from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
import argparse
import yaml

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior
from data_loading.data_loader import Seg4DWholeImage_SAX, Seg3DWholeImage_SAX


@dataclass
class Params:
    initial_val: bool = False
    pos_encoding: str = "none"  # nerf, none, gaussian
    num_frequencies: Tuple[int, ...] = (4, 4, 4)
    # num_frequencies: Tuple[int, ...] = (128,)
    freq_scale: float = 1.0
    latent_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.00
    num_hidden_layers: int = 8
    side_length: Tuple[int, int, Optional[int]] = (100, 100, -1, 1)
    heart_pad: int = 10
    batch_size: int = 1
    max_epochs: int = 3001
    train_log_interval: int = 1
    val_max_epochs: int = 2001  # Lightning is faster doing (2000 epochs x 1 batch) than (1 epoch x 2000 batches)
    val_interval: int = 10
    val_log_interval: int = 500
    val_x_holdout_rate: int = 1  # Height
    val_y_holdout_rate: int = 1  # Width
    val_z_holdout_rate: int = 1  # Slices
    val_t_holdout_rate: int = 3  # Time
    seg_class_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    lr: float = 1e-4
    fine_tune_lr: float = 1e-4
    latent_reg: float = 1e-4
    weight_reg: float = 1e-4
    activation: str = "wire"  # periodic, relu
    skip_connections: bool = True
    input_coord_to_all_layers: bool = False
    model_type: str = "shared"  # shared, separate, mounted
    augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    # augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = (("translation", {"x_lim": 0.25, "y_lim": 0.25}),
    #                                                          ("gamma", {"gamma_lim": (0.7, 1.4)}))


def main_train(config_path=None, exp_name=''):
    config = {}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    params = Params(**config)
    # dataset = Seg3DWholeImage_SAX(**params.__dict__)
    dataset = Seg4DWholeImage_SAX(**params.__dict__)
    coord_dimensions = dataset.sample_coords.shape[-1]
    assert coord_dimensions == len(params.side_length)
    if params.model_type == "separate":
        model = ImplicitNetSeparateSegPrior(num_train_samples=len(dataset), coord_dimensions=coord_dimensions, aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    elif params.model_type == "shared":
        model = ImplicitNetSegPrior(**dict(num_train_samples=len(dataset), coord_dimensions=coord_dimensions, aug_num_parameters=dataset.num_aug_params, **params.__dict__))
    elif params.model_type == "mounted":
        model = ImplicitNetMountedSegPrior(num_train_samples=len(dataset), coord_dimensions=coord_dimensions, aug_num_parameters=dataset.num_aug_params, **params.__dict__)
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    root_dir = Path(r"C:\Users\nilst\Documents\Implicit_segmentation\logs") / "4d"
    if exp_name:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(dataset.__class__.__name__)}'
    print(root_dir)
    os.makedirs(str(root_dir), exist_ok=True)
    ckpt_saver = ModelCheckpoint(save_top_k=1, dirpath=root_dir / "checkpoint", monitor="step", mode="max")
    trainer = pl.Trainer(max_epochs=params.max_epochs, log_every_n_steps=50, accelerator="gpu", default_root_dir=root_dir, callbacks=[ckpt_saver])
    # sd = torch.load(r"C:\Users\nilst\Documents\Implicit_segmentation\logs\4d\NumLayers_8_hidSize_128_enc_none_modelType_shared\20230202-233548_Seg4DWholeImage_SAX\checkpoint\epoch=57-step=112056.ckpt")
    # model.load_state_dict(sd["state_dict"])
    # model = model.load_from_checkpoint(r"C:\Users\nilst\Documents\Implicit_segmentation\logs\4d\NumLayers_8_hidSize_128_enc_none_modelType_shared\20230202-233548_Seg4DWholeImage_SAX\checkpoint\epoch=57-step=112056.ckpt",
    #                                    strict=True)
    start = datetime.now()
    trainer.fit(model, train_dataloaders=train_dataloader)
    print("Elapsed time:", datetime.now() - start)


def main_eval():
    raise NotImplementedError


def parse_command_line():
    main_parser = argparse.ArgumentParser(description="Implicit Segmentation",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    main_subparsers = main_parser.add_subparsers(dest='pipeline')

    # train
    parser_train = main_subparsers.add_parser("train")
    parser_train.add_argument("-c", "--config",
                              help="path to configuration file", required=False,
                              default=r"C:\Users\nilst\Documents\Implicit_segmentation\configs\4d_cardiac_config.yaml"
                              )
    parser_train.add_argument("-n", "--exp_name",
                              help="custom experiment name", required=False,
                              default=""
                              )
    parser_train.set_defaults(func=main_train)

    # eval
    parser_eval = main_subparsers.add_parser("eval")
    parser_eval.add_argument("-c", "--config",
                             help="path to configuration yml file", required=False,
                             default=r"C:\Users\nilst\Documents\Implicit_segmentation\configs\4d_cardiac_config.yaml"
                             )
    parser_eval.set_defaults(func=main_eval)

    return main_parser.parse_args()


if __name__ == '__main__':
    parse_command_line()
    args = parse_command_line()
    if args.pipeline is None:
        main_train()
    else:
        args.func(args)
