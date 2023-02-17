from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
import argparse
import yaml

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior, \
    ImplicitNetSeparateSegLatent, ImplicitNetSegLatent, ImplicitNetMountedSegLatent
from data_loading.data_loader import Seg4DWholeImage_SAX, Seg3DWholeImage_SAX, Seg4DWholeImage_SAX_test
from utils import ValProgressBar

LATEST_CHECKPOINT_DIR = "latest_checkpoint"
BEST_WEIGHTS_PATH = "best_weights.pt"
CONFIG_SAVE_PATH = "config.yaml"


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
    side_length: Tuple[int, ...] = (100, 100, -1, 1)
    coord_noise_std: float = 1e-3
    heart_pad: int = 10
    max_epochs: int = 3001
    log_interval: int = 1
    val_interval: int = 10
    fine_tune_max_epochs: int = 2001  # Lightning is faster doing (2000 epochs x 1 batch) than (1 epoch x 2000 batches)
    fine_tune_optimal_epochs: int = -1  # To be set during training
    fine_tune_log_interval: int = 500
    x_holdout_rate: int = 1  # Height
    y_holdout_rate: int = 1  # Width
    z_holdout_rate: int = 1  # Slices
    t_holdout_rate: int = 3  # Time
    seg_class_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    lr: float = 1e-4
    fine_tune_lr: float = 1e-4
    latent_reg: float = 1e-4
    weight_reg: float = 1e-4
    activation: str = "wire"  # periodic, relu, wire
    wire_omega_0: float = 10.
    wire_sigma_0: float = 10.
    skip_connections: bool = True
    input_coord_to_all_layers: bool = False
    model_type: str = "shared"  # shared, separate, mounted
    augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    # augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ({"translation", {"x_lim": 0.25, "y_lim": 0.25}},
    #                                                          {"gamma", {"gamma_lim": (0.7, 1.4)}})


@dataclass
class EvalParams:
    side_length: Tuple[int, int, Optional[int]] = (100, 100, -1, 1)
    heart_pad: int = 10
    fine_tune_max_epochs: int = 3001  # Lightning is faster doing (2000 epochs x 1 batch) than (1 epoch x 2000 batches)
    fine_tune_log_interval: int = 500
    x_holdout_rate: int = 1  # Height
    y_holdout_rate: int = 1  # Width
    z_holdout_rate: int = 1  # Slices
    t_holdout_rate: int = 1  # Time
    fine_tune_lr: float = 1e-4
    latent_reg: float = 1e-4
    weight_reg: float = 1e-4
    augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ()


def init_model(dataset, val_dataset, **params):
    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    else:
        raise NotImplementedError
    return model


def main_train(config_path: Optional[str] = None, exp_name: Optional[str] = None):
    # Config and hyper params
    config = {"params": {}}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    params = Params(**config["params"])
    root_dir = Path(config["log_dir"])

    # Dataset
    dataset = Seg4DWholeImage_SAX(load_dir=config["train_data_dir"],
                                  case_start_idx=config.get("train_start_idx", 0),
                                  num_cases=config["num_train"],
                                  **params.__dict__)
    coord_dimensions = dataset.sample_coords.shape[-1]
    assert coord_dimensions == len(params.side_length)
    train_dataloader = DataLoader(dataset, shuffle=True)

    val_dataset = Seg4DWholeImage_SAX_test(load_dir=config["val_data_dir"],
                                           case_start_idx=config.get("val_start_idx", config["num_train"]),
                                           num_cases=config["num_val"],
                                           **params.__dict__)
    # Model dir creation
    if exp_name is not None:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(dataset.__class__.__name__)}'
    os.makedirs(str(root_dir), exist_ok=True)
    print(root_dir)

    # Save config to model dir
    config["params"] = {k: v if not isinstance(v, tuple) else list(v) for k, v in params.__dict__.items()}
    with open(str(root_dir / CONFIG_SAVE_PATH), "w") as f:
        yaml.dump(config, f)

    # Model
    best_weights_path = root_dir / BEST_WEIGHTS_PATH
    model = init_model(dataset, val_dataset, best_checkpoint_path=best_weights_path, **params.__dict__)
    ckpt_latest_saver = ModelCheckpoint(save_top_k=1, dirpath=root_dir / LATEST_CHECKPOINT_DIR,
                                        monitor="step", mode="max")
    # Trainer
    trainer = pl.Trainer(max_epochs=params.max_epochs, accelerator="gpu",
                         default_root_dir=root_dir, callbacks=[ckpt_latest_saver])
    start = datetime.now()
    trainer.fit(model, train_dataloaders=train_dataloader)
    print("Train elapsed time:", datetime.now() - start)

    # Save updated config to model dir
    params.fine_tune_optimal_epochs = model.overall_best_num_fine_tune_epochs
    config["params"] = {k: v if not isinstance(v, tuple) else list(v) for k, v in params.__dict__.items()}
    with open(str(root_dir / CONFIG_SAVE_PATH), "w") as f:
        yaml.dump(config, f)
    return best_weights_path


def main_eval(weights_path: str, config_path: Optional[str] = None):

    source_dir = Path(weights_path).parent

    # Load original model's config
    source_config_path = source_dir / CONFIG_SAVE_PATH
    source_config = {"params": {}}
    if source_config_path.exists():
        with open(str(source_config_path), "r") as f:
            source_config = yaml.safe_load(f)

    # Load user defined config
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Merged user defined config with original model config
    merged_params = {**source_config["params"], **config["params"]}  # User defined config takes precedence
    params = Params(**merged_params)
    params = params.__dict__

    if "log_dir" in config:
        # If user defined a log_dir use that one
        log_dir = config["log_dir"]
    else:
        # Otherwise defined log_dir beside original model's dir
        log_dir = str(source_dir.parent)
    root_dir = Path(log_dir) / (str(source_dir.name) + f'_test_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(str(root_dir), exist_ok=True)
    print(root_dir)

    # Define dataset and model
    dataset = Seg4DWholeImage_SAX_test(load_dir=config["test_data_dir"],
                                       case_start_idx=config.get("test_start_idx", config["num_train"] + config["num_val"]),
                                       num_cases=config["num_test"],
                                       **params)
    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegLatent(dataset=dataset, split_name="test", **params)
    else:
        raise ValueError("Unknown model type.")

    # Load trained model's weights
    sd = torch.load(weights_path)
    del sd["h"]
    a = model.load_state_dict(sd, strict=False)
    assert len(a.missing_keys) == 1 and a.missing_keys[0] == 'h'
    assert len(a.unexpected_keys) == 0
    # Fine tune model
    if params["fine_tune_optimal_epochs"] > 0:
        max_epochs = params["fine_tune_optimal_epochs"]
    else:
        max_epochs = params["fine_tune_max_epochs"]
    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator="gpu",
                         default_root_dir=root_dir,
                         enable_model_summary=False,
                         enable_checkpointing=False, callbacks=[ValProgressBar()])
    trainer.fit(model, train_dataloaders=DataLoader(dataset, shuffle=False))


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
    # eval
    parser_eval = main_subparsers.add_parser("eval")
    parser_eval.add_argument("-c", "--config",
                             help="path to configuration yml file", required=False,
                             )
    parser_eval.add_argument("-w", "--weights",
                             help="path to the desired checkpoint .ckpt file meant for evaluation", required=True,
                             )
    return main_parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    if args.pipeline is None or args.pipeline == "train":
        config_path, exp_name = args.config, args.exp_name
        weights_path, fine_tune_epochs = main_train(config_path, exp_name)
        main_eval(weights_path, config_path)
    elif args.pipeline == "eval":
        config_path, weights_path = args.config, args.weights
        main_eval(weights_path, config_path)
    else:
        raise ValueError("Unknown pipeline selected.")
