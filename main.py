from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path

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


def main(params, exp_name=''):
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
    main(params, exp_name=f"NumLayers_{params.num_hidden_layers}_hidSize_{params.hidden_size}_enc_{params.pos_encoding}_modelType_{params.model_type}")
    # main(p, exp_name=f"SkipCon_{p.skip_connections}_NumLayers_{p.num_hidden_layers}_posEnc_{p.pos_encoding}_hiddenCoords_{p.input_coord_to_all_layers}")
