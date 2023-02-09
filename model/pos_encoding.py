from typing import Any

import torch
from torch import nn
import math
import numpy as np


class PosEncodingNone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_features = kwargs.get("coord_dimensions")
        self.out_dim = self.in_features

    def forward(self, coords):
        return coords

    def __repr__(self):
        d = "xyzt"
        return f"None ({d[:self.in_features]})"


class PosEncodingNeRF(PosEncodingNone):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("num_frequencies")
        assert isinstance(self.num_frequencies, tuple)
        assert len(self.num_frequencies) == self.in_features

        self.out_dim = self.in_features + 2 * np.sum(self.num_frequencies)

    def __repr__(self):
        d = "xyzt"
        return f"NeRF ({d[:self.in_features]} [{self.num_frequencies}])"

    def forward(self, coords):
        coords = coords.view(coords.shape[0], self.in_features)

        coords_pos_enc = coords
        for j, dim_freqs in enumerate(self.num_frequencies):
            for i in range(dim_freqs):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


class PosEncodingNeRFOptimized(PosEncodingNeRF):
    ''' Vectorized version of the class above. LOOK MA, NO LOOPS! '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.freq_scale = kwargs.get("freq_scale")
        self.exp_i_pi = torch.cat([2**torch.arange(i, dtype=torch.float32, device=device)[None] * self.freq_scale * np.pi for i in self.num_frequencies], dim=1)

    def __repr__(self):
        d = "xyzt"
        return f"NeRF Optimized ({d[:self.in_features]} [{self.num_frequencies}])"

    def forward(self, coords):
        coords_ = torch.cat([torch.tile(coords[..., j:j+1], (1, n)) for j, n in enumerate(self.num_frequencies)], dim=-1)
        exp_i_pi = torch.tile(self.exp_i_pi, (coords_.shape[0], 1))
        prod = exp_i_pi * coords_
        out = torch.cat((coords, torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class PosEncodingGaussian(PosEncodingNone):
    ''' https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frequencies = kwargs.get("num_frequencies")
        self.freq_scale = kwargs.get("freq_scale")
        assert isinstance(self.num_frequencies, tuple)
        assert len(self.num_frequencies) == 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.B_gauss = torch.normal(0.0, 1.0, size=(3, self.num_frequencies[0]), requires_grad=False).to(device) * self.freq_scale
        self.B_gauss_pi = 2. * np.pi * self.B_gauss

        # self.out_dim = 2 * total_freqs
        self.out_dim = 3 + 2 * self.num_frequencies[0]

    def __repr__(self):
        d = "xyzt"
        return f"Gaussian ({d[:self.in_features]} [{self.num_frequencies}])"

    def get_extra_state(self) -> Any:
        return {"B_gauss_pi": self.B_gauss_pi}  # Required to store gaussian array into network state dict

    def set_extra_state(self, state: Any):
        self.B_gauss_pi = state["B_gauss_pi"]  # Required to store gaussian array into network state dict

    def forward(self, coords):
        prod = coords @ self.B_gauss_pi
        out = torch.cat((coords, torch.sin(prod), torch.cos(prod)), dim=-1)
        return out
