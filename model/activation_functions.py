import torch
from torch import nn
import numpy as np


class Sine(nn.Module):
    """ See SIREN paper and github. """

    def forward(self, x):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * x)

    @staticmethod
    def weight_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


class ComplexWIRE(nn.Module):
    '''
        See WIRE paper and github.
        Implicit representation with complex Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

    def forward(self, x):
        omega = self.omega_0 * x
        scale = self.scale_0 * x

        return torch.exp(1j * omega - scale.abs().square())


class WIRE(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_features=128, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0

        self.freqs = nn.Linear(in_features, in_features, bias=bias)
        self.scale = nn.Linear(in_features, in_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale ** 2))
