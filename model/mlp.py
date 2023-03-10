import torch
from torch import nn
from typing import Tuple
from model.activation_functions import Layer


class MLP(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, layer_class: Layer, **kwargs):
        super(MLP, self).__init__()
        hidden_size = kwargs.get("hidden_size")
        input_coord_to_all_layers = kwargs.get("input_coord_to_all_layers")
        num_hidden_layers = kwargs.get("num_hidden_layers")

        hidden_input_size = hidden_size + (coord_size if input_coord_to_all_layers else 0)
        a = [layer_class(coord_size + embed_size, hidden_size, **kwargs)]
        for i in range(num_hidden_layers - 1):
            a.append(layer_class(hidden_input_size, hidden_size, **kwargs))
        self.hid_layers = nn.ModuleList(a)
        self.out_size = hidden_size

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(x, dim=1)
        for layer in self.hid_layers:
            x = layer(x)
        return x


class ResMLP(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, prev_x = x
        x = torch.cat(x, dim=1)
        for layer in self.hid_layers:
            x = layer(x) + prev_x
            prev_x = x
        return x


class MLPHiddenCoords(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, x = x
        for layer in self.hid_layers:
            x = layer(torch.cat((coord, x), dim=1))
        return x


class ResMLPHiddenCoords(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, x = x
        for layer in self.hid_layers:
            x = layer(torch.cat((coord, x), dim=1)) + x
        return x


class SegmentationHead(nn.Module):
    def __init__(self, input_size, num_classes=4, **kwargs):
        super(SegmentationHead, self).__init__()
        self.seg_layer = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor):
        out = self.seg_layer(x)
        out = nn.functional.softmax(out, dim=1)
        return out


class ReconstructionHead(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out
