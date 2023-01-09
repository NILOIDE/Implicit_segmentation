import torch
from torch import nn
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, coord_size, embed_size, activation_func_class, **kwargs):
        super(MLP, self).__init__()
        hidden_size = kwargs.get("hidden_size")
        input_coord_to_all_layers = kwargs.get("input_coord_to_all_layers")
        num_hidden_layers = kwargs.get("num_hidden_layers")
        dropout = kwargs.get("dropout")

        hidden_input_size = hidden_size + (coord_size if input_coord_to_all_layers else 0)
        a = [nn.Linear(coord_size + embed_size, hidden_size), nn.Dropout(dropout)]
        for i in range(num_hidden_layers - 1):
            a.append(nn.Linear(hidden_input_size, hidden_size))
            a.append(nn.Dropout(dropout))
        self.hid_layers = nn.ModuleList(a)
        # self.hid_layers = nn.ModuleList([nn.Linear(coord_size + embed_size, hidden_size),
        #                                  *[nn.Linear(hidden_input_size, hidden_size) for _ in range(num_hidden_layers - 1)]])

        self.out_size = hidden_size
        self.activation_func = activation_func_class()
        if hasattr(self.activation_func, "weight_init"):
            self.hid_layers.apply(self.activation_func.weight_init)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(x, dim=1)
        for layer in self.hid_layers:
            x = self.activation_func(layer(x))
        return x


class ResMLP(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, prev_x = x
        x = torch.cat(x, dim=1)
        for layer in self.hid_layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x)
                continue
            x = self.activation_func(layer(x)) + prev_x
            prev_x = x
        return x


class MLPHiddenCoords(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, x = x
        for layer in self.hid_layers:
            x = self.activation_func(layer(torch.cat((coord, x), dim=1)))
        return x


class ResMLPHiddenCoords(MLP):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, x = x
        for layer in self.hid_layers:
            x = self.activation_func(layer(torch.cat((coord, x), dim=1))) + x
        return x


class SegmentationHead(nn.Module):
    def __init__(self, input_size, num_classes=4, **kwargs):
        super(SegmentationHead, self).__init__()
        self.seg_layer = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor):
        out = self.seg_layer(x)
        out = nn.functional.softmax(out, dim=-1)
        return out


class ReconstructionHead(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out
