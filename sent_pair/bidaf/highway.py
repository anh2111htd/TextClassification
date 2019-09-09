from torch import nn
import torch.nn.functional as f_torch


class HighwayConfig(object):
    def __init__(self, hw_n_layers):
        self.hw_n_layers = hw_n_layers


class Highway(nn.Module):
    def __init__(self, in_size, config):
        super(Highway, self).__init__()
        self.hw_n_layers = config.hw_n_layers
        self.activation_func = f_torch.relu

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(self.hw_n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(self.hw_n_layers)])

    def forward(self, x):
        for i in range(self.hw_n_layers):
            normal_layer_ret = self.activation_func(self.normal_layer[i](x))
            gate = f_torch.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x
