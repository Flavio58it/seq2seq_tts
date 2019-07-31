"""Linear layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Feedforward linear layer with Xavier Uniform initialization
    """
    def __init__(self, in_dim, out_dim, bias=True, init_gain="linear"):
        """Constructor
        """
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self._init_weights(init_gain)

    def _init_weights(self, init_gain):
        """Initialize the weights of the linear layer
        """
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, inputs):
        """Forward pass
        """
        return self.dense_layer(inputs)


class Prenet(nn.Module):
    """Prenet with feedforward linear layers + ReLU activations + dropout
    """
    def __init__(self, in_dim, prenet_layers, dropout=0.5):
        """Constructor
        """
        super().__init__()

        layer_sizes = [in_dim] + prenet_layers
        self.layers = nn.ModuleList([Linear(in_size, out_size, bias=True) for in_size, out_size in
                                     zip(layer_sizes, layer_sizes[1:])])
        self.dropout = dropout

    def forward(self, inputs):
        """Forward pass
        """
        for layer in self.layers:
            # dropout in prenet in applied in inference also; to increase diversity
            inputs = F.dropout(F.relu(layer(inputs)), p=self.dropout, training=True)

        return inputs
