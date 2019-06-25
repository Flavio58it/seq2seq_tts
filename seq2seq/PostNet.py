"""Post Processing Network"""

import torch.nn as nn
import torch

from layers.Convolution import BatchNormConv1D


class Tacotron2PostNet(nn.Module):
    """Post Processing Network
    """
    def __init__(self, in_dim, num_conv_layers, conv_filters, conv_kernel_size, dropout):
        """Constructor
        """
        super().__init__()

        layer_sizes = [in_dim] + (num_conv_layers - 1) * [conv_filters] + [in_dim]
        layer_activations = (num_conv_layers - 1) * [torch.tanh] + [None]

        self.convs = nn.ModuleList([BatchNormConv1D(in_channels, out_channels, kernel_size=conv_kernel_size,
                                                    activation=activation, dropout=dropout) 
                                    for in_channels, out_channels, activation in zip(layer_sizes, layer_sizes[1:],
                                    layer_activations)])

    def forward(self, inputs):
        """Forward pass
        """
        inputs = inputs.transpose(1, 2)

        for conv in self.convs:
            inputs = conv(inputs, training=True)

        inputs = inputs.transpose(1, 2)

        return inputs

    def inference(self, inputs):
        """Inference
        """
        inputs = inputs.transpose(1, 2)

        for conv in self.convs:
            inputs = conv(inputs, training=True)

        inputs = inputs.transpose(1, 2)

        return inputs
