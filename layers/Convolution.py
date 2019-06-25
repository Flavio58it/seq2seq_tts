"""Convolution layers"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class Conv1D(nn.Module):
    """1-D convolution layer with Xavier uniform initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True,
                 w_init_gain="linear"):
        """Constructor
        """
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = dilation * (kernel_size - 1) // 2

        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, inputs, training=False):
        """Forward pass
        """
        return self.conv(inputs)


class BatchNormConv1D(nn.Module):
    """1-D convolution layer + batchnorm layer + activation + dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, dropout=0.5, w_init_gain="linear"):
        """Constructor
        """
        super().__init__()

        self.activation = activation
        self.dropout = dropout

        self.conv1d = Conv1D(in_channels, out_channels, kernel_size=kernel_size, w_init_gain=w_init_gain)

        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, inputs, training=False):
        """Forward pass
        """
        inputs = self.batchnorm(self.conv1d(inputs))

        if self.activation is not None:
            inputs = self.activation(inputs)

        inputs = F.dropout(inputs, p=self.dropout, training=training)

        return inputs
