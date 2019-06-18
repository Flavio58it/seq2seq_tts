"""seq2seq Encoder"""

import torch.functional as F
import torch.nn as nn

from layers.Convolution import BatchNormConv1D


class Tacotron2Encoder(nn.Module):
    """Tacotron2 Encoder
    """
    def __init__(self, in_dim, num_conv_layers, conv_filters, conv_kernel_size, BLSTM_size, dropout):
        """Constructor
        """
        super().__init__()

        filters = [in_dim] + num_conv_layers * [conv_filters]
        self.convs = nn.ModuleList([BatchNormConv1D(in_channels, out_channels, kernel_size=conv_kernel_size,
                                    activation=F.relu, dropout=dropout) for in_channels, out_channels in
                                    zip(filters, filters[1:])])
        self.BLSTM = nn.LSTM(conv_filters, BLSTM_size//2, num_layers=1, bias=True, batch_first=True, dropout=dropout,
                             bidirectional=True)

    def forward(self, inputs, input_lengths):
        """Forward pass
        """
        inputs = inputs.transpose(1, 2)

        for conv in self.convs:
            inputs = conv(inputs, training=True)

        inputs = inputs.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)

        self.blstm.flatten_parameters()
        outputs, _ = self.blstm(inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, inputs):
        """Inference
        """
        inputs = inputs.transpose(1, 2)

        for conv in self.convs:
            inputs = conv(inputs, training=False)

        inputs = inputs.transpose(1, 2)

        self.blstm.flatten_parameters()
        outputs, _ = self.blstm(inputs)

        return outputs
