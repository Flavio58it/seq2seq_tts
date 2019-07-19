"""LSTM layer"""

import torch.nn as nn


class LSTM(nn.Module):
    """LSTM layer
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, bidirectional=True):
        """Constructor
        """
        super().__init__()

        if bidirectional:
            self.lstm_layer = nn.LSTM(in_dim, hidden_dim//2, num_layers=num_layers, bias=True, batch_first=True,
                                      bidirectional=True)
        else:
            self.lstm_layer = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, bias=True, batch_first=True,
                                      bidirectional=False)

    def forward(self, inputs, input_lengths):
        """Forward pass
        """
        input_lengths = input_lengths.cpu().numpy()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)

        self.lstm_layer.flatten_parameters()
        outputs, _ = self.lstm_layer(inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, inputs):
        """Inference
        """
        self.lstm_layer.flatten_parameters()

        outputs, _ = self.lstm_layer(inputs)

        return outputs
