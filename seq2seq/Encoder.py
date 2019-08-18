"""seq2seq Encoder"""

import tensorflow as tf

from layers.Convolution import BatchNormConv1D
from layers.LSTM import LSTM


class Tacotron2Encoder(tf.keras.Model):
    """Tacotron2 Encoder
    """
    def __init__(self, num_conv_layers, conv_filters, conv_kernel_size, BLSTM_size, dropout):
        """Constructor
        """
        super().__init__()

        # convolutional layers
        filter_sizes = num_conv_layers * [conv_filters]
        self.convs = [BatchNormConv1D(filters=filter, kernel_size=conv_kernel_size,
                                      activation=tf.keras.activations.relu, dropout=dropout) for filter in
                      filter_sizes]

        # BLSTM layer
        self.BLSTM = LSTM(units=BLSTM_size, bidirectional=True, return_sequences=True, bias=True)

    def call(self, inputs, training=False):
        """Forward pass
        """
        for conv in self.convs:
            inputs = conv(inputs, training=training)

        outputs = self.BLSTM(inputs)

        return outputs
