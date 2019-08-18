"""LSTM layers"""

import tensorflow as tf


class LSTM(tf.keras.Model):
    """LSTM layer
    """
    def __init__(self, units, bidirectional=False, return_sequences=True, bias=True):
        """Constructor
        """
        super().__init__()

        self.units = units
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.bias = bias

        if bidirectional:
            self.fw_LSTM = tf.keras.layers.LSTM(units=units//2, use_bias=bias, kernel_initializer="glorot_uniform",
                                                return_sequences=return_sequences)
            self.bw_LSTM = tf.keras.layers.LSTM(units=units//2, use_bias=bias, kernel_initializer="glorot_uniform",
                                                return_sequences=return_sequences, go_backwards=True)
        else:
            self.LSTM = tf.keras.layers.LSTM(units=units, use_bias=bias, kernel_initializer="glorot_uniform",
                                             return_sequences=return_sequences)

    def call(self, inputs):
        """Forward pass
        """
        if self.bidirectional:
            fw_output = self.fw_LSTM(inputs)
            bw_output = self.bw_LSTM(inputs)

            outputs = tf.concat([fw_output, bw_output], axis=-1)
        else:
            outputs = self.LSTM(inputs)

        return outputs
