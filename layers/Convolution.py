"""Convolution layers"""

import tensorflow as tf


class Conv1D(tf.keras.Model):
    """1-D convolution layer with Xavier uniform initialization
    """
    def __init__(self, filters, kernel_size=1, stride=1, dilation=1, bias=True):
        """Constructor
        """
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same",
                                           data_format="channels_last", dilation_rate=dilation, activation=None,
                                           use_bias=bias, kernel_initializer="glorot_uniform")

    def call(self, inputs):
        """Forward pass
        """
        return self.conv(inputs)


class BatchNormConv1D(tf.keras.Model):
    """1-D convolution layer + batchnorm layer + activation + dropout
    """
    def __init__(self, filters, kernel_size, activation=None, dropout=0.5):
        """Constructor
        """
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout

        self.conv = Conv1D(filters=filters, kernel_size=kernel_size)
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=False):
        """Forward pass
        """
        inputs = self.batchnorm(self.conv(inputs))

        if self.activation is not None:
            inputs = self.activation(inputs)

        inputs = self.dropout(inputs, training=training)

        return inputs
