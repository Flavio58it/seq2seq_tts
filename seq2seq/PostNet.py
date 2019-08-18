"""Post Processing Network"""

import tensorflow as tf

from layers.Convolution import BatchNormConv1D


class Tacotron2PostNet(tf.keras.Model):
    """Tacotron2 Post Processing Network
    """
    def __init__(self, model_target_dim, num_conv_layers, conv_filters, conv_kernel_size, dropout):
        """Constuctor
        """
        super().__init__()

        self.model_target_dim = model_target_dim
        self.num_conv_layers = num_conv_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout

        sizes = (num_conv_layers - 1) * [conv_filters] + [model_target_dim]
        activations = (num_conv_layers - 1) * [tf.keras.activations.tanh] + [None]

        self.convs = [BatchNormConv1D(filters=layer_size, kernel_size=conv_kernel_size, activation=layer_activation,
                                      dropout=dropout) for layer_size, layer_activation in zip(sizes, activations)]

    def call(self, inputs, training=False):
        """Forward pass
        """
        for conv in self.convs:
            inputs = conv(inputs, training=training)

        return inputs
