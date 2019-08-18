"""Linear layers"""

import tensorflow as tf


class Linear(tf.keras.Model):
    """Feedforward linear layer with Xavier uniform initialization
    """
    def __init__(self, hidden_dim, bias=True):
        """Constructor
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bias = bias

        self.linear_layer = tf.keras.layers.Dense(units=hidden_dim, use_bias=bias, kernel_initializer="glorot_uniform")

    def call(self, inputs):
        """Forward pass
        """
        return self.linear_layer(inputs)


class Prenet(tf.keras.Model):
    """Prenet with feedforward linear layers + ReLU activation + dropout
    """
    def __init__(self, prenet_layers, dropout=0.5):
        """Constructor
        """
        self.layers = [Linear(hidden_dim=layer_dim, bias=True) for layer_dim in prenet_layers]
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs):
        """Forward pass
        """
        for layer in self.layers:
            # dropout in prenet is applied in inference also; to increase diversity during synthesis
            inputs = self.dropout(tf.keras.activations.relu(layer(inputs)), training=True)

        return inputs
