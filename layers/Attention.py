"""Attention layers"""

import tensorflow as tf

from layers.Convolution import Conv1D
from layers.Linear import Linear
from utils.Common import _masked_fill


class LocationLayer(tf.keras.Model):
    """Location features from current and cumulative alignments
    """
    def __init__(self, attn_dim, location_filters, location_kernel_size):
        """Constructor
        """
        super().__init__()

        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size

        self.location_conv = Conv1D(filters=location_filters, kernel_size=location_kernel_size, bias=True)
        self.location_dense = Linear(hidden_dim=attn_dim, bias=True)

    def call(self, alignments):
        """Forward pass
        """
        location_features = self.location_conv(alignments)
        processed_alignments = self.location_dense(location_features)

        return processed_alignments


class LocationSensitiveAttention(tf.keras.Model):
    """Location sensitive attention (extends additive attention to use cumulative attention weights from previous
    decoder timesteps as a feature)
    """
    def __init__(self, attn_dim, location_filters, location_kernel_size):
        """Constructor
        """
        super().__init__()

        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size

        self.query_layer = Linear(hidden_dim=attn_dim, bias=True)
        self.memory_layer = Linear(hidden_dim=attn_dim, bias=True)
        self.location_layer = LocationLayer(attn_dim=attn_dim, location_filters=location_filters,
                                            location_kernel_size=location_kernel_size)
        self.v = Linear(hidden_dim=1, bias=True)

        self.score_mask_value = -float("inf")

    def compute_attention_score(self, query, processed_memory, alignments):
        """Compute the attention score
        """
        expanded_query = tf.expand_dims(query, axis=1)
        processed_query = self.query_layer(expanded_query)

        processed_alignments = self.location_layer(alignments)

        attention_score = self.v(tf.keras.activations.tanh(processed_query + processed_memory + processed_alignments))
        attention_score = attention_score.squeeze(-1)

        return attention_score

    def call(self, query, memory, alignments, mask):
        """Forward pass
        """
        # compute the attention score
        processed_memory = self.memory_layer(memory)
        attention_score = self.compute_attention_score(query, processed_memory, alignments)

        # apply masking
        if mask is not None:
            attention_score = _masked_fill(attention_score, mask, self.score_mask_value)

        # normalize the attention values
        alignment = tf.keras.activations.softmax(attention_score, axis=-1)

        # compute attention context
        attention_context = tf.expand_dims(alignment, axis=1) * memory
        attention_context = tf.reduce_sum(attention_context, axis=1)

        return attention_context, alignment
