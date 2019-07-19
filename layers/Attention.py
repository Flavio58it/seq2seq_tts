"""Attention layer"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers.Convolution import Conv1D
from layers.Linear import Linear


class LocationLayer(nn.Module):
    """Location features from current and cumulative attention attention weights from previous decoder timesteps
    """
    def __init__(self, attn_dim, location_filters, location_kernel_size):
        """Constructor
        """
        super().__init__()

        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size

        self.location_conv = Conv1D(2, location_filters, kernel_size=location_kernel_size, stride=1, dilation=1)
        self.location_dense = Linear(location_filters, attn_dim, bias=True, w_init_gain="tanh")

    def forward(self, attention_weights):
        """Forward pass
        """
        attention_weights = attention_weights.transpose(1, 2)
        location_features = self.location_conv(attention_weights)
        location_features = location_features.transpose(1, 2)

        processed_attention_weights = self.location_dense(location_features)

        return processed_attention_weights


class LocationSensitiveAttention(nn.Module):
    """Location sensitive attention (extends additive attention to use cumulative attention weights from previous
    decoder timesteps as a feature)
    """
    def __init__(self, query_dim, memory_dim, attn_dim, location_filters, location_kernel_size):
        """Constructor
        """
        super().__init__()

        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size

        self.query_layer = Linear(query_dim, attn_dim, bias=True, w_init_gain="tanh")
        self.memory_layer = Linear(memory_dim, attn_dim, bias=True, w_init_gain="tanh")
        self.location_layer = LocationLayer(attn_dim, location_filters, location_kernel_size)
        self.v = Linear(attn_dim, 1, bias=True)

        self.score_mask_value = -float("inf")

    def compute_alignment(self, processed_query, processed_memory, processed_attention_weights):
        """Compute the alignment score
        """
        score = self.v(torch.tanh(processed_query + processed_memory + processed_attention_weights))

        score = score.squeeze(-1)

        return score

    def forward(self, query, memory, prev_attention_weights, mask=None):
        """Forward pass
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_memory = self.memory_layer(memory)
        processed_attention_weights = self.location_layer(prev_attention_weights)

        alignment = self.compute_alignment(processed_query, processed_memory, processed_attention_weights)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=-1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
