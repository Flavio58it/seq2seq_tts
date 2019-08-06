"""Attention layer"""

import torch
import torch.nn as nn

from layers.Convolution import Conv1D
from layers.Linear import Linear


class LocationLayer(nn.Module):
    """Location features from current and cumulative alignments
    """
    def __init__(self, attn_dim, location_filters, location_kernel_size):
        """Constructor
        """
        super().__init__()

        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size

        self.location_conv = Conv1D(2, location_filters, kernel_size=location_kernel_size, stride=1)
        self.location_dense = Linear(location_filters, attn_dim, bias=True, init_gain="tanh")

    def forward(self, alignments):
        """Forward pass
        """
        alignments = alignments.transpose(1, 2)
        location_features = self.location_conv(alignments)
        location_features = location_features.transpose(1, 2)

        processed_alignments = self.location_dense(location_features)

        return processed_alignments


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

        self.query_layer = Linear(query_dim, attn_dim, bias=True, init_gain="tanh")
        self.memory_layer = Linear(memory_dim, attn_dim, bias=True, init_gain="tanh")
        self.location_layer = LocationLayer(attn_dim, location_filters, location_kernel_size)
        self.v = Linear(attn_dim, 1, bias=True)

        self.score_mask_value = -float("inf")

    def compute_attention_score(self, query, processed_memory, alignments):
        """Compute the attention score
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_alignments = self.location_layer(alignments)

        attention_score = self.v(torch.tanh(processed_query + processed_memory + processed_alignments))
        attention_score = attention_score.squeeze(-1)

        return attention_score

    def forward(self, query, memory, alignments, mask):
        """Forward pass
        """
        # compute the attention score
        processed_memory = self.memory_layer(memory)
        attention_score = self.compute_attention_score(query, processed_memory, alignments)

        # apply masking
        if mask is not None:
            attention_score.data.masked_fill_(mask, self.score_mask_value)

        # normalize the attention values
        alignment = torch.softmax(attention_score, dim=-1)

        # compute attention context
        attention_context = torch.bmm(alignment.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, alignment
