"""seq2seq Decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.Attention import LocationSensitiveAttention
from layers.Linear import Dense, Prenet
from utils.Common import get_mask


class Tacotron2Decoder(nn.Module):
    """Tacotron2 Decoder
    """
    def __init__(self, target_dim, memory_dim, prenet_layers, attn_dim, location_filters, location_kernel_size,
                 num_LSTMCells, dec_LSTMCell_size, dropout, max_decoder_steps):
        """Constructor
        """
        super().__init__()

        self.target_dim = target_dim
        self.memory_dim = memory_dim
        self.attn_dim = attn_dim
        self.num_LSTMCells = num_LSTMCells
        self.dec_LSTMCell_size = dec_LSTMCell_size
        self.dropout = dropout
        self.max_decoder_steps = max_decoder_steps

        self.prenet = Prenet(target_dim, prenet_layers, dropout=dropout)

        rnn_sizes = [prenet_layers[-1] + memory_dim] + num_LSTMCells * [dec_LSTMCell_size]
        self.decoder_LSTMs = nn.ModuleList([nn.LSTMCell(in_dim, hidden_dim, bias=True) for in_dim, hidden_dim in
                                            zip(rnn_sizes, rnn_sizes[1:])])

        self.attention = LocationSensitiveAttention(dec_LSTMCell_size, memory_dim, attn_dim, location_filters,
                                                    location_kernel_size)
        self.acoustic_projection = Dense(dec_LSTMCell_size + memory_dim, target_dim, bias=True)
        self.gate_projection = Dense(dec_LSTMCell_size + memory_dim, 1, bias=True, init_gain="sigmoid")

    def _get_go_frame(self, memory):
        """Get all zeros frame to be used to start the decoding (as first input to the decoder)
        """
        batch_size = memory.size(0)
        go_frame = Variable(memory.data.new(batch_size, self.target_dim).zero_())

        return go_frame

    def _init_states(self, memory):
        """Initialize all states (decoder_rnn / attention states)
        """
        batch_size = memory.size(0)
        max_time = memory.size(1)

        self.LSTM_hidden = [Variable(memory.data.new(batch_size, self.dec_LSTMCell_size).zero_())] *\
            self.num_LSTMCells

        self.LSTM_cell = [Variable(memory.data.new(batch_size, self.dec_LSTMCell_size).zero_())] *\
            self.num_LSTMCells

        self.alignment = Variable(memory.data.new(batch_size, max_time).zero_())
        self.cumulative_alignments = Variable(memory.data.new(batch_size, max_time).zero_())
        self.attention_context = Variable(memory.data.new(batch_size, self.memory_dim).zero_())

    def parse_inputs(self, decoder_inputs):
        """Parses decoder inputs used in teacher forcing training
        """
        # (batch_size, max_time, target_dim) -> (max_time, batch_size, target_dim)
        decoder_inputs = decoder_inputs.transpose(0, 1)

        return decoder_inputs

    def parse_outputs(self, outputs, stop_tokens, alignments):
        """Parses the outputs of the decoder for further processing
        """
        # alignments: (batch_size, max_time)
        alignments = torch.stack(alignments).transpose(0, 1).contiguous()

        # acoustic outputs: (batch_size, max_time, target_dim)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        # stop tokens
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).contiguous()

        return outputs, stop_tokens, alignments

    def step(self, decoder_input, memory, mask=None, training=False):
        """Decoder step using stored decoder states, attention and memory
            Single decoder step:
                (1) Prenet to compress last output information
                (2) Concat compressed inputs with previous context vector (input feeding)
                (3) Decoder RNN (actual decoding) to predict current state s_{i}
                (4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments
                (5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
                (6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)
        """
        decoder_input = self.prenet(decoder_input)
        cell_input = torch.cat((decoder_input, self.attention_context), dim=-1)

        for idx in range(self.num_LSTMCells):
            self.LSTM_hidden[idx], self.LSTM_cell[idx] = self.decoder_LSTMs[idx](cell_input, (self.LSTM_hidden[idx],
                                                                                              self.LSTM_cell[idx]))
            self.LSTM_hidden[idx] = F.dropout(self.LSTM_hidden[idx], self.dropout, training=training)
            self.LSTM_cell[idx] = F.dropout(self.LSTM_cell[idx], self.dropout, training=training)
            cell_input = self.LSTM_hidden[idx]

        alignments_cat = torch.cat((self.alignment.unsqueeze(-1), self.cumulative_alignments.unsqueeze(-1)), dim=-1)

        self.attention_context, self.alignment = self.attention(self.LSTM_hidden[-1], memory, alignments_cat,
                                                                mask=mask)

        self.cumulative_alignments += self.alignment

        cell_output = torch.cat((self.LSTM_hidden[-1], self.attention_context), dim=-1)

        acoustic_frame = self.acoustic_projection(cell_output)
        stop_token = self.gate_projection(cell_output)

        return acoustic_frame, stop_token, self.alignment

    def forward(self, inputs, memory, memory_lengths):
        """Forward pass for training
        """
        go_frame = self._get_go_frame(memory).unsqueeze(1)
        inputs = torch.cat((go_frame, inputs), dim=1)
        inputs = self.parse_inputs(inputs)

        self._init_states(memory)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < inputs.size(0) - 1:
            inputs = inputs[len(outputs)]
            acoustic_frame, stop_token, alignment = self.step(inputs, memory, mask=get_mask(memory_lengths),
                                                              training=True)
            outputs += [acoustic_frame.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [alignment]

        outputs, stop_tokens, alignments = self.parse_outputs(outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments

    def inference(self, memory):
        """Inference
        """
        inputs = self._get_go_frame(memory)

        self._init_states(memory)

        outputs, stop_tokens, alignments = [], [], []

        while True:
            acoustic_frame, stop_token, alignment = self.step(inputs, memory, mask=None, training=False)

            outputs += [acoustic_frame.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [alignment]

            if torch.sigmoid(stop_token.data) >= 0.5:
                break

            if len(outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            inputs = acoustic_frame

        outputs, stop_tokens, alignments = self.parse_outputs(outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments
