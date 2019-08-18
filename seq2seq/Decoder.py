"""seq2seq Decoder"""

import tensorflow as tf

from layers.Attention import LocationSensitiveAttention
from layers.Linear import Linear, Prenet


class Tacotron2Decoder(tf.kears.Model):
    """Tacotron2 Decoder
    """
    def __init__(self, prenet_layers, model_target_dim, attn_dim, location_filters, location_kernel_size,
                 num_rnn_cells, rnn_cell_size, dropout, max_decoder_steps):
        """Constructor
        """
        super().__init__()

        self.prenet_layers = prenet_layers
        self.model_target_dim = model_target_dim
        self.attn_dim = attn_dim
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size
        self.num_rnn_cells = num_rnn_cells
        self.rnn_cell_size = rnn_cell_size
        self.dropout = dropout
        self.max_decoder_steps = max_decoder_steps

        self.prenet = Prenet(prenet_layers=prenet_layers, dropout=dropout)

        self.attention = LocationSensitiveAttention(attn_dim=attn_dim, location_filters=location_filters,
                                                    location_kernel_size=location_kernel_size)

        self.decoder = [tf.keras.layers.LSTMCell(units=rnn_cell_size, use_bias=True,
                                                 kernel_initializer="glorot_uniform") for _ in range(num_rnn_cells)]

        self.acoustic_projection = Linear(hidden_dim=model_target_dim, bias=True)
        self.stop_token_projection = Linear(hidden_dim=1, bias=True)

    def _get_go_frame(self, memory):
        """Get all zeros frame to be used to start the decoding (as decoder input for zeroth timestep)
        """
        B = memory.get_shape()[0]
        go_frame = tf.zeros([B, self.model_target_dim])

        return go_frame

    def _init_states(self, memory):
        """Initialize all states (decoder rnn / attention states)
        """
        B = memory.get_shape()[0]
        T = memory.get_shape()[1]
        M = memory.get_shape()[2]

        self.h = [tf.zeros(B, self.rnn_cell_size)] * self.num_rnn_cells
        self.c = [tf.zeros(B, self.rnn_cell_size)] * self.num_rnn_cells

        self.alignment = tf.zeros([B, T])
        self.cumulative_alignments = tf.zeros([B, T])
        self.context = tf.zeros([B, M])

    def parse_inputs(self, decoder_inputs):
        """Parses decoder inputs used in teacher forcing training
        """
        # [B, T, target_dim] -> [T, B, target_dim]
        decoder_inputs = tf.transpose(decoder_inputs, perm=[1, 0, 2])

        return decoder_inputs

    def parse_outputs(self, outputs, stop_tokens, alignments):
        """Parses the outputs of the decoder for further processing
        """
        # alignments [batch_size, max_time]
        alignments = tf.stack(alignments, axis=1)

        # acoustic outputs [batch_size, max_time, target_dim]
        outputs = tf.stack(outputs, axis=1)

        # stop tokens
        stop_tokens = tf.stack(stop_tokens, axis=1)

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

        cell_input = tf.concat([decoder_input, self.context], axis=-1)
        for idx in range(self.num_rnn_cells):
            _, [self.h[idx], self.c[idx]] = self.decoder[idx](cell_input, [self.h[idx], self.c[idx]])
            self.h[idx] = self.dropout(self.h[idx], training=training)
            self.c[idx] = self.dropout(self.c[idx], training=training)
            cell_input = self.h[idx]

        alignments_cat = tf.concat([tf.expand_dims(self.alignment, axis=-1),
                                    tf.expand_dims(self.cumulative_alignments, axis=-1)], axis=-1)
        self.context, self.alignment = self.attention(self.h[-1], memory, alignments_cat, mask=mask)
        self.cumulative_alignments += self.alignment

        cell_output = tf.concat([self.h[-1], self.context], axis=-1)
        acoustic_frame = self.acoustic_projection(cell_output)
        stop_token = self.stop_token_projection(cell_output)

        return acoustic_frame, stop_token, self.alignment

    def call(self, inputs, memory, mask):
        """Training forward pass (Teacher forcing mode used in training, i.e previous timestep ground truth used as
        input to current timestep)
        """
        go_frame = tf.expand_dims(self._get_go_frame(memory), axis=1)
        inputs = tf.concat([go_frame, inputs], axis=1)
        inputs = self.parse_inputs(inputs)

        self._init_states(memory)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < inputs.get_shape()[0] - 1:
            decoder_input = inputs[len(outputs)]
            acoustic_frame, stop_token, alignment = self.step(decoder_input, memory, mask, training=True)

            outputs += [tf.squeeze(acoustic_frame, axis=1)]
            stop_tokens += [tf.squeeze(stop_token, axis=1)]
            alignments += [alignment]

        outputs, stop_tokens, alignments = self.parse_outputs(outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments

    def inference(self, memory):
        """Inference (previous timestep prediction used as input to the current timestep)
        """
        decoder_input = self._get_go_frame(memory)
        self._init_states(memory)

        outputs, stop_tokens, alignments = [], [], []

        while True:
            acoustic_frame, stop_token, alignment = self.step(decoder_input, memory, mask=None, training=False)

            outputs += [tf.squeeze(acoustic_frame, axis=1)]
            stop_tokens += [tf.squeeze(stop_token, axis=1)]
            alignments += [alignment]

            if tf.keras.activations.sigmoid(stop_token).numpy() >= 0.5:
                break
            if len(outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = acoustic_frame

        outputs, stop_tokens, alignments = self.parse_outputs(outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments
