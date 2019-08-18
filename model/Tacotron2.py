"""Tacotron2 Model"""

import tensorflow as tf

from seq2seq.Decoder import Tacotron2Decoder
from seq2seq.Encoder import Tacotron2Encoder
from seq2seq.PostNet import Tacotron2PostNet
from utils.Common import _masked_fill, sequence_mask


class Tacotron2(tf.keras.Model):
    """Tacotron2 Model
    """
    def __init__(self, vocab_size, config):
        """Constructor
        """
        super().__init__()

        if config["data_processors"]["audio_processor"] == "mag":
            self.target_dim = config["audio"]["n_fft"]//2 + 1
        elif config["data_processors"]["audio_processor"] == "mel":
            self.target_dim = config["audio"]["n_mel"]
        else:
            raise NotImplementedError

        # embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=config["architecture"]["embedding_dim"])

        # encoder
        self.encoder = Tacotron2Encoder(num_conv_layers=config["architecture"]["encoder"]["num_layers"],
                                        conv_filters=config["architecture"]["encoder"]["filters"],
                                        conv_kernel_size=config["architecture"]["encoder"]["kernel_size"],
                                        BLSTM_size=config["architecture"]["encoder"]["BLSTM_size"],
                                        dropout=config["architecture"]["dropout_rate"])

        # decoder
        self.decoder = Tacotron2Decoder(prenet_layers=config["architecture"]["decoder"]["prenet"],
                                        model_target_dim=self.target_dim,
                                        attn_dim=config["architecture"]["attention"]["attn_size"],
                                        location_filters=config["architecture"]["attention"]["filters"],
                                        location_kernel_size=config["architecture"]["attention"]["kernel_size"],
                                        num_rnn_cells=config["architecture"]["decoder"]["num_rnn_cells"],
                                        rnn_cell_size=config["architecture"]["decoder"]["rnn_cell_size"],
                                        dropout=config["architecture"]["dropout_rate"],
                                        max_decoder_steps=config["architecture"]["decoder"]["max_steps"])

        # post processing network
        self.postnet = Tacotron2PostNet(model_target_dim=self.target_dim,
                                        num_conv_layers=config["architecture"]["postnet"]["num_layers"],
                                        conv_filters=config["architecture"]["postnet"]["filters"],
                                        conv_kernel_size=config["architecture"]["postnet"]["kernel_size"],
                                        dropout=config["architecture"]["dropout_rate"])

    def parse_inputs(self, inputs):
        """Parse the inputs to the model
        """
        texts_padded, text_lengths, acoustics_padded, stop_tokens_padded, acoustic_lengths = inputs

        return ((texts_padded, text_lengths, acoustics_padded, acoustic_lengths),
                (acoustics_padded, stop_tokens_padded))

    def parse_outputs(self, outputs, output_lengths=None):
        """Parse the outputs of the model
        """
        postnet_acoustic_outputs, acoustic_outputs, stop_tokens, alignments = outputs

        if output_lengths is not None:
            mask = sequence_mask(output_lengths, expand=True)

            postnet_acoustic_outputs = _masked_fill(postnet_acoustic_outputs, mask, 0.0)
            acoustic_outputs = _masked_fill(acoustic_outputs, mask, 0.0)
            stop_tokens = _masked_fill(stop_tokens, mask[:, 0, :], 1e3)

        return (postnet_acoustic_outputs, acoustic_outputs, stop_tokens, alignments)

    def call(self, inputs):
        """Training forward pass
        """
        texts_padded, text_lengths, acoustics_padded, stop_tokens_padded, acoustic_lengths = inputs

        # embedding layer
        embeddings = self.embedding_layer(texts_padded)

        # encoder
        memory = self.encoder(embeddings, training=True)

        # decoder
        acoustic_outputs, stop_tokens, alignments = self.decoder(acoustics_padded, memory,
                                                                 mask=sequence_mask(text_lengths, expand=False))

        # post processing network
        postnet_acoustic_outputs = self.postnet(acoustic_outputs, training=True)
        postnet_acoustic_outputs = postnet_acoustic_outputs + acoustic_outputs

        return self.parse_outputs((postnet_acoustic_outputs, acoustic_outputs, stop_tokens, alignments),
                                  output_lengths=acoustic_lengths)

    def inference(self, text):
        """Inference
        """
        # embedding layer
        embeddings = self.embedding_layer(text)

        # encoder
        memory = self.encoder(embeddings, training=False)

        # decoder
        acoustic_outputs, stop_tokens, alignments = self.decoder.inference(memory)

        # post processing network
        postnet_acoustic_outputs = self.postnet(acoustic_outputs, training=False)
        postnet_acoustic_outputs = postnet_acoustic_outputs + acoustic_outputs

        return self.parse_outputs((postnet_acoustic_outputs, acoustic_outputs, stop_tokens, alignments))