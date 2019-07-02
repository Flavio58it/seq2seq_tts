"""Tacotron2 Model"""

import math.sqrt as sqrt

import torch.nn as nn

from seq2seq.Decoder import Tacotron2Decoder
from seq2seq.Encoder import Tacotron2Encoder
from seq2seq.PostNet import Tacotron2PostNet
from utils.Common import get_mask


class Tacotron2Model(nn.Module):
    """Tacotron2 Model
    """
    def __init__(self, vocab_size, config):
        """Constructor
        """
        super().__init__()

        if config["audio_processor"] == "mag":
            self.target_dim = config["n_fft"]//2 + 1
        else:
            raise NotImplementedError

        # embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, config["embedding_dim"])
        std = sqrt(2.0 / (vocab_size + config["embedding_dim"]))
        val = sqrt(3.0) * std
        self.embedding_layer.weight.data.uniform_(-val, val)

        # encoder
        self.encoder = Tacotron2Encoder(config["embedding_dim"], num_conv_layers=config["enc_num_layers"],
                                        conv_filters=config["enc_filters"], conv_kernel_size=config["enc_kernel_size"],
                                        BLSTM_size=config["enc_BLSTM_size"], dropout=config["dropout_rate"])

        # decoder
        self.decoder = Tacotron2Decoder(target_dim=self.target_dim, memory_dim=config["enc_BLSTM_size"],
                                        prenet_layers=config["dec_prenet"], attn_dim=config["attn_size"],
                                        location_filters=config["location_filters"],
                                        location_kernel_size=config["location_kernel_size"],
                                        num_LSTMCells=config["num_LSTMCells"],
                                        dec_LSTMCell_size=config["dec_LSTMCell_size"], dropout=config["dropout_rate"],
                                        max_decoder_steps=config["max_decoder_steps"])

        # post processing network
        self.postnet = Tacotron2PostNet(in_dim=self.target_dim, num_conv_layers=config["postnet_num_layers"],
                                        conv_filters=config["postnet_filters"],
                                        conv_kernel_size=config["postnet_kernel_size"], dropout=config["dropout_rate"])

    def parse_output(self, outputs, output_lengths=None):
        """Parse the outputs of the model
        """
        postnet_acoustic_outputs, acoustic_outputs, gate_outputs, alignments = outputs

        if output_lengths is not None:
            mask = get_mask(output_lengths)
            mask = mask.expand(self.target_dim, mask.size(0), mask.size(1))
            mask = mask.permute(1, 2, 0)

            postnet_acoustic_outputs.masked_fill_(mask, 0.0)
            acoustic_outputs.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return (postnet_acoustic_outputs, acoustic_outputs, gate_outputs, alignments)

    def forward(self, inputs):
        """Forward pass
        """
        texts, text_lengths, max_text_len, acoustic_feats, acoustic_feats_len = inputs

        text_lengths, acoustic_feats_len = text_lengths.data, acoustic_feats_len.data

        embedded_texts = self.embedding_layer(texts)

        memory = self.encoder(embedded_texts, text_lengths, max_text_len)

        acoustic_outputs, gate_outputs, alignments = self.decoder(acoustic_feats, memory, text_lengths)

        postnet_acoustic_outputs = self.postnet(acoustic_outputs)
        postnet_acoustic_outputs = acoustic_outputs + postnet_acoustic_outputs

        return self.parse_output(postnet_acoustic_outputs, acoustic_outputs, gate_outputs, alignments)

    def inference(self, inputs):
        """Inference
        """
        texts = inputs
        embedded_texts = self.embedding_layer(texts)

        memory = self.encoder.inference(embedded_texts)

        acoustic_outputs, gate_outputs, alignments = self.decoder.inference(memory)

        postnet_acoustic_outputs = self.postnet.inference(acoustic_outputs)
        postnet_acoustic_outputs = acoustic_outputs + postnet_acoustic_outputs

        return self.parse_output(postnet_acoustic_outputs, acoustic_outputs, gate_outputs, alignments)
