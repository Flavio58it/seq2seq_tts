"""Tacotron2 Model"""

import math

import torch.nn as nn

from seq2seq.Decoder import Tacotron2Decoder
from seq2seq.Encoder import Tacotron2Encoder
from seq2seq.PostNet import Tacotron2PostNet
from utils.Common import sequence_mask


class Tacotron2Model(nn.Module):
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
        self.embedding_layer = nn.Embedding(vocab_size, config["architecture"]["embedding_dim"])
        std = math.sqrt(2.0 / (vocab_size + config["architecture"]["embedding_dim"]))
        val = math.sqrt(3.0) * std
        self.embedding_layer.weight.data.uniform_(-val, val)

        # encoder
        self.encoder = Tacotron2Encoder(config["architecture"]["embedding_dim"],
                                        num_conv_layers=config["architecture"]["encoder"]["num_layers"],
                                        conv_filters=config["architecture"]["encoder"]["filters"],
                                        conv_kernel_size=config["architecture"]["encoder"]["kernel_size"],
                                        BLSTM_size=config["architecture"]["encoder"]["BLSTM_size"],
                                        dropout=config["architecture"]["dropout_rate"])

        # decoder
        self.decoder = Tacotron2Decoder(target_dim=self.target_dim,
                                        memory_dim=config["architecture"]["encoder"]["BLSTM_size"],
                                        prenet_layers=config["architecture"]["decoder"]["prenet"],
                                        attn_dim=config["architecture"]["attention"]["attn_size"],
                                        location_filters=config["architecture"]["attention"]["filters"],
                                        location_kernel_size=config["architecture"]["attention"]["kernel_size"],
                                        num_LSTMCells=config["architecture"]["decoder"]["num_LSTMCells"],
                                        dec_LSTMCell_size=config["architecture"]["decoder"]["LSTMCell_size"],
                                        dropout=config["architecture"]["dropout_rate"],
                                        max_decoder_steps=config["architecture"]["decoder"]["max_steps"])

        # post processing network
        self.postnet = Tacotron2PostNet(in_dim=self.target_dim,
                                        num_conv_layers=config["architecture"]["postnet"]["num_layers"],
                                        conv_filters=config["architecture"]["postnet"]["filters"],
                                        conv_kernel_size=config["architecture"]["postnet"]["kernel_size"],
                                        dropout=config["architecture"]["dropout_rate"])

    def forward(self, texts, text_lengths, acoustic_feats):
        """Forward pass
        """
        # compute sequence mask
        mask = sequence_mask(text_lengths).to(texts.device)

        # encoder
        embeddings = self.embedding_layer(texts)
        memory = self.encoder(embeddings, text_lengths)

        # decoder
        acoustic_outputs, stop_tokens, alignments = self.decoder(acoustic_feats, memory, mask)

        # post processing network
        acoustic_outputs_postnet = self.postnet(acoustic_outputs)
        acoustic_outputs_postnet = acoustic_outputs_postnet + acoustic_outputs

        return acoustic_outputs_postnet, acoustic_outputs, stop_tokens, alignments

    def inference(self, text):
        """Inference
        """
        # encoder
        embeddings = self.embedding_layer(text)
        memory = self.encoder.inference(embeddings)

        # decoder
        acoustic_outputs, stop_tokens, alignments = self.decoder.inference(memory)

        # post processing network
        acoustic_outputs_postnet = self.postnet.inference(acoustic_outputs)
        acoustic_outputs_postnet = acoustic_outputs_postnet + acoustic_outputs

        return acoustic_outputs_postnet, acoustic_outputs, stop_tokens, alignments
