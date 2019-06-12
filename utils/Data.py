"""Data handling / Utilities"""

import random
import numpy as np
from text.EnglishText import EnglishText
from utils.Common import to_gpu, make_filepaths
import torch
import torch.utils.data

random.seed(1234)


class TTSDatasetLoader(torch.utils.data.Dataset):
    """TTS Dataset Loader:
        (1) Loads text, feats pairs
        (2) Converts text to a sequence of ids (to be fed to an embedding layer)
    """
    def __init__(self, config, text_dir, feats_dir, scp_file):
        """Constructor
        """
        if config["text_processor"] == "english":
            self.text_processor = EnglishText()

        self.paths = make_filepaths(text_dir, feats_dir, scp_file)
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.get_text_feats_pair(self.paths[index])

    def get_text_feats_pair(self, path):
        textpath, featspath = path[0], path[1]

        text = self.get_text(textpath)
        audio_feats = self.get_audio_feats(featspath)

        return (text, audio_feats)

    def get_audio_feats(self, featspath):
        audio_feats = torch.from_numpy(np.load(featspath))

        return audio_feats

    def get_text(self, textpath):
        text = torch.IntTensor(self.text_processor.text_to_sequence(textpath))

        return text


class TTSDatasetCollate():
    """Creates padded batches
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        """Creates padded batch from text and audio feats
            Args:
                batch: [text, audio_feats]
        """
        # right pad all text sequences with zeros to max text length
        text_lengths, idx_sorted = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_text_len = text_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_text_len)
        text_padded.zero_()
        for idx in range(len(idx_sorted)):
            text = batch[idx_sorted[idx]][0]
            text_padded[idx, :text.size(0)] = text

        # right pad all audio_feats with zeros
        audio_feats_dim = batch[0][1].size(1)
        max_audio_feats_len = max([x[1].size(0) for x in batch])

        audio_feats_padded = torch.FloatTensor(len(batch), max_audio_feats_len, audio_feats_dim)
        audio_feats_padded.zero_()

        gate_padded = torch.FloatTensor(len(batch), max_audio_feats_len)
        gate_padded.zero_()

        audio_feats_len = torch.LongTensor(len(batch))

        for idx in range(len(idx_sorted)):
            audio_feats = batch[idx_sorted[idx]][1]
            audio_feats_padded[idx, : audio_feats.size(0), :] = audio_feats
            gate_padded[idx, audio_feats.size(0) - 1:] = 1
            audio_feats_len[idx] = audio_feats.size(0)

        return text_padded, text_lengths, audio_feats_padded, gate_padded, audio_feats_len


def batch_to_gpu(batch):
    """Parse batch inputs; and place them on the GPU
    """
    text_padded, text_lengths, audio_feats_padded, gate_padded, audio_feats_len = batch

    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    audio_feats_padded = to_gpu(audio_feats_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    audio_feats_len = to_gpu(audio_feats_len).long()

    x = (text_padded, text_lengths, audio_feats_padded, audio_feats_len)
    y = (audio_feats_padded, gate_padded)

    return (x, y)