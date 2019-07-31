"""Common utilities"""

import os

import torch


def sequence_mask(sequence_lengths, maxlen=None):
    """Generate sequence mask given sequence lengths
    """
    if maxlen is None:
        maxlen = sequence_lengths.data.max()

    batch_size = sequence_lengths.size(0)
    seq_range = torch.arange(0, maxlen).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)

    if sequence_lengths.is_cuda:
        seq_range_expand = seq_range_expand.cuda()

    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(seq_range_expand))

    return seq_range_expand < seq_length_expand


def to_gpu(x):
    """Place a tensor on the GPU (if available)
    """
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return torch.autograd.Variable(x)


def make_filepaths(text_dir, feats_dir, scp_file):
    """Make full paths for text / feats files
    """
    with open(scp_file, "r") as fp:
        filenames = fp.readlines()
    filenames = [name.strip("\n") for name in filenames]

    textpaths = [os.path.join(text_dir, name + ".txt") for name in filenames]
    featspaths = [os.path.join(feats_dir, name + ".npy") for name in filenames]

    return list(zip(textpaths, featspaths))
