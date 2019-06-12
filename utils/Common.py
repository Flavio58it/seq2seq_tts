"""Common utilities"""

import os

import torch


def get_mask(sequence_lengths):
    """Generate mask given sequence lengths
    """
    max_len = torch.max(sequence_lengths).item()
    idx = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (idx < sequence_lengths.unsqueeze(1)).byte()

    return ~mask


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
