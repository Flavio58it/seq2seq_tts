"""Common utilities"""

import os

import torch


def sequence_mask(sequence_lengths):
    """Generate sequence mask given sequence lengths
    """
    maxlen = torch.max(sequence_lengths).item()

    if torch.cuda.is_available():
        idx = torch.arange(0, maxlen, out=torch.cuda.LongTensor(maxlen))
    else:
        idx = torch.arange(0, maxlen, out=torch.LongTensor(maxlen))

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
