"""Common utilities"""

import os

import tensorflow as tf


def sequence_mask(sequence_lengths, expand=True):
    """Returns a 2-D or a 3-D sequence mask
    """
    maxlen = tf.reduce_max(sequence_lengths)
    if expand:
        return tf.expand_dims(tf.sequence_mask(sequence_lengths, maxlen=maxlen, dtype=tf.float32), axis=-1)

    return tf.sequence_mask(sequence_lengths, maxlen=maxlen, dtype=tf.float32)


def _masked_fill(x, mask, mask_value):
    """Fill the values of tensor x corresponding to mask with the mask_value
    """
    mask_values_tensor = mask_value * tf.ones_like(score)

    return tf.where(mask, score, mask_values_tensor)


def make_filepaths(text_dir, feats_dir, scp_file):
    """Make full paths for text / feats files
    """
    with open(scp_file, "r") as fp:
        filenames = fp.readlines()
    filenames = [name.strip("\n") for name in filenames]

    textpaths = [os.path.join(text_dir, name + ".txt") for name in filenames]
    featspaths = [os.path.join(feats_dir, name + ".npy") for name in filenames]

    return list(zip(textpaths, featspaths))
