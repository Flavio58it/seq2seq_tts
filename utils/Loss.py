"""Loss utilities"""

import tensorflow as tf

from utils.Common import sequence_mask


def MaskedMSE(targets, predictions, target_lengths):
    """Computes a masked mean squared error
    """
    mask = sequence_mask(target_lengths, expand=True)
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
    mask_ = mask * ones

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
        return tf.losses.mean_squared_error(y_true=targets, y_pred=predictions, sample_weight=mask_)


def MaskedSigmoidCrossEntropyLoss(targets, predictions, target_lengths):
    """Computes a masked Sigmoid Cross Entropy loss with logits
    """
    mask = sequence_mask(target_lengths, expand=False)

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(loss))]):
        masked_loss = loss * mask

    return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)
