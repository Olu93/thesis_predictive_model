import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


# https://stackoverflow.com/questions/61799546/how-to-custom-losses-by-subclass-tf-keras-losses-loss-class-in-tensorflow2-x
class CrossEntropyLoss(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, reduction=keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # y_true -> (batch_size, max_seq_len, vocab_len)
        # y_pred -> (batch_size, max_seq_len, vocab_len)
        result = self.loss(y_true, y_pred)
        return result


# class AccuracyMetric(keras.metrics.Metric):
#     pass


class SparseCrossEntropyLoss(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, reduction=keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # y_true -> (batch_size, max_seq_len, vocab_len)
        # y_pred -> (batch_size, max_seq_len, vocab_len)
        # tf.print(y_true.shape)
        # tf.print(y_pred.shape)
        result = self.loss(y_true, y_pred)
        return result


class SparseAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(SparseAccuracyMetric, self).__init__(**kwargs)
        self.acc_value = tf.constant(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true[0], tf.int32)
        # y_pred = tf.cast(y_pred, tf.int32)
        self.acc_value = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true[0], y_pred))

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)


class CrossEntropyLossModified(CrossEntropyLoss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.offset = 1
        self.pad = layers.ZeroPadding1D((0, self.offset))

    def call(self, y_true, y_pred):
        y_true = y_true[:, self.offset:]
        y_true = self.pad(y_true)
        result = self.loss(y_true, y_pred)
        return result


def cross_entropy_function(self, y_true, y_pred):
    # prone to numerical issues
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)