# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.utils import losses_utils

y_true = tf.constant([[3, 1, 2], [2, 1, 1], [1, 1, 0], [0, 2, 3]], dtype=tf.float32)
y_pred_sparse = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=tf.float32)
# y_pred_sparse = tf.constant([
#     [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
#     [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
#     [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
#     [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
# ], dtype=tf.float32)
y_pred = tf.constant([
    [[0.04, 0.95, 0.01, 0.00], [0.10, 0.80, 0.10, 0.00], [0.20, 0.70, 0.10, 0.00]],
    [[0.01, 0.02, 0.97, 0.00], [0.10, 0.80, 0.10, 0.00], [0.10, 0.80, 0.01, 0.00]],
    [[0.95, 0.01, 0.04, 0.00], [0.09, 0.90, 0.01, 0.00], [0.05, 0.90, 0.05, 0.00]],
    [[0.90, 0.05, 0.05, 0.00], [0.08, 0.90, 0.02, 0.00], [0.05, 0.90, 0.05, 0.00]],
], dtype=tf.float32)

cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.AUTO)
# cce(y_true, tf.reshape(y_pred, (-1, 1))).numpy()
print(cce(y_true, y_pred).numpy())
print(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))
print(tf.keras.losses.sparse_categorical_crossentropy(y_true, tf.reshape(y_pred_sparse, (-1, 1)), from_logits=True))
# %%
# tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

# %%


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
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
        result = self.loss(y_true_masked, y_pred_masked)
        return result


cce = CrossEntropyLossModified()
cce(y_true, y_pred).numpy()

# %%
