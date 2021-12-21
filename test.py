# %%
import tensorflow as tf
import numpy as np

x = tf.constant(np.arange(0,9).reshape((3,3)))
x
# %%
positions = tf.range(start=1, limit=4)
positions
# %%
zero_indices = tf.cast(x != 0, tf.int32)
# %%
all_pos = tf.repeat(tf.transpose(tf.reshape(positions, (-1, 1))), 3, axis=0)
all_pos
# %%
all_pos * zero_indices
# %%
