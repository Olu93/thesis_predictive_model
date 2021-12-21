import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Dropout, Embedding, Multiply
from tensorflow.keras.models import Model


class TransformerModel(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1):
        super(TransformerModel, self).__init__()
        # self.inputs = InputLayer(input_shape=(max_len,))
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_len, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate1)
        # self.avg_pooling_layer = layers.GlobalAveragePooling1D()
        self.dropout1 = Dropout(rate2)
        self.dense = Dense(20, activation='relu')
        self.dropout2 = Dropout(rate2)
        self.output_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        # x = self.avg_pooling_layer(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        y_pred = self.activation_layer(x)

        return y_pred


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=0)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=0)
        self.zero = tf.constant(0, dtype=tf.float32)
        self.multiply = Multiply()

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1, dtype=tf.float32)
        # zero_indices = tf.cast(tf.not_equal(x, self.zero), tf.float32)
        # positions = self.multiply([positions, zero_indices])
        positions = self.pos_emb(tf.cast(positions, tf.int32))
        x = self.token_emb(x)
        return (x + positions) 