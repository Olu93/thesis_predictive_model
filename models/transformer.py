import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Dropout, Embedding, Multiply, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.optimizer_v2.adam import Adam


class TransformerModelUnidirectional(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1):
        super(TransformerModelUnidirectional, self).__init__()
        self.max_len = max_len
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

    def summary(self):
        x = Input(shape=(self.max_len, ))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class TransformerModelBidirectional(TransformerModelUnidirectional):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1) -> None:
        super(TransformerModelBidirectional, self).__init__(vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1)
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_len, embed_dim)
        self.embedding_reverse = TokenAndPositionEmbedding(max_len, vocab_len, embed_dim)
        self.reverse = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reverse(x, axes=-1), output_shape=(max_len, ))
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        x = inputs
        x_reverse = self.reverse(inputs)
        x = self.embedding(x)
        x_reverse = self.embedding_reverse(x_reverse)
        x = self.transformer_block(x)
        x_reverse = self.transformer_block(x_reverse)
        x = self.concat([x, x_reverse])
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        y_pred = self.activation_layer(x)

        return y_pred


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


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


if __name__ == "__main__":
    vocab_len = 11
    max_len = 21
    epochs = 1
    batch_size = 10
    adam_init = 0.001
    start_id = 0
    end_id = 21
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [[1, 2, 4], [3, 4, 6]],
        maxlen=max_len,
        padding='post',
    )
    inputs = tf.constant(padded_inputs)
    print("Transformer Bi:")
    transformer_model = TransformerModelBidirectional(vocab_len, max_len)
    transformer_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    transformer_model.summary()

    transformer_model(inputs)
