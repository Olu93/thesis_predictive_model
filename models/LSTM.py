from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# https://keras.io/guides/functional_api/
class SimpleLSTMModelUnidrectional(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SimpleLSTMModelUnidrectional, self).__init__()
        self.max_len = max_len
        # self.inputs = InputLayer(input_shape=(max_len,))
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        # self.lstm_layer = Bidirectional(LSTM(ff_dim, return_sequences=True))
        self.lstm_layer = LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        # x = self.inputs(inputs)
        x = self.embedding(inputs)
        x = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        
        return y_pred

    def summary(self):
        x = Input(shape=(self.max_len,))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class SimpleLSTMModelBidrectional(SimpleLSTMModelUnidrectional):
    # Makes no sense
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SimpleLSTMModelBidrectional, self).__init__(vocab_len, max_len, embed_dim=10, ff_dim=20)
        self.lstm_layer = Bidirectional(LSTM(ff_dim, return_sequences=True))