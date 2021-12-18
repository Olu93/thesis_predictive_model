from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from readers.BPIC12 import BPIC12W
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# https://keras.io/guides/functional_api/
class SimpleLSTMModel(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SimpleLSTMModel, self).__init__()
        # self.inputs = InputLayer(input_shape=(max_len,))
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=True)
        self.lstm_layer = Bidirectional(LSTM(ff_dim, return_sequences=True))
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        # x = self.inputs(inputs)
        x = self.embedding(inputs)
        x = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        
        return y_pred