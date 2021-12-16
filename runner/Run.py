from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from helper.evaluation import results_by_instance, results_by_len
from models.lstm import ProcessLSTMSimpleModel
from models.transformer import TransformerModel

from readers.BPIC12 import BPIC12W
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

if __name__ == "__main__":
    data = BPIC12W(debug=False, subset=BPIC12W.subsets.W)
    # data = data.init_log(save=True)
    data = data.init_data()
    train_dataset = data.get_train_dataset()
    val_dataset = data.get_val_dataset()
    test_dataset = data.get_test_dataset()

    epochs = 1
    batch_size = 10
    adam_init = 0.001

    print("LSTM:")
    lstm_model = TransformerModel(data.vocab_len, data.max_len)
    lstm_model.build((None, data.max_len))
    lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    lstm_model.summary()
    lstm_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance(data, test_dataset, lstm_model, 'results_LSTM_by_instance.csv')
    results_by_len(data, test_dataset, lstm_model, 'results_LSTM_by_len.csv')

    print("Transformer:")
    transformer_model = TransformerModel(data.vocab_len, data.max_len)
    transformer_model.build((None, data.max_len))
    transformer_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    transformer_model.summary()
    transformer_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance(data, test_dataset, transformer_model, 'results_Transformer_by_instance.csv')
    results_by_len(data, test_dataset, transformer_model, 'results_Transformer_by_len.csv')
