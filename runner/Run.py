from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from helper.evaluation import FULL, results_by_instance, results_by_instance_seq2seq, results_by_len, show_predicted_seq
from models.lstm import SimpleLSTMModelTwoWay, SimpleLSTMModelOneWay
from models.transformer import TransformerModelTwoWay, TransformerModelOneWay
from readers.AbstractProcessLogReader import AbstractProcessLogReader, TaskModes

from readers.BPIC12LogReader import BPIC12LogReader
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

if __name__ == "__main__":
    data = BPIC12LogReader(debug=False, subset=BPIC12LogReader.subsets.W, mode=TaskModes.ENCODER_DECODER)
    # data = data.init_log(save=True)
    data = data.init_data()
    train_dataset = data.get_train_dataset()
    val_dataset = data.get_val_dataset()
    test_dataset = data.get_test_dataset()

    epochs = 1
    batch_size = 10
    adam_init = 0.001
    start_id = data.start_id
    end_id = data.end_id

    print("LSTM Uni:")
    lstm_model = SimpleLSTMModelOneWay(data.vocab_len, data.max_len)
    lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    lstm_model.summary()
    lstm_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance_seq2seq(data.idx2vocab, start_id, end_id, test_dataset, lstm_model, 'results/LSTM_by_instance.csv')

    print("LSTM Bi:")
    lstm_model = SimpleLSTMModelTwoWay(data.vocab_len, data.max_len)
    lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    lstm_model.summary()
    lstm_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance_seq2seq(data.idx2vocab, start_id, end_id, test_dataset, lstm_model, 'results/LSTM_by_instance_bi.csv')

    print("Transformer Uni:")
    transformer_model = TransformerModelOneWay(data.vocab_len, data.max_len)
    transformer_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    transformer_model.summary()
    transformer_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance_seq2seq(data.idx2vocab, start_id, end_id, test_dataset, transformer_model, 'results/Transformer_by_instance.csv')

    print("Transformer Bi:")
    transformer_model = TransformerModelTwoWay(data.vocab_len, data.max_len)
    transformer_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    transformer_model.summary()
    transformer_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance_seq2seq(data.idx2vocab, start_id, end_id, test_dataset, transformer_model, 'results/Transformer_by_instance_bi.csv')
