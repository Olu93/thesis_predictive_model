from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from helper.evaluation import FULL, results_by_instance, results_by_len, show_predicted_seq
from models.lstm import SimpleLSTMModel
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
    lstm_model = SimpleLSTMModel(data.vocab_len, data.max_len)
    lstm_model.build((None, data.max_len))
    lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    lstm_model.summary()
    start_id = data.start_id
    end_id = data.end_id
    lstm_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance(data.idx2vocab, start_id, end_id, test_dataset, lstm_model, 'results/LSTM_by_instance.csv')
    # results_by_len(data.idx2vocab, test_dataset, lstm_model, 'results/LSTM_by_len.csv')
    # show_predicted_seq(data.idx2vocab, test_dataset, lstm_model, save_path='results/LSTM_decoded_sequences_None.csv', mode=None)
    # show_predicted_seq(data.idx2vocab, test_dataset, lstm_model, save_path='results/LSTM_decoded_sequences_FULL.csv', mode=FULL)

    print("Transformer:")
    transformer_model = TransformerModel(data.vocab_len, data.max_len)
    transformer_model.build((None, data.max_len))
    transformer_model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_init), metrics=['accuracy'])
    transformer_model.summary()
    transformer_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
    results_by_instance(data.idx2vocab, start_id, end_id, test_dataset, transformer_model, 'results/Transformer_by_instance.csv')
    # results_by_len(data.idx2vocab, test_dataset, transformer_model, 'results/Transformer_by_len.csv')
    # show_predicted_seq(data.idx2vocab, test_dataset, transformer_model, save_path='results/Transformer_decoded_sequences_None.csv', mode=None)
    # show_predicted_seq(data.idx2vocab, test_dataset, transformer_model, save_path='results/Transformer_decoded_sequences_FULL.csv', mode=FULL)
