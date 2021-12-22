from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from helper.evaluation import FULL, results_by_instance, results_by_instance_seq2seq, results_by_len, show_predicted_seq
from models.lstm import SimpleLSTMModelTwoWay, SimpleLSTMModelOneWay
from models.seq2seq_lstm import Seq2seqLSTMModelUnidrectional
from models.transformer import TransformerModelTwoWay, TransformerModelOneWay
from readers.AbstractProcessLogReader import AbstractProcessLogReader, TaskModes
import pathlib
from readers.BPIC12LogReader import BPIC12LogReader
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Runner(object):
    def __init__(self, data, model, epochs, batch_size, adam_init, num_train=None, num_val=None, num_test=None):
        self.data = data
        self.model = model
        self.train_dataset = self.data.get_train_dataset()
        self.val_dataset = self.data.get_val_dataset()
        self.test_dataset = self.data.get_test_dataset()
        if num_train:
            self.train_dataset = self.train_dataset.take(num_train)
        if num_val:
            self.val_dataset = self.val_dataset.take(num_val)
        if num_test:
            self.test_dataset = self.test_dataset.take(num_test)

        self.epochs = epochs
        self.batch_size = batch_size
        self.adam_init = adam_init
        self.start_id = data.start_id
        self.end_id = data.end_id

        self.label = model.name

    def get_results_from_model(self, label=None, train_dataset=None, val_dataset=None, test_dataset=None):
        label = label or self.label
        train_dataset = train_dataset or self.train_dataset
        val_dataset = val_dataset or self.val_dataset
        test_dataset = test_dataset or self.test_dataset

        print(f"{label}:")
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(self.adam_init), metrics=['accuracy'])
        self.model.summary()
        self.model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_dataset)
        self.results = results_by_instance_seq2seq(self.data.idx2vocab, self.start_id, self.end_id, test_dataset, self.model)
        return self

    def save_csv(self, save_path="results", prefix="full", label=None):
        label = label or self.label
        save_path = save_path or self.save_path
        self.results.to_csv(pathlib.Path(save_path) / (f"{prefix}_{label}.csv"))
        return self


if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=False, mode=TaskModes.ENCODER_DECODER)
    # data = data.init_log(save=True)
    data = data.init_data()
    folder = "results"
    epochs = 1
    batch_size = 10
    adam_init = 0.001
    num_instances = {"num_train": 100, "num_val": 100, "num_test": 100}
    r1 = Runner(data, Seq2seqLSTMModelUnidrectional(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    # r1 = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    # r2 = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    # r3 = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    # r4 = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
