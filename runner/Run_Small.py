import tensorflow as tf
from helper.runner import Runner
from models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from readers.AbstractProcessLogReader import TaskModes
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    folder = "results"
    epochs = 1
    batch_size = 10
    adam_init = 0.001
    num_instances = {"num_train": 1000, "num_val": 100, "num_test": 1000}
    r = Runner(data, SeqToSeqLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
