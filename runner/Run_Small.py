import tensorflow as tf
from helper.runner import Runner
from models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from models.seq2seq_lstm import Seq2seqLSTMModelUnidrectional
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
    r1 = Runner(data, Seq2seqLSTMModelUnidrectional(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r1 = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r2 = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r3 = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
    r4 = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model().save_csv(folder, "test")
