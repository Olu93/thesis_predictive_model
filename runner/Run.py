import tensorflow as tf
from helper.runner import Runner
from helper.loss_functions import CrossEntropyLoss, CrossEntropyLossModified, SparseCrossEntropyLoss
from models.direct_data_lstm import FullLSTMModelOneWay
from models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from thesis_data_readers.AbstractProcessLogReader import TaskModes
from thesis_data_readers import RequestForPaymentLogReader

if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    folder = "results"
    prefix = "result"
    epochs = 10
    batch_size = 32
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = SparseCrossEntropyLoss()
    # loss_fn_mod = CrossEntropyLossModified()
    r = Runner(data, FullLSTMModelOneWay(data.vocab_len, data.max_len, data.feature_len-1), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
    r = Runner(data, SeqToSeqLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
    r = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
    r = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
    r = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
    r = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(folder, prefix)
