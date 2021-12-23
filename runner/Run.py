import tensorflow as tf
from helper.runner import Runner
from helper.loss_functions import CrossEntropyLoss, CrossEntropyLossModified
from models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from readers.AbstractProcessLogReader import TaskModes
from readers import RequestForPaymentLogReader

if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    folder = "results"
    prefix = "result"
    epochs = 5
    batch_size = 10
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = CrossEntropyLoss()
    # loss_fn_mod = CrossEntropyLossModified()
    r = Runner(data, SeqToSeqLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_csv(folder, prefix)
    r = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_csv(folder, prefix)
    # r = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_csv(folder, prefix)
    r = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_csv(folder, prefix)
    # r = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_csv(folder, prefix)
