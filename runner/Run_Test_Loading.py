import tensorflow as tf
from helper.runner import Runner
from helper.loss_functions import CrossEntropyLoss, CrossEntropyLossModified, SparseCrossEntropyLoss
from models.direct_data_lstm import FullLSTMModelOneWay
from models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from thesis_data_readers.AbstractProcessLogReader import TaskModes
from thesis_data_readers import RequestForPaymentLogReader, VolvoIncidentsReader

if __name__ == "__main__":
    data = VolvoIncidentsReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    results_folder = "results"
    build_folder = "models_bin"
    prefix = "result"
    epochs = 5
    batch_size = 64
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = SparseCrossEntropyLoss()
    
    r5 = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).get_results_from_model(loss_fn).save_results(results_folder, prefix)
    r5.save_model(build_folder, prefix)
