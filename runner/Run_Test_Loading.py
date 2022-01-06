import tensorflow as tf
from helper.runner import Runner
from helper.loss_functions import CrossEntropyLoss, CrossEntropyLossModified, SparseAccuracyMetric, SparseCrossEntropyLoss
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
    epochs = 2
    batch_size = 64
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = SparseCrossEntropyLoss()
    metric = SparseAccuracyMetric()
    
    r5 = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(results_folder, prefix)
    model = tf.keras.models.load_model(r5.save_model(build_folder, prefix).model_path, custom_objects={'sparse_accuracy_metric': metric})
    print(model)
    
    
    
