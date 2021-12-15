from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from helper.evaluation import results_by_instance, results_by_len
from models.lstm import ProcessLSTMSimpleModel

from readers.BPIC12 import BPIC12W
from readers import RequestForPaymentLogReader

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

if __name__ == "__main__":
    data = BPIC12W(debug=False)
    # data = data.init_log(save=True)
    data = data.init_data()
    train_dataset = data.get_train_dataset()
    val_dataset = data.get_val_dataset()
    test_dataset = data.get_test_dataset()

    model = ProcessLSTMSimpleModel(data.vocab_len, data.max_len)
    model.build((None, data.max_len))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    model.summary()
    

    model.fit(train_dataset, batch_size=10, epochs=2, validation_data=val_dataset)
    
    results_by_instance(data, test_dataset, model, 'test_results_by_instance.csv')
    results_by_len(data, test_dataset, model, 'test_results_by_len.csv')

