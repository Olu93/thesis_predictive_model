from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from readers import RequestForPaymentLogReader

if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=True)
    dataset = tf.data.Dataset.from_generator(
        data._generate_examples,
        args=[True],
        output_types=(tf.int64, tf.int64),
        output_shapes=((None, ), (
            None,
            None,
        )),
    ).batch(1)

    model = Sequential()
    model.add(InputLayer(input_shape=(data.max_len,)))
    model.add(Embedding(data.vocab_len, 128))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(data.vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    model.summary()
    sample = next(iter(dataset))
    # model(sample[0])
    # train_y_onehot = tf.keras.utils.to_categorical(train_y, num_classes=data.vocab_len, dtype='float32')
    model.fit(dataset, batch_size=10, epochs=3)
