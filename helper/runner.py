import tqdm
from helper.evaluation import FULL, results_by_instance, results_by_instance_seq2seq, results_by_len, show_predicted_seq
from tensorflow.keras.optimizers import Adam
import pathlib
from readers import AbstractProcessLogReader

from readers.AbstractProcessLogReader import DatasetModes, ShapeModes


class Runner(object):
    statistics = {}
    
    def __init__(self,
                 data: AbstractProcessLogReader,
                 model,
                 epochs,
                 batch_size,
                 adam_init,
                 feature_mode: ShapeModes,
                 target_mode: ShapeModes,
                 num_train: int = None,
                 num_val: int = None,
                 num_test: int = None):
        self.data = data
        self.model = model
        self.feature_mode = feature_mode
        self.target_mode = target_mode
        self.train_dataset = self.data.get_dataset(batch_size, DatasetModes.TRAIN, feature_mode, target_mode)
        self.val_dataset = self.data.get_dataset(batch_size, DatasetModes.VAL, feature_mode, target_mode)
        self.test_dataset = self.data.get_dataset(batch_size, DatasetModes.TEST, feature_mode, target_mode)
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

    def get_results_from_model(self, loss_fn="categorical_crossentropy", label=None, train_dataset=None, val_dataset=None, test_dataset=None):
        label = label or self.label
        train_dataset = train_dataset or self.train_dataset
        val_dataset = val_dataset or self.val_dataset
        test_dataset = test_dataset or self.test_dataset

        print(f"{label}:")
        self.model.compile(loss=loss_fn, optimizer=Adam(self.adam_init), metrics=['accuracy'])
        self.model.summary()

        vd_1 = []
        vd_2 = []
        for datapoint in val_dataset:
            vd_1.extend((datapoint[0], ))
            vd_2.extend((datapoint[1], ))
        for epoch in tqdm.tqdm(range(self.epochs)):
            for X, y in train_dataset:
                train_results = self.model.fit(datapoint[0], datapoint[1][0], verbose=0)
                self.statistics[epoch] = {"history": train_results}
            val_loss, val_acc = self.model.evaluate(vd_1[0], vd_2[0])
            self.statistics[epoch].update({
                "train_loss" : train_results.history['loss'][-1],
                "train_acc" : train_results.history['accuracy'][-1],
                "val_loss" : val_loss,
                "val_acc" : val_acc,
            })

        self.results = results_by_instance_seq2seq(self.data.idx2vocab, self.start_id, self.end_id, test_dataset, self.model)
        return self

    def save_csv(self, save_path="results", prefix="full", label=None):
        label = label or self.label
        save_path = save_path or self.save_path
        self.results.to_csv(pathlib.Path(save_path) / (f"{prefix}_{label}.csv"))
        return self
