import io
from tensorflow.python.keras.engine.training import Model
import tqdm
import json
from tensorflow.keras.optimizers import Adam
import pathlib
from thesis_data_readers import AbstractProcessLogReader
from thesis_data_readers.AbstractProcessLogReader import DatasetModes, ShapeModes
from ..helper.evaluation import FULL, results_by_instance, results_by_instance_seq2seq, results_by_len, show_predicted_seq
from ..helper.loss_functions import SparseAccuracyMetric, SparseCrossEntropyLoss


class Runner(object):
    statistics = {}

    def __init__(self,
                 data: AbstractProcessLogReader,
                 model: Model,
                 epochs: int,
                 batch_size: int,
                 adam_init: float,
                 num_train: int = None,
                 num_val: int = None,
                 num_test: int = None):
        self.data = data
        self.model = model
        self.train_dataset = self.data.get_dataset(batch_size, DatasetModes.TRAIN)
        self.val_dataset = self.data.get_dataset(batch_size, DatasetModes.VAL)
        self.test_dataset = self.data.get_dataset(batch_size, DatasetModes.TEST)
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

    def train_model(self, loss_fn=SparseCrossEntropyLoss(), metrics=[SparseAccuracyMetric()], label=None, train_dataset=None, val_dataset=None):
        label = label or self.label
        train_dataset = train_dataset or self.train_dataset
        val_dataset = val_dataset or self.val_dataset
        self.metrics = metrics
        self.loss_fn = loss_fn

        print(f"{label}:")
        self.model.compile(loss=loss_fn, optimizer=Adam(self.adam_init), metrics=metrics)
        self.model.summary()

        # vd_1, vd_2 = [], []
        # for datapoint in val_dataset:
        #     vd_1.extend((datapoint[0], ))
        #     vd_2.extend((datapoint[1], ))
        # for epoch in tqdm.tqdm(range(self.epochs)):
        #     for X, y in train_dataset:
        #         train_results = self.model.fit(X, y[0], verbose=1)
        #         self.statistics[epoch] = {"history": train_results}
        #     val_loss, val_acc = self.model.evaluate(vd_1[0], vd_2[0])
        #     self.statistics[epoch].update({
        #         "train_loss" : train_results.history['loss'][-1],
        #         "train_acc" : train_results.history['accuracy'][-1],
        #         "val_loss" : val_loss,
        #         "val_acc" : val_acc,
        #     })
        self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs)

        return self

    def evaluate(self, save_path="results", prefix="full", label=None, test_dataset=None, dont_save=False):
        test_dataset = test_dataset or self.test_dataset
        self.results = results_by_instance_seq2seq(self.data.idx2vocab, self.start_id, self.end_id, test_dataset, self.model)
        if not dont_save:
            label = label or self.label
            save_path = save_path or self.save_path
            self.results.to_csv(pathlib.Path(save_path) / (f"{prefix}_{label}.csv"))
        return self

    def save_model(self, save_path="build", prefix="full", label=None):
        label = label or self.label
        save_path = save_path or self.save_path
        target_folder = pathlib.Path(save_path) / (f"{prefix}_{label}")
        self.model.save(target_folder)
        self.model_path = target_folder
        json.dump(self._transform_model_history(), io.open(target_folder / 'history.json', 'w'), indent=4, sort_keys=True)
        return self

    def _transform_model_history(self):
        tmp_history = dict(self.history.history)
        tmp_history["epochs"] = self.history.epoch
        history = {
            "history": tmp_history,
            "params": self.history.params,
        }
        return history