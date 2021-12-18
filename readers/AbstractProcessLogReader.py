import math
import random
from enum import Enum, auto
from typing import Counter, Dict, Iterable, Iterator, List, Union
import pathlib
import pandas as pd
import pm4py
from IPython.display import display
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.util import constants
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.visualization.petrinet import visualizer as petrinet_visualization
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
import tensorflow_datasets as tfds
import io
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


class TaskModes(Enum):
    SIMPLE = auto()
    EXTENSIVE = auto()
    EXTENSIVE_RANDOM = auto()
    FINAL_OUTCOME = auto()


class DatasetModes(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class AbstractProcessLogReader():
    """DatasetBuilder for my_dataset dataset."""

    log = None
    log_path: str = None
    _original_data: pd.DataFrame = None
    data: pd.DataFrame = None
    debug: bool = False
    caseId: str = None
    activityId: str = None
    _vocab: dict = None
    mode: TaskModes = TaskModes.SIMPLE
    padding_token: str = "<P>"
    end_token: str = "<E>"
    start_token: str = "<S>"
    transform = None

    def __init__(self,
                 log_path: str,
                 csv_path: str,
                 caseId: str = 'case:concept:name',
                 activityId: str = 'concept:name',
                 debug=False,
                 mode: TaskModes = TaskModes.SIMPLE,
                 max_tokens: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_len = None
        self.debug = debug
        self.mode = mode
        self.log_path = pathlib.Path(log_path)
        self.csv_path = pathlib.Path(csv_path)
        self.caseId = caseId
        self.activityId = activityId

    def init_log(self, save=False):
        self.log = pm4py.read_xes(self.log_path.as_posix())
        if self.debug:
            print(self.log[1])  #prints the first event of the first trace of the given log
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if save:
            self._original_data.to_csv(self.csv_path, index=False)
        return self

    def init_data(self):
        self._original_data = self._original_data if self._original_data is not None else pd.read_csv(self.csv_path)
        if self.debug:
            display(self._original_data.head())
        self.preprocess_level_general()
        self.preprocess_level_specialized()
        self.register_vocabulary()
        self.compute_sequences()
        return self

    def show_dfg(self):
        dfg = dfg_discovery.apply(self.log)
        gviz = dfg_visualization.apply(dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        dfg_visualization.view(gviz)

    @property
    def original_data(self):
        return self._original_data.copy()

    @original_data.setter
    def original_data(self, data: pd.DataFrame):
        self._original_data = data

    def preprocess_level_general(self, **kwargs):
        self.data = self.original_data
        remove_cols = kwargs.get('remove_cols')
        if remove_cols:
            self.data = self.original_data.drop(remove_cols, axis=1)

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data

    def register_vocabulary(self):
        all_unique_tokens = list(self.data[self.activityId].unique())

        self._vocab = {word: idx for idx, word in enumerate(all_unique_tokens, 1)}
        self._vocab[self.padding_token] = 0
        self._vocab[self.start_token] = len(self._vocab) + 1
        self._vocab[self.end_token] = len(self._vocab) + 1
        self.vocab_len = len(self._vocab) + 1
        self._vocab_r = {idx: word for word, idx in self._vocab.items()}

    def compute_sequences(self):
        grouped_traces = list(self.data.groupby(by=self.caseId))

        self._traces = {idx: tuple(df[self.activityId].values) for idx, df in grouped_traces}

        self.length_distribution = Counter([len(tr) for tr in self._traces.values()])
        self.max_len = max(list(self.length_distribution.keys())) + 2
        self.min_len = min(list(self.length_distribution.keys())) + 2
        self.instantiate_dataset()

    def instantiate_dataset(self):
        print("Preprocess data")
        loader = tqdm(self._traces.values(), total=len(self._traces))
        start_id = [self.vocab2idx[self.start_token]]
        end_id = [self.vocab2idx[self.end_token]]

        loader = (start_id + [self.vocab2idx[word] for word in tr] + end_id for tr in loader)
        if self.mode == TaskModes.SIMPLE:
            self.traces = ([tr[0:], tr[1:]] for tr in loader if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE:
            self.traces = ([tr[0:end - 1], tr[1:end]] for tr in loader for end in range(2, len(tr) + 1) if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE_RANDOM:
            tmp_traces = [tr[random.randint(0, len(tr) - 1):] for tr in loader for sample in self._heuristic_bounded_sample_size(tr) if len(tr) > 1]
            self.traces = [tr[:random.randint(2, len(tr))] for tr in tqdm(tmp_traces, desc="random-samples") if len(tr) > 1]

        if self.mode == TaskModes.FINAL_OUTCOME:
            self.traces = ([tr[0:], tr[-2] * (len(tr) - 1)] for tr in loader if len(tr) > 1)

        self.traces, self.targets = zip(*[tr_couple for tr_couple in tqdm(self.traces)])
        self.padded_traces = pad_sequences(self.traces, self.max_len, padding='post')
        self.padded_targets = pad_sequences(self.targets, self.max_len, padding='post')

        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.padded_traces, self.padded_targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")

    def _heuristic_sample_size(self, sequence):
        return range((len(sequence)**2 + len(sequence)) // 4)

    def _heuristic_bounded_sample_size(self, sequence):
        return range(min((len(sequence)**2 + len(sequence) // 4), 5))

    def _generate_examples(self, set_name='train') -> Iterator[Dict[str, list]]:
        """Generator of examples for each split."""
        data = None
        set_name = set_name.decode('utf8') if type(set_name) != str else set_name
        if set_name == 'train':
            data = zip(self.trace_train, self.target_train)
        if set_name == 'val':
            data = zip(self.trace_val, self.target_val)
        if set_name == 'test':
            data = zip(self.trace_test, self.target_test)
        for trace, target in data:
            yield trace, to_categorical(target, num_classes=self.vocab_len)

    def get_train_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generate_examples,
            args=['train'],
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, ), (
                None,
                None,
            )),
        ).batch(1)

    def get_val_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generate_examples,
            args=['val'],
            output_types=(tf.int64, tf.int64),
            output_shapes=((None, ), (
                None,
                None,
            )),
        ).batch(1)

    def get_test_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generate_examples,
            args=['test'],
            output_types=(tf.int64, tf.int64),
            output_shapes=((None, ), (
                None,
                None,
            )),
        ).batch(1)

    @property
    def tokens(self) -> List[str]:
        return list(self._vocab.keys())
    
    @property
    def start_id(self) -> List[str]:
        return self.vocab2idx[self.start_token]
    
    @property
    def end_id(self) -> List[str]:
        return self.vocab2idx[self.end_token]

    @property
    def vocab2idx(self) -> List[str]:
        return self._vocab

    @property
    def idx2vocab(self) -> List[str]:
        return self._vocab_r

    def get_data_statistics(self):
        return {
            "log_size": self._log_size,
            "min_seq_len": self._min_seq_len,
            "max_seq_len": self._max_seq_len,
            "distinct_trace_ratio": self._distinct_trace_ratio,
            "num_distinct_events": self._num_distinct_events,
        }

    @property
    def _log_size(self):
        return len(self._traces)

    @property
    def _distinct_trace_ratio(self):
        return len(set(tuple(tr) for tr in self._traces.values())) / self._log_size

    @property
    def _min_seq_len(self):
        return self.min_len - 2

    @property
    def _max_seq_len(self):
        return self.max_len - 2

    @property
    def _num_distinct_events(self):
        return len([ev for ev in self.vocab2idx.keys() if ev not in [self.padding_token]])

    def get_example_trace_subset(self, num_traces=10):
        random_starting_point = random.randint(0, self._log_size - num_traces - 1)
        df_traces = pd.DataFrame(self._traces.items()).set_index(0).sort_index()
        example = df_traces[random_starting_point:random_starting_point + num_traces]
        return [val for val in example.values]


if __name__ == '__main__':
    data = AbstractProcessLogReader(log_path='data/dataset_bpic2020_tu_travel/RequestForPayment.xes', csv_path='data/RequestForPayment.csv',
                                    mode=TaskModes.SIMPLE).init_log(save=True).init_data()
    ds_counter = data.get_train_dataset()

    print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(iter(ds_counter))[0].shape)
    print(data.get_data_statistics())
    print(data.get_example_trace_subset())