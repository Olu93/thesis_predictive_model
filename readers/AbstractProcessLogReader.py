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
    data_path: str = None
    _original_data: pd.DataFrame = None
    data: pd.DataFrame = None
    debug: bool = False
    caseId: str = None
    activityId: str = None
    _vocab: dict = None
    mode: TaskModes = TaskModes.SIMPLE
    padding_token: str = "<PAD>"
    transform = None

    def __init__(self,
                 data_path: str,
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
        self.data_path = pathlib.Path(data_path)
        self.caseId = caseId
        self.activityId = activityId
        self.log = pm4py.read_xes(self.data_path.as_posix())
        if self.debug:
            print(self.log[1])  #prints the first event of the first trace of the given log
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if self.debug:
            display(self._original_data.head())
        self.preprocess_level_general()
        self.preprocess_level_specialized()
        self.register_vocabulary()
        self.compute_sequences()

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
        all_unique_tokens = list(self.data[self.activityId].unique()) + [self.padding_token]
        self.vocab_len = len(all_unique_tokens) + 1

        self._vocab = {word: idx for idx, word in enumerate(all_unique_tokens, 1)}
        self._vocab_r = {idx: word for word, idx in self._vocab.items()}

    def compute_sequences(self):
        grouped_traces = list(self.data.groupby(by=self.caseId))

        self._traces = {idx: list(df[self.activityId].values) for idx, df in grouped_traces}
        self.length_distribution = Counter([len(tr) for tr in self._traces.values()])
        self.max_len = self.length_distribution.most_common(1)[0][0]
        self.instantiate_dataset()

    def instantiate_dataset(self):
        loader = tqdm(self._traces.values(), total=len(self._traces))
        loader = ([self.vocab2idx[word] for word in tr] for tr in loader)
        if self.mode == TaskModes.SIMPLE:
            self.traces = ([tr[0:-1], tr[1:]] for tr in loader if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE:
            self.traces = ([tr[0:end - 1], tr[1:end]] for tr in loader for end in range(2, len(tr) + 1) if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE_RANDOM:
            tmp_traces = [tr[random.randint(0, len(tr) - 1):] for tr in loader for sample in self._heuristic_bounded_sample_size(tr) if len(tr) > 1]
            self.traces = [tr[:random.randint(2, len(tr))] for tr in tqdm(tmp_traces, desc="random-samples") if len(tr) > 1]

        if self.mode == TaskModes.FINAL_OUTCOME:
            self.traces = ([tr[0:-1], tr[-1] * (len(tr) - 1)] for tr in loader if len(tr) > 1)

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

    def _train_test_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.8)
        train_traces = traces[:len_train_traces]
        # len_val_traces = int(len_dataset * 0.6)
        # val_traces = traces[:len_val_traces]
        len_test_traces = int(len_dataset * 0.2)
        test_traces = traces[:len_test_traces]
        return train_traces, test_traces

    def _train_val_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.6)
        train_traces = traces[:len_train_traces]
        len_val_traces = int(len_dataset * 0.4)
        val_traces = traces[:len_val_traces]
        return train_traces, val_traces

    @property
    def tokens(self) -> List[str]:
        return list(self._vocab.keys())

    @property
    def vocab2idx(self) -> List[str]:
        return self._vocab

    @property
    def idx2vocab(self) -> List[str]:
        return self._vocab_r

    def _generate_examples(self, is_train=True) -> Iterator[Dict[str, list]]:
        """Generator of examples for each split."""
        data = zip(self.trace_train, self.target_train) if is_train else zip(self.trace_train, self.target_train)
        for trace, target in data:
            yield trace, to_categorical(target, num_classes=self.vocab_len)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     X, y = [self.vocab2idx[wrd] for wrd in self.train_traces[idx][:-1]], [self.vocab2idx[self.train_traces[idx][-1]]]
    #     sample = {'sequence': X, 'target': y}

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample
