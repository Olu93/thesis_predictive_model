from readers.AbstractProcessLogReader import AbstractProcessLogReader
import random

class BPIC12W(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(log_path ='data/financial_log.xes', csv_path ='data/financial_log.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data['lifecycle:transition']=='COMPLETE'][self.data[self.activityId].str.startswith("W_", na=False)]

    def _train_val_test_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.3)
        train_traces = traces[:len_train_traces]
        len_val_traces = int(len_dataset * 0.6)
        val_traces = traces[:len_val_traces]
        len_test_traces = int(len_dataset * 0.1)
        test_traces = traces[:len_test_traces]
        return train_traces, val_traces, test_traces
    
    
if __name__ == '__main__':
    data = BPIC12W().init_log(save=True).init_data()
    ds_counter = data.get_train_dataset()

    print(next(iter(ds_counter.repeat().batch(10).take(10))))