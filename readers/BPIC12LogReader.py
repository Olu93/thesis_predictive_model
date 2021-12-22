from enum import Enum, auto
from readers.AbstractProcessLogReader import AbstractProcessLogReader
import random

class BPIC12LogReader(AbstractProcessLogReader):
    class subsets:
        A = ('A_')
        W = ('W_')
        O = ('O_')
        AW = ('A_', 'W_')
        AO = ('A_', 'O_')
        WO = ('W_', 'O_')
        AWO = ('A_', 'W_', 'O_')
    

    def __init__(self, **kwargs) -> None:
        self.subset = kwargs.get('subset', BPIC12LogReader.subsets.AW)
        if 'subset' in kwargs:
            del kwargs['subset']
        super().__init__(log_path ='data/dataset_bpic2012_financial_loan/financial_log.xes', csv_path ='data/financial_log.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data['lifecycle:transition']=='COMPLETE']
        self.data = self.data[self.data[self.col_activity_id].str.startswith(self.subset, na=False)]
    
    
if __name__ == '__main__':
    data = BPIC12LogReader()
    data = data.init_log(True)
    data = data.init_data()
    ds_counter = data._generate_examples()

    # print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(ds_counter)[0].shape)