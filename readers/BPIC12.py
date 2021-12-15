from enum import Enum, auto
from readers.AbstractProcessLogReader import AbstractProcessLogReader
import random

class BPIC12W(AbstractProcessLogReader):
    class subsets:
        A = ('A_')
        W = ('W_')
        O = ('O_')
        AW = ('A_', 'W_')
        AO = ('A_', 'O_')
        WO = ('W_', 'O_')
        AWO = ('A_', 'W_', 'O_')
    

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path ='data/financial_log.xes', csv_path ='data/financial_log.csv', **kwargs)
        self.subset = kwargs.get('subsets', BPIC12W.subsets.AW)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data['lifecycle:transition']=='COMPLETE']
        self.data = self.data[self.data[self.activityId].str.startswith(self.subset, na=False)]
    
    
if __name__ == '__main__':
    data = BPIC12W()
    # data = data.init_log(True)
    data = data.init_data()
    ds_counter = data._generate_examples()

    # print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(ds_counter)[0].shape)