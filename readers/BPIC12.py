from readers.AbstractProcessLogReader import AbstractProcessLogReader
import random

class BPIC12W(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(log_path ='data/financial_log.xes', csv_path ='data/financial_log.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data['lifecycle:transition']=='COMPLETE']#[self.data[self.activityId].str.startswith("W_", na=False)]
    
    
if __name__ == '__main__':
    data = BPIC12W()
    # data = data.init_log(True)
    data = data.init_data()
    ds_counter = data._generate_examples()

    # print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(ds_counter)[0].shape)