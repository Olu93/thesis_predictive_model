from readers.AbstractProcessLogReader import AbstractProcessLogReader
import pandas as pd

class PermitLogReader(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(log_path='data/dataset_bpic2020_tu_travel/PermitLog.xes',csv_path='data/PermitLog.csv', **kwargs)

    def preprocess_level_general(self):
        
        super().preprocess_level_general(remove_cols=[
            'case:ProjectNumber',
            'case:TaskNumber',
            'case:ActivityNumber'
        ])
        
    
    # def prepr
    #     print("prep")
        
        
if __name__ == '__main__':
    reader = PermitLogReader()
    reader = reader.init_log(save=1)
    reader = reader.init_data()
    ds_counter = reader.get_dataset()

    print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(iter(ds_counter))[0].shape)
    print(reader.get_data_statistics())
    # print(data.get_example_trace_subset())
    reader.viz_bpmn("white")
    reader.viz_process_map("white")
    reader.viz_dfg("white")