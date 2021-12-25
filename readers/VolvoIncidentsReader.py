from readers.AbstractProcessLogReader import AbstractProcessLogReader
import pandas as pd


class VolvoIncidentsReader(AbstractProcessLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path='data/dataset_bpic2013_volvo_incidents/bpi_challenge_2013_incidents.xes', csv_path='data/VolvoIncidents.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

    def preprocess_level_specialized(self, **kwargs):
        super().preprocess_level_specialized(**kwargs)

    # def preprocess_level_specialized(self, **kwargs):
        
    #     # self.data = self.data[self.data['lifecycle:transition']=='COMPLETE']


if __name__ == '__main__':
    reader = VolvoIncidentsReader()
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