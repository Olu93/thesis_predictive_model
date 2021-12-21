from readers.AbstractProcessLogReader import AbstractProcessLogReader
import pandas as pd


class DomesticDeclarationsLogReader(AbstractProcessLogReader):
    COL_DECLARATION_NUM = "case:DeclarationNumber"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path='data/dataset_bpic2020_tu_travel/DomesticDeclarations.xes', csv_path='data/DomesticDeclarations.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

    def preprocess_level_specialized(self, **kwargs):
        self.data[self.col_activity_id] = self.data[self.col_activity_id].replace(
            'Declaration ',
            'DEC',
            regex=True,
        ).replace(
            ' by ',
            ' ',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )
        self.data[self.col_case_id] = self.data[self.col_case_id].replace(
            ' ',
            '_',
            regex=True,
        )
        self.data[DomesticDeclarationsLogReader.COL_DECLARATION_NUM] = self.data[DomesticDeclarationsLogReader.COL_DECLARATION_NUM].replace(
            'declaration number ',
            'no_',
            regex=True,
        )


if __name__ == '__main__':
    reader = DomesticDeclarationsLogReader()
    reader = reader.init_log(save=1) 
    reader = reader.init_data()
    ds_counter = reader.get_train_dataset()

    print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(iter(ds_counter))[0].shape)
    print(reader.get_data_statistics())
    # print(data.get_example_trace_subset())
    reader.viz_bpmn("white")
    reader.viz_process_map("white")
    reader.viz_dfg("white")