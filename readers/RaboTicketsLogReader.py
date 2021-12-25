from readers.AbstractProcessLogReader import AbstractProcessLogReader, CSVLogReader
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import pm4py

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG

class RabobankItilLogReader(CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(
            log_path='data/dataset_bpic2014_rabobank_itil/Detail Incident Activity short.csv',
            csv_path='data/RabobankItil.csv',
            sep=";",
            col_case_id="Incident ID",
            col_event_id="IncidentActivity_Type",
            col_timestamp="DateStamp",
            **kwargs,
        )

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

    def preprocess_level_specialized(self, **kwargs):
        super().preprocess_level_specialized(**kwargs)


    
    
if __name__ == '__main__':
    reader = RabobankItilLogReader(debug=True)
    # reader = reader.init_log(save=1)
    reader = reader.init_data()
    ds_counter = reader.get_dataset()

    print(next(iter(ds_counter.take(10)))[0].shape)
    print(next(iter(ds_counter))[0].shape)
    print(reader.get_data_statistics())
    print(reader.get_example_trace_subset())
    reader.viz_bpmn("white")
    reader.viz_process_map("white")
    reader.viz_dfg("white")