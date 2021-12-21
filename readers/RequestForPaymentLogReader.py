from readers.AbstractProcessLogReader import AbstractProcessLogReader, TaskModes
import tensorflow as tf


class RequestForPaymentLogReader(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(log_path='data/dataset_bpic2020_tu_travel/RequestForPayment.xes', csv_path='data/RequestForPayment.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[
            "case:Cost Type",
            'case:Rfp_id',
            'case:Activity',
            'case:Cost Type',
        ])

    def preprocess_level_specialized(self, **kwargs):
        self.data[self.col_activity_id] = self.data[self.col_activity_id].replace(
            'Request For Payment ',
            '',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )
        self.data[self.col_case_id] = self.data[self.col_case_id].replace(
            'request for ',
            '',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )


if __name__ == '__main__':
    data = RequestForPaymentLogReader(mode=TaskModes.SIMPLE).init_log(save=True).init_data()
    ds_counter = data.get_train_dataset()

    print(next(iter(ds_counter.repeat().batch(10).take(10))))