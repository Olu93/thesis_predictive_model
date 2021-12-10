from readers.AbstractProcessLogReader import AbstractProcessLogReader, TaskModes
import tensorflow as tf

class RequestForPaymentLogReader(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__('data/RequestForPayment.xes_', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[
            "case:Cost Type",
            'case:Rfp_id',
            'case:Activity',
            'case:Cost Type',
        ])

    def preprocess_level_specialized(self, **kwargs):
        self.data[self.activityId] = self.data[self.activityId].replace(
            'Request For Payment ',
            '',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )
        self.data[self.caseId] = self.data[self.caseId].replace(
            'request for ',
            '',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )

if __name__ == '__main__':
    data = RequestForPaymentLogReader(mode=TaskModes.SIMPLE)
    ds_counter = tf.data.Dataset.from_generator(data._generate_examples, args=[True], output_types=(tf.int64, tf.int64), output_shapes = ((None,), (None,)))

    print(next(iter(ds_counter.repeat().batch(10).take(10))))