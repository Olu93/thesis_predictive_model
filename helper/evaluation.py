from collections import defaultdict

from tensorflow.python.keras.metrics import CategoricalAccuracy
from models.lstm import ProcessLSTMSimpleModel
from readers.BPIC12 import BPIC12W
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from tqdm import tqdm

def results_by_len(data, test_dataset, model, save_path = None):
    print("Step 1: Segmented data by len")
    test_set_by_len = defaultdict(list)
    for instance in test_dataset:
        len_instance = np.count_nonzero(instance[0])
        test_set_by_len[len_instance].append(instance)
    
    print("Step 2: Compute metrics")
    eval_results = {}
    for seq_len, instance in test_set_by_len.items():
        X_test, y_test = zip(*instance)
        X_test, y_test = np.vstack(X_test), np.vstack(y_test)
        y_pred = model.predict(X_test)
        non_zero_indices = np.nonzero(y_test.argmax(axis=-1).flatten())[0]
        flat_y_test = y_test.argmax(axis=-1).flatten()[non_zero_indices]
        flat_y_pred = y_pred.argmax(axis=-1).flatten()[non_zero_indices]
        labels_2_include = range(1, data.vocab_len)
        eval_results[seq_len] = {
            "num_instances": len(y_pred), 
            "acc": accuracy_score(flat_y_test, flat_y_pred),
            "recall": recall_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
            "precision": precision_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
            "f1": f1_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
        }

    results = pd.DataFrame(eval_results).T.sort_index()
    print(results)
    print("Step 3: Save results")
    if save_path:
        results.to_csv(save_path)


def results_by_instance(data, test_dataset, model, save_path = None):
    print("Step 1: Iterate through data")
    test_set_list = list()
    for instance in test_dataset:
        test_set_list.append(instance)
    
    print("Step 2: Compute metrics")
    eval_results = {}
    for idx, (X_test, y_test) in tqdm(enumerate(test_set_list), total=len(test_set_list)):
        X_test, y_test = np.vstack(X_test), np.vstack(y_test)
        y_pred = model.predict(X_test)
        non_zero_indices = np.nonzero(y_test.argmax(axis=-1).flatten())[0]
        flat_y_test = y_test.argmax(axis=-1).flatten()[non_zero_indices]
        flat_y_pred = y_pred.argmax(axis=-1).flatten()[non_zero_indices]
        labels_2_include = range(1, data.vocab_len)
        eval_results[idx] = {
            "len_seq": len(flat_y_test), 
            "acc": accuracy_score(flat_y_test, flat_y_pred),
            "recall": recall_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
            "precision": precision_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
            "f1": f1_score(flat_y_test, flat_y_pred, labels=labels_2_include, average='micro'),
        }

    results = pd.DataFrame(eval_results).T.sort_index()
    print(results)
    print("Step 3: Save results")
    if save_path:
        results.to_csv(save_path)

if __name__ == "__main__":
    data = BPIC12W(debug=False)
    data = data.init_data()
    train_dataset = data.get_train_dataset()
    val_dataset = data.get_val_dataset()
    test_dataset = data.get_test_dataset()

    model = ProcessLSTMSimpleModel(data.vocab_len, data.max_len)
    model.build((None, data.max_len))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=[CategoricalAccuracy()])
    model.summary()

    # model.fit(train_dataset, batch_size=10, epochs=1, validation_data=val_dataset)

    results_by_instance(data, test_dataset, model, 'test_results_by_instance.csv')
    results_by_len(data, test_dataset, model, 'test_results_by_len.csv')
    
