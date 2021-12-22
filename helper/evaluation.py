from collections import defaultdict

from tensorflow.python.keras.metrics import CategoricalAccuracy
from helper.constants import NUMBER_OF_INSTANCES, SEQUENCE_LENGTH
from models.lstm import SimpleLSTMModelUnidrectional
from models.transformer import TransformerModelUnidirectional
from readers.BPIC12 import BPIC12W
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from tqdm import tqdm
import textdistance

STEP1 = "Step 1: Iterate through data"
STEP2 = "Step 2: Compute Metrics"
STEP3 = "Step 3: Save results"
FULL = 'FULL'
symbol_mapping = {index: char for index, char in enumerate(set([chr(i) for i in range(1, 3000) if len(chr(i)) == 1]))}


def results_by_instance(idx2vocab, start_id, end_id, test_dataset, model, save_path=None, mode='weighted'):
    print("Start results by instance evaluation")
    print(STEP1)
    test_set_list = list()
    for instance in test_dataset:
        test_set_list.append(instance)
    X_test, y_test = zip(*test_set_list)
    X_test, y_test = np.vstack(X_test), np.vstack(y_test).argmax(axis=-1)
    non_zero_indices = np.nonzero(y_test)
    non_zero_mask = np.zeros_like(y_test)
    non_zero_mask[non_zero_indices] = 1

    print(STEP2)
    eval_results = []
    y_pred_masked = model.predict(X_test).argmax(axis=-1)
    iterator = enumerate(zip(X_test, y_test, y_pred_masked))
    for idx, (row_X_test, row_y_test, row_y_pred) in tqdm(iterator, total=len(y_test)):
        last_word_test = np.argmax(row_y_test == 0)
        last_word_pred = np.argmax(row_y_pred == 0)
        # last_word_test = last_word_test + 1 if last_word_test != 0 else len(row_y_test) + 1
        # last_word_pred = last_word_pred + 1 if last_word_pred != 0 else len(row_y_pred) + 1
        longer_sequence_stop = max([last_word_test, last_word_pred])
        # last_word_test, last_word_pred = take_non_zeros_test[0].max() + 1, take_non_zeros_pred[0].max() + 1
        instance_result = {
            "trace": idx,
            f"true_{SEQUENCE_LENGTH}": last_word_test,
            f"pred_{SEQUENCE_LENGTH}": last_word_pred,
        }
        instance_result.update(compute_traditional_metrics(mode, row_y_test[:longer_sequence_stop], row_y_pred[:longer_sequence_stop]))
        instance_result.update(compute_sequence_metrics(row_y_test[:last_word_test], row_y_pred[:last_word_pred]))
        instance_result.update(compute_pred_seq(idx2vocab, row_y_pred, row_y_test, row_X_test, last_word_test, last_word_pred))
        eval_results.append(instance_result)

    results = pd.DataFrame(eval_results)
    print(STEP3)
    print(results)
    if save_path:
        results.to_csv(save_path, index=None)


def compute_traditional_metrics(mode, row_y_test_zeros, row_y_pred_zeros):
    return {
        "acc": accuracy_score(row_y_pred_zeros, row_y_test_zeros),
        "recall": recall_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
        "precision": precision_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
        "f1": f1_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
    }


def compute_sequence_metrics(true_seq, pred_seq):

    true_seq_symbols = "".join([symbol_mapping[idx] for idx in true_seq])
    pred_seq_symbols = "".join([symbol_mapping[idx] for idx in pred_seq])
    dict_instance_distances = {
        "levenshtein": textdistance.levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "damerau_levenshtein": textdistance.damerau_levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "local_alignment": textdistance.smith_waterman.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "global_alignment": textdistance.needleman_wunsch.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "emph_start": textdistance.jaro_winkler.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "longest_subsequence": textdistance.lcsseq.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "longest_substring": textdistance.lcsstr.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "overlap": textdistance.overlap.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "entropy": textdistance.entropy_ncd.normalized_similarity(true_seq_symbols, pred_seq_symbols),
    }
    return dict_instance_distances


def compute_pred_seq(idx2vocab, row_y_pred, row_y_test, row_x_test, last_word_test, last_word_pred):
    x_convert = [f"{i:03d}" for i in row_x_test[:last_word_test]]
    if len(row_x_test[:last_word_test]) <= 1:
        print("Check")
    return {
        "input": " | ".join(["-".join(x_convert[:lim + 1]) for lim in range(len(x_convert))]),
        "true_encoded": " -> ".join([f"{i:03d}" for i in row_y_test[:last_word_test]]),
        "pred_encoded": " -> ".join([f"{i:03d}" for i in row_y_pred[:last_word_pred]]),
        "true_encoded_with_padding": " -> ".join([f"{i:03d}" for i in row_y_test]),
        "pred_encoded_with_padding": " -> ".join([f"{i:03d}" for i in row_y_pred]),
        "true_decoded": " -> ".join([idx2vocab[i] for i in row_y_test[:last_word_test]]),
        "pred_decoded": " -> ".join([idx2vocab[i] for i in row_y_pred[:last_word_pred]]),
    }


def show_predicted_seq(idx2vocab, test_dataset, model, save_path=None, mode=None):
    print("Start results by instance evaluation")
    print(STEP1)
    test_set_list = list()
    for instance in test_dataset:
        test_set_list.append(instance)
    X_test, y_test = zip(*test_set_list)
    X_test, y_test = np.vstack(X_test), np.vstack(y_test).argmax(axis=-1)
    non_zero_indices_test = np.nonzero(y_test)
    non_zero_mask_test = np.zeros_like(y_test)
    non_zero_mask_test[non_zero_indices_test] = 1

    print(STEP2)
    eval_results = []
    y_pred_masked = model.predict(X_test).argmax(axis=-1)
    y_test_masked = y_test
    iterator = enumerate(zip(y_pred_masked, y_test_masked))
    for idx, (row_y_pred, row_y_test) in tqdm(iterator, total=len(y_pred_masked)):
        take_non_zeros_test = np.nonzero(row_y_test)
        take_non_zeros_pred = np.nonzero(row_y_pred)
        last_word_test, last_word_pred = take_non_zeros_test[0].max() + 1, take_non_zeros_pred[0].max() + 1
        eval_results.append(compute_pred_seq(idx2vocab, row_y_pred, row_y_test, X_test[idx], last_word_test, last_word_pred))
        eval_results[-1]["case"] = idx

    results = pd.DataFrame(eval_results)
    print(results)
    if save_path:
        results.to_csv(save_path, index=None)
    return results


def results_by_len(idx2vocab, test_dataset, model, save_path=None, mode='weighted'):
    print("Start results by len evaluation")
    print(STEP1)
    test_set_by_len = defaultdict(list)
    for instance in test_dataset:
        len_instance = np.count_nonzero(instance[0])
        test_set_by_len[len_instance].append(instance)

    print(STEP2)
    eval_results = {}
    for seq_len, instance in test_set_by_len.items():
        X_test, y_test = zip(*instance)
        X_test, y_test = np.vstack(X_test), np.vstack(y_test)
        y_pred = model.predict(X_test)
        y_test_argmax_indices = y_test.argmax(axis=-1)
        y_pred_argmax_indices = y_pred.argmax(axis=-1)
        non_zero_indices = np.nonzero(y_test_argmax_indices.flatten())[0]
        flat_y_test = y_test_argmax_indices.flatten()[non_zero_indices]
        flat_y_pred = y_pred_argmax_indices.flatten()[non_zero_indices]
        labels_2_include = range(1, len(idx2vocab))
        # print(len(eval_results))
        eval_results[seq_len] = {
            NUMBER_OF_INSTANCES: len(y_pred),
            "acc": accuracy_score(flat_y_test, flat_y_pred),
            "precision": precision_score(flat_y_test, flat_y_pred, average=mode, zero_division=0),
            "recall": recall_score(flat_y_test, flat_y_pred, average=mode, zero_division=0),
            "f1": f1_score(flat_y_test, flat_y_pred, average=mode, zero_division=0),
            "dl_distance": damerau_levenshtein_score(y_test_argmax_indices[:, :seq_len], y_pred_argmax_indices[:, :seq_len]),
        }

    results = pd.DataFrame(eval_results).T.sort_index().reset_index().rename(columns={'index': SEQUENCE_LENGTH})
    print(STEP3)
    print(results)
    if save_path:
        results.to_csv(save_path, index=None)


def damerau_levenshtein_score(true_seq, pred_seq):
    if true_seq.ndim == 1 and pred_seq.ndim == 1:
        true_seq = [true_seq]
        pred_seq = [pred_seq]
    true_seq_symbols = ["".join([symbol_mapping[idx] for idx in row]) for row in true_seq]
    pred_seq_symbols = ["".join([symbol_mapping[idx] for idx in row]) for row in pred_seq]
    all_distances = [textdistance.damerau_levenshtein.normalized_similarity(t_seq, p_seq) for t_seq, p_seq in zip(true_seq_symbols, pred_seq_symbols)]
    return np.mean(all_distances)


if __name__ == "__main__":
    data = BPIC12W(debug=False)
    data = data.init_log(True)
    data = data.init_data()
    train_dataset = data.get_train_dataset().take(1000)
    val_dataset = data.get_val_dataset().take(100)
    test_dataset = data.get_test_dataset()

    model = SimpleLSTMModelUnidrectional(data.vocab_len, data.max_len)
    # model = TransformerModel(data.vocab_len, data.max_len)
    model.build((None, data.max_len))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=[CategoricalAccuracy()])
    model.summary()

    model.fit(train_dataset, batch_size=100, epochs=1, validation_data=val_dataset)

    results_by_instance(data.idx2vocab, data.start_id, data.end_id, test_dataset, model, 'junk/test1_.csv')
    # results_by_len(data.idx2vocab, test_dataset, model, 'junk/test2_.csv')
    # show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test3_.csv', mode=None)
    # show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test4_.csv', mode=FULL)
