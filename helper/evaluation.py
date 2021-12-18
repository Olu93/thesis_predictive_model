from collections import defaultdict

from tensorflow.python.keras.metrics import CategoricalAccuracy
from helper.constants import SEQUENCE_LENGTH
from models.lstm import SimpleLSTMModel
from models.transformer import TransformerModel
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
            "num_instances": len(y_pred),
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


def results_by_instance(idx2vocab, test_dataset, model, save_path=None, mode='weighted'):
    print("Start results by instance evaluation")
    print(STEP1)
    test_set_list = list()
    for instance in test_dataset:
        test_set_list.append(instance)
    X_test, y_test = zip(*test_set_list)
    X_test, y_test = np.vstack(X_test), np.vstack(y_test).argmax(axis=-1)
    non_zero_indices = np.nonzero(y_test)
    non_zero_mask = np.zeros_like(X_test)
    non_zero_mask[non_zero_indices] = 1

    print(STEP2)
    eval_results = {}
    seq_lens = non_zero_mask.sum(axis=-1)
    y_pred_masked = model.predict(X_test).argmax(axis=-1) * non_zero_mask
    y_test_masked = y_test * non_zero_mask
    iterator = enumerate(zip(y_test_masked, y_pred_masked))
    for idx, (row_y_test, row_y_pred) in tqdm(iterator, total=len(y_test_masked)):
        labels_2_include = range(1, len(idx2vocab))
        take_non_zeros = np.nonzero(row_y_test)
        take_non_zeros_ = np.nonzero(row_y_pred)
        last_word_test, last_word_pred = take_non_zeros[0].max()+1, take_non_zeros_[0].max()+1
        row_y_pred_zeros, row_y_test_zeros = row_y_pred[take_non_zeros], row_y_test[take_non_zeros]
        eval_results[idx] = {
            SEQUENCE_LENGTH: seq_lens[idx],
            "acc": accuracy_score(row_y_pred_zeros, row_y_test_zeros),
            "recall": recall_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
            "precision": precision_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
            "f1": f1_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
        }
        eval_results[idx].update(compute_sequence_metrics(row_y_test[:last_word_test], row_y_pred[:last_word_pred]))

    results = pd.DataFrame(eval_results).T.sort_index()
    print(STEP3)
    print(results)
    if save_path:
        results.to_csv(save_path, index=None)


def show_predicted_seq(idx2vocab, test_dataset, model, save_path=None, mode=None):
    print("Start results by instance evaluation")
    print(STEP1)
    test_set_list = list()
    for instance in test_dataset:
        test_set_list.append(instance)
    X_test, y_test = zip(*test_set_list)
    X_test, y_test = np.vstack(X_test), np.vstack(y_test).argmax(axis=-1)
    non_zero_indices = np.nonzero(y_test)
    non_zero_mask = np.zeros_like(X_test)
    non_zero_mask[non_zero_indices] = 1

    print(STEP2)
    eval_results = []
    seq_lens = non_zero_mask.sum(axis=-1)
    if mode == FULL:
        seq_lens = np.full(seq_lens.shape, None)
    y_pred_masked = model.predict(X_test).argmax(axis=-1) * non_zero_mask
    y_test_masked = y_test * non_zero_mask
    iterator = enumerate(zip(y_pred_masked, y_test_masked))
    for idx, (row_y_pred, row_y_test) in tqdm(iterator, total=len(y_pred_masked)):
        eval_results.append({
            "true": " -> ".join([idx2vocab[i] for i in row_y_test[:seq_lens[idx]]]),
            "pred": " -> ".join([idx2vocab[i] for i in row_y_pred[:seq_lens[idx]]]),
        })

    results = pd.DataFrame(eval_results)
    print(results)
    if save_path:
        results.to_csv(save_path, index=None)
    return results


def damerau_levenshtein_score(true_seq, pred_seq):
    if true_seq.ndim == 1 and pred_seq.ndim == 1:
        true_seq = [true_seq]
        pred_seq = [pred_seq]
    true_seq_symbols = ["".join([symbol_mapping[idx] for idx in row]) for row in true_seq]
    pred_seq_symbols = ["".join([symbol_mapping[idx] for idx in row]) for row in pred_seq]
    all_distances = [textdistance.damerau_levenshtein.normalized_similarity(t_seq, p_seq) for t_seq, p_seq in zip(true_seq_symbols, pred_seq_symbols)]
    return np.mean(all_distances)

def compute_sequence_metrics(true_seq, pred_seq):

    true_seq_symbols = "".join([symbol_mapping[idx] for idx in true_seq])
    pred_seq_symbols = "".join([symbol_mapping[idx] for idx in pred_seq])
    dict_instance_distances = {
        "levenshtein":textdistance.levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "damerau_levenshtein":textdistance.damerau_levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "local_alignment":textdistance.smith_waterman.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "global_alignment":textdistance.needleman_wunsch.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "emph_start":textdistance.jaro_winkler.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "longest_subsequence":textdistance.lcsseq.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "longest_substring":textdistance.lcsstr.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "overlap":textdistance.overlap.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        "entropy":textdistance.entropy_ncd.normalized_similarity(true_seq_symbols, pred_seq_symbols),
    }
    return dict_instance_distances

if __name__ == "__main__":
    data = BPIC12W(debug=False)
    data = data.init_data()
    train_dataset = data.get_train_dataset()
    val_dataset = data.get_val_dataset()
    test_dataset = data.get_test_dataset()

    model = TransformerModel(data.vocab_len, data.max_len)
    model.build((None, data.max_len))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=[CategoricalAccuracy()])
    model.summary()

    # model.fit(train_dataset, batch_size=1000, epochs=1, validation_data=val_dataset)

    results_by_instance(data.idx2vocab, test_dataset, model, 'junk/test1_.csv')
    results_by_len(data.idx2vocab, test_dataset, model, 'junk/test2_.csv')
    show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test3_.csv', mode=None)
    show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test4_.csv', mode=FULL)
