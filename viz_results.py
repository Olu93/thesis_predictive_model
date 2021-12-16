# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

from helper.constants import NUMBER_OF_INSTANCES, SEQUENCE_LENGTH
# %%
METRIC = "f1"
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = pd.read_csv('results_Transformer_by_len.csv').set_index(SEQUENCE_LENGTH).sort_index()
lstm_results = pd.read_csv('results_LSTM_by_len.csv').set_index(SEQUENCE_LENGTH).sort_index()
ax.plot(lstm_results[METRIC].index, lstm_results[METRIC], label="LSTM")
ax.plot(transformer_results[METRIC].index, transformer_results[METRIC], label="Transformer")
ax.legend()
ax.set_title(f'{METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = pd.read_csv('results_Transformer_by_instance.csv').set_index(SEQUENCE_LENGTH).sort_index()[METRIC]
lstm_results = pd.read_csv('results_LSTM_by_instance.csv').set_index(SEQUENCE_LENGTH).sort_index()[METRIC]
ax.plot(lstm_results.index, lstm_results, label="LSTM")
ax.plot(transformer_results.index, transformer_results, label="Transformer")
ax.legend()
ax.set_title(f'{METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = transformer_results.sort_values(NUMBER_OF_INSTANCES)
lstm_results = pd.read_csv('results_LSTM_by_instance.csv').set_index(SEQUENCE_LENGTH).sort_index()[METRIC]
ax.plot(lstm_results.index, lstm_results, label="LSTM")
ax.plot(transformer_results.index, transformer_results, label="Transformer")
ax.legend()
ax.set_title(f'{METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()
