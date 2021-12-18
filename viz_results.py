# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

from helper.constants import NUMBER_OF_INSTANCES, SEQUENCE_LENGTH
# %%
METRIC = "damerau_levenshtein"
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = pd.read_csv('results/Transformer_by_instance.csv').set_index("trace")
lstm_results = pd.read_csv('results/LSTM_by_instance.csv').set_index("trace")

t_res = transformer_results.groupby("true_seq_len").agg(["mean", "std", "median"])[METRIC]
l_res = lstm_results.groupby("true_seq_len").agg(["mean", "std", "median"])[METRIC]
# t_res = t_res.dropna()
# l_res = l_res.dropna()
t_res = t_res.fillna(0.001)
l_res = l_res.fillna(0.001)

t_res["min"] = t_res["mean"]-(2*t_res["std"])
t_res["max"] = t_res["mean"]+(2*t_res["std"])
l_res["min"] = l_res["mean"]-(2*l_res["std"])
l_res["max"] = l_res["mean"]+(2*l_res["std"])

ax.plot(t_res.index, t_res["mean"], label="Transformer", color="blue")
ax.plot(l_res.index, l_res["mean"], label="LSTM", color="red")
ax.fill_between(t_res.index, t_res["min"], t_res["max"], color="blue", alpha=0.1)
ax.fill_between(l_res.index, l_res["min"], l_res["max"], color="red", alpha=0.1)
# ax.errorbar(t_res.index, t_res["mean"], yerr = 2 * t_res["std"], label="Transformer", errorevery=2)
# ax.errorbar(l_res.index, l_res["mean"], yerr = 2 * l_res["std"], label="LSTM", errorevery=2)
ax.legend()
ax.set_title(f'Mean {METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_ylim(0, 1.2)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()
# %%

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
