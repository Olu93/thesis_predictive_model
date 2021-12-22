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
def extract_plot_data(METRIC, results):
    t_res = results.groupby("true_seq_len").agg(["mean", "std", "median"])[METRIC]
    t_res = t_res.fillna(0.001)
    t_res["min"] = t_res["mean"]-(2*t_res["std"])
    t_res["max"] = t_res["mean"]+(2*t_res["std"])
    return t_res

def plot_curve(ax, res, color, title):
    ax.plot(res.index, res["mean"], label=title, color=color)
    ax.fill_between(res.index, res["min"], res["max"], color=color, alpha=0.1)
    return ax
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = pd.read_csv('junk/Transformer_by_instance.csv').set_index("trace")
lstm_results = pd.read_csv('junk/LSTM_by_instance.csv').set_index("trace")
transformer_results_bi = pd.read_csv('junk/Transformer_by_instance_bi.csv').set_index("trace")
lstm_results_bi = pd.read_csv('junk/LSTM_by_instance_bi.csv').set_index("trace")

t_res = extract_plot_data(METRIC, transformer_results)
l_res = extract_plot_data(METRIC, lstm_results)
t_res_bi = extract_plot_data(METRIC, transformer_results_bi)
l_res_bi = extract_plot_data(METRIC, lstm_results_bi)

plot_curve(ax, t_res, color="red", title="Transformer Uni")
plot_curve(ax, l_res, color="blue", title="LSTM Uni")
plot_curve(ax, t_res_bi, color="green", title="Transformer Bi")
plot_curve(ax, l_res_bi, color="black", title="LSTM Bi")

ax.legend()
ax.set_title(f'Mean {METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_ylim(0, 1.2)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
transformer_results = pd.read_csv('results/Transformer_by_instance.csv').set_index("trace")
lstm_results = pd.read_csv('results/LSTM_by_instance.csv').set_index("trace")
transformer_results_bi = pd.read_csv('results/Transformer_by_instance_bi.csv').set_index("trace")
lstm_results_bi = pd.read_csv('results/LSTM_by_instance_bi.csv').set_index("trace")

t_res = extract_plot_data(METRIC, transformer_results)
l_res = extract_plot_data(METRIC, lstm_results)
t_res_bi = extract_plot_data(METRIC, transformer_results_bi)
l_res_bi = extract_plot_data(METRIC, lstm_results_bi)

plot_curve(ax, t_res, color="red", title="Transformer Uni")
plot_curve(ax, l_res, color="blue", title="LSTM Uni")
plot_curve(ax, t_res_bi, color="green", title="Transformer Bi")
plot_curve(ax, l_res_bi, color="black", title="LSTM Bi")

ax.legend()
ax.set_title(f'Mean {METRIC} over {SEQUENCE_LENGTH}')
ax.set_ylabel(METRIC)
ax.set_ylim(0, 1.2)
ax.set_xlabel(SEQUENCE_LENGTH)
plt.show()

# %%
