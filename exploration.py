#!/usr/bin/env python3
# Plot tables and figures around datasets and crossover strategies' exploration capabilities

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
import pandas as pd
import pydot
import seaborn as sns
import search_state

from adjustText import adjust_text
from collections import Counter, deque
from main import compile_fn
from math import log
from networkx.drawing.nx_pydot import graphviz_layout
from pathlib import Path
from pickle import load, dump
from tqdm import tqdm

# -----------------------------------------------------------------------------------------------------------

pd.options.mode.chained_assignment = None
sns.set_theme(style="ticks", context="paper")

# -----------------------------------------------------------------------------------------------------------

seeds = [0, 3, 4]
crossovers = ["None", "one_point", "constrained_smith_waterman"]
mutations = ["random"]

root_path = "results/einspace/"

# -----------------------------------------------------------------------------------------------------------

d = {}
for seed in seeds:
    d[seed] = {}
    for generational in [False, True]:
        for regularised in [False, True]:
            for crossover_rate in ["0.0", "0.5", "1.0"]:
                for crossover in crossovers:
                    for mutation_rate in ["0.0", "0.5", "1.0"]:
                        for mutation in mutations:
                            gen_key = "Generational" if generational else "Steady-State"
                            gen_key += f" crossover={crossover}(p={crossover_rate})"
                            gen_key += f" mutation={mutation}(p={mutation_rate})"
                            d[seed][gen_key] = {}
                            for dataset in [
                                "cifar10", "addnist", "language", "multnist", "cifartile",
                                "gutenberg", "isabella", "geoclassing", "chesseract"
                            ]:
                                # seed_results = []
                                csv_path = (
                                    root_path +
                                    f"{dataset}/evolution/seed={seed}/" +
                                    f"backtrack=True/mode=iterative/" +
                                    f"time_limit=300/max_id_limit=10000/" +
                                    f"depth_limit=20/mem_limit=8192/" +
                                    f"load_from=None/generational={generational}/" +
                                    f"regularised={regularised}/" +
                                    f"population_size=100/architecture_seed=None/" +
                                    f"mutation_strategy={mutation}/mutation_rate={mutation_rate}/" +
                                    f"crossover_strategy={crossover}/crossover_rate={crossover_rate}/" +
                                    f"selection_strategy=tournament/tournament_size=10/" +
                                    f"elitism={5 if generational else 0}/best_architecture.csv"
                                )
                                # print(csv_path, os.path.exists(csv_path))
                                if os.path.exists(csv_path):
                                    c = open(csv_path, "rb")
                                    df = pd.read_csv(c)
                                    # print(df)
                                    # seed_results.append(df["Accuracy"][0])
                                    d[seed][gen_key][dataset] = df["Accuracy"][0]
df = {seed: pd.DataFrame().from_dict(d[seed]) for seed in seeds}
# add avg row
for seed in seeds:
    print(df[seed])
    for key in df[seed].columns:
        d[seed][key].update({"avg": df[seed][key].mean(skipna=True)})
df = {seed: pd.DataFrame().from_dict(d[seed]) for seed in seeds}
# drop nan columns
for seed in seeds:
    df[seed] = df[seed].dropna(axis=1, how='all')
    df[seed] = df[seed].reindex([
        "cifar10", "addnist", "language", "multnist", "cifartile",
        "gutenberg", "isabella", "geoclassing", "chesseract", "avg"
    ])

# -----------------------------------------------------------------------------------------------------------

for seed in seeds:
    _df = df[seed][[
            "Steady-State crossover=None(p=0.0) mutation=random(p=1.0)",
            "Steady-State crossover=one_point(p=1.0) mutation=random(p=0.5)",
            "Steady-State crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
        #     "Generational crossover=None(p=0.0) mutation=random(p=1.0)",
        #     "Generational crossover=one_point(p=1.0) mutation=random(p=0.5)",
        #     "Generational crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    ]] * 100.
    _df = _df.rename({
        "Steady-State crossover=None(p=0.0) mutation=random(p=1.0)": "No Crossover",
        "Steady-State crossover=one_point(p=1.0) mutation=random(p=0.5)": "Subtree Crossover",
        "Steady-State crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)": "CSWX",
    }, axis=1)
    _df = _df.fillna('-')
    latex = _df.round(2).T.to_latex(float_format="%.2f", )
    print(latex)

# -----------------------------------------------------------------------------------------------------------

d = {}
for generational in [False, True]:
    for regularised in [False, True]:
        for crossover_rate in ["0.0", "0.5", "1.0"]:
            for crossover in crossovers:
                for mutation_rate in ["0.0", "0.5", "1.0"]:
                    for mutation in mutations:
                        gen_key = "Generational" if generational else "Steady-State"
                        gen_key += f" crossover={crossover}(p={crossover_rate})"
                        gen_key += f" mutation={mutation}(p={mutation_rate})"
                        d[gen_key] = {}
                        for dataset in [
                            "cifar10", "addnist", "language", "multnist", "cifartile",
                            "gutenberg", "isabella", "geoclassing", "chesseract"
                        ]:
                            d[gen_key][dataset] = {}
                            for seed in seeds:
                                pkl_path = (
                                    root_path +
                                    f"{dataset}/evolution/seed={seed}/" +
                                    f"backtrack=True/mode=iterative/" +
                                    f"time_limit=300/max_id_limit=10000/" +
                                    f"depth_limit=20/mem_limit=8192/" +
                                    f"load_from=None/generational={generational}/" +
                                    f"regularised={regularised}/" +
                                    f"population_size=100/architecture_seed=None/" +
                                    f"mutation_strategy={mutation}/mutation_rate={mutation_rate}/" +
                                    f"crossover_strategy={crossover}/crossover_rate={crossover_rate}/" +
                                    f"selection_strategy=tournament/tournament_size=10/" +
                                    f"elitism={5 if generational else 0}/search_results.pkl"
                                )
                                # print(csv_path, os.path.exists(csv_path))
                                if os.path.exists(pkl_path):
                                    c = open(pkl_path, "rb")
                                    ckpt = load(c)
                                    data = []
                                    for idx, row in enumerate(ckpt["rewards"]):
                                        arch, accuracy = row[0], row[1]
                                        data.append({
                                            "iteration": idx, "arch": arch, "accuracy": accuracy
                                        })
                                    df = pd.DataFrame(data)
                                    # print(df)
                                    d[gen_key][dataset][seed] = df
                                    print(dataset, seed, gen_key, len(data))

# -----------------------------------------------------------------------------------------------------------

import math

from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl

# Adhere to conference style
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Libertine']
mpl.rcParams['text.latex.preamble'] = (
    r'\usepackage[nofontspec,semibold,lining]{libertine}'
    r'\usepackage[T1]{fontenc}'
    r'\usepackage[varqu,varl,scaled=0.96]{zi4}'
    r'\usepackage[libertine,vvarbb,upint]{newtxmath}'
    r'\usepackage[cal=cm,bb=ams,scr=boondoxo]{mathalpha}'
)
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
# mpl.rcParams.update({'font.size': 12})
plt.style.use('seaborn-v0_8-paper')

def plot_max_accuracy(seed, datasets, algorithms, d, title, iter_max=1000, rows=2, cols=4):
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(7, 3.5), sharex=True, sharey=False)
    axes = axes.flatten()

    # Get the default color palette
    colors = sns.color_palette()

    for i, dataset in enumerate(datasets):
        axes[i].set_title(f"{dataset}")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Max Accuracy")

        # Create dataframes
        legend_names = []
        lines = []
        for j, algorithm in enumerate(algorithms):
            if algorithm in d and dataset in d[algorithm]:
                legend_names.append(algorithm)

                D = d[algorithm][dataset][seed].loc[:iter_max,:]

                # Compute cumulative maximum accuracies
                D["max_accuracy"] = D["accuracy"].cummax()

                # Plot
                line = sns.lineplot(x=D["iteration"], y=D["max_accuracy"], label=algorithm.replace(" ", "\n"), ax=axes[i], color=colors[legend_names.index(algorithm)])
                lines.append(line)

                sns.despine()
                axes[i].grid(alpha=0.3)
                axes[i].legend().set_visible(False)
                axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) ## Makes X-axis label with two decimal points
                # only show xlabels for bottom row and ylabels for left column
                if i < cols * (rows - 1):
                    axes[i].set_xlabel("")
                if i % cols != 0:
                    axes[i].set_ylabel("")

    # Manually set the handles and labels for the legend
    handles = [line.lines[0] for line in lines]
    labels = [algorithm.replace(" ", "\n").replace("one_point", "subtree").replace("constrained_smith_waterman", "CSWX") for algorithm in legend_names]

    # Create a single legend
    ncol = len(legend_names) if len(legend_names) <= 3 else int(math.sqrt(len(legend_names)))
    leg = fig.legend(handles, labels, loc='lower center', ncol=ncol)
    for i in range(len(legend_names)):
        leg.legend_handles[i].set_color(colors[i])

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    # plt.suptitle(title)
    # Adjust layout
    plt.tight_layout(rect=[0, 0.14 * (len(legend_names) // ncol), 1, 1])
    plt_path = f"{title.lower().replace(' ', '-')}.pdf"
    print(f"Saving to {plt_path}")
    plt.savefig(plt_path)
    plt.close()

# -----------------------------------------------------------------------------------------------------------

import math

from matplotlib.ticker import FormatStrFormatter


def plot_max_accuracy_mean_sem(datasets, algorithms, d, title, iter_max=1000, rows=2, cols=4):
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(7, 3.3), sharex=True, sharey=False)
    axes = axes.flatten()

    # Get the default color palette
    colors = sns.color_palette()

    for i, dataset in enumerate(datasets):
        axes[i].set_title(f"{dataset}")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Max Accuracy")

        # Create dataframes
        legend_names = []
        lines = []
        for j, algorithm in enumerate(algorithms):
            if algorithm in d and dataset in d[algorithm]:
                legend_names.append(algorithm)

                y = np.zeros((len(d[algorithm][dataset]), iter_max))
                for k, seed in enumerate(d[algorithm][dataset]):
                    D = d[algorithm][dataset][seed].loc[:iter_max,:]
                    # Compute cumulative maximum accuracies
                    cummax = D["accuracy"].cummax().tolist()
                    y[k] = np.concatenate([cummax, np.ones(iter_max - len(cummax)) * cummax[-1]])

                # Plot
                line = sns.lineplot(
                    x=np.arange(len(y[0])), y=y.mean(0),
                    label=algorithm.replace(" ", "\n"), ax=axes[i],
                    color=colors[legend_names.index(algorithm)]
                )
                error = y.std(0, ddof=1) / np.sqrt(len(d[algorithm][dataset]))
                lower = y.mean(0) - error
                upper = y.mean(0) + error
                axes[i].fill_between(np.arange(len(y[0])), lower, upper, alpha=0.2)
                lines.append(line)

                sns.despine()
                axes[i].grid(alpha=0.3)
                axes[i].legend().set_visible(False)
                axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) ## Makes X-axis label with two decimal points
                # only show xlabels for bottom row and ylabels for left column
                if i < cols * (rows - 1):
                    axes[i].set_xlabel("")
                if i % cols != 0:
                    axes[i].set_ylabel("")

    # Manually set the handles and labels for the legend
    handles = [line.lines[0] for line in lines]
    labels = ["No Crossover", "Subtree Crossover", "CSWX"]

    # Create a single legend
    ncol = len(legend_names) if len(legend_names) <= 3 else int(math.sqrt(len(legend_names)))
    leg = fig.legend(handles, labels, loc='lower center', ncol=ncol)
    for i in range(len(legend_names)):
        leg.legend_handles[i].set_color(colors[i])

    # plt.suptitle(title)
    # Adjust layout
    plt.tight_layout(rect=[0, 0.06 * (len(legend_names) // ncol), 1, 1])
    plt_path = f"{title.lower().replace(' ', '-')}.pdf"
    print(f"Saving to {plt_path}")
    plt.savefig(plt_path)
    plt.close()

# -----------------------------------------------------------------------------------------------------------

# Define the datasets and algorithms
datasets = [
    "addnist", "language", "multnist", "cifartile",
    "gutenberg", "isabella", "geoclassing", "chesseract"
]
algorithms = [
    # "Steady-State crossover=None(p=0.0) mutation=random(p=0.5)",
    "Steady-State crossover=None(p=0.0) mutation=random(p=1.0)",
    "Steady-State crossover=one_point(p=1.0) mutation=random(p=0.5)",
    "Steady-State crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    # "Steady-State crossover=shortest_edit_path(p=1.0) mutation=random(p=0.5)",
    # "Generational crossover=None(p=0.0) mutation=random(p=0.5)",
    # "Generational crossover=None(p=0.0) mutation=random(p=1.0)",
    # "Generational crossover=one_point(p=1.0) mutation=random(p=0.5)",
    # "Generational crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    # "Generational crossover=shortest_edit_path(p=1.0) mutation=random(p=0.5)",
]

plot_max_accuracy_mean_sem(datasets, algorithms, d,
                f"Comparing Steady-State Evolution Across Subtree vs. CSWX vs. No Crossover. Seed {seed}. Mean and SEM")
for seed in seeds:
    # Plot the maximum accuracy for each dataset
    plot_max_accuracy(seed, datasets, algorithms, d,
                    f"Comparing Steady-State Evolution Across Subtree vs. CSWX vs. No Crossover. Seed {seed}")

# -----------------------------------------------------------------------------------------------------------

# Define the datasets and algorithms
datasets = [
    "addnist", "language", "multnist", "cifartile",
    "gutenberg", "isabella", "geoclassing", "chesseract"
]
algorithms = [
    # "Steady-State crossover=None(p=0.0) mutation=random(p=0.5)",
    # "Steady-State crossover=None(p=0.0) mutation=random(p=1.0)",
    # "Steady-State crossover=one_point(p=1.0) mutation=random(p=0.5)",
    # "Steady-State crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    # "Steady-State crossover=shortest_edit_path(p=1.0) mutation=random(p=0.5)",
    # "Generational crossover=None(p=0.0) mutation=random(p=0.5)",
    "Generational crossover=None(p=0.0) mutation=random(p=1.0)",
    "Generational crossover=one_point(p=1.0) mutation=random(p=0.5)",
    "Generational crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    # "Generational crossover=shortest_edit_path(p=1.0) mutation=random(p=0.5)",
]

plot_max_accuracy_mean_sem(datasets, algorithms, d,
                f"Comparing Generational Evolution Across Subtree vs. CSWX vs. No Crossover. Seed {seed}. Mean and SEM")

for seed in seeds:
    # Plot the maximum accuracy for each dataset
    plot_max_accuracy(seed, datasets, algorithms, d,
                    f"Comparing Generational Evolution Across Subtree vs. CSWX vs. No Crossover. Seed {seed}")

