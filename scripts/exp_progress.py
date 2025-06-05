import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_theme(style="ticks", context="paper")

from adjustText import adjust_text

from pickle import load, dump
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from math import log
from collections import Counter, deque

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

import os, sys
sys.path.append("../")
from os.path import join
from tqdm import tqdm
from IPython.display import display

import search_state
from main import compile_fn

src = "juwels_booster:/p/scratch/hai_1006/results"
dst = "/localdisk/home/lericsso/code/juwels_results/"
cmd = f"rsync -ah --info=progress2 {src} {dst}"

import subprocess

p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(line)
retval = p.wait()

seeds = [0, 3, 4]
crossovers = ["None", "one_point", "constrained_smith_waterman"]
mutations = ["random"]

root_path = os.path.join(dst, "results/einspace/")


print("Loading results files...")
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


it_d = {}
for seed in seeds:
    it_d[seed] = {}
    for algorithm in [
            "Steady-State crossover=None(p=0.0) mutation=random(p=1.0)",
            "Steady-State crossover=one_point(p=1.0) mutation=random(p=0.5)",
            "Steady-State crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
            "Generational crossover=None(p=0.0) mutation=random(p=1.0)",
            "Generational crossover=one_point(p=1.0) mutation=random(p=0.5)",
            "Generational crossover=constrained_smith_waterman(p=1.0) mutation=random(p=0.5)",
    ]:
        it_d[seed][algorithm] = {}
        for dataset in [
            "cifar10", "addnist", "language", "multnist", "cifartile",
            "gutenberg", "isabella", "geoclassing", "chesseract"
        ]:
            try:
                it_d[seed][algorithm][dataset] = len(d[algorithm][dataset][seed]["iteration"])
            except:
                pass
for seed in seeds:
    it_df = pd.DataFrame().from_dict(it_d[seed])
    it_df = it_df.T
    display(it_df)
    it_df.to_csv(f"progress_seed_{seed}.csv")
