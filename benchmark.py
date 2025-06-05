#!/usr/bin/env python3
import click
import copy
import gc
import glob
import humanize
import json
import logging
import os
import pickle
import random
import sys
import time

import numpy as np

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.trees import random_unlabeled_tree
from tqdm import tqdm

## from sepx.similarity import graph_edit_distance
from search_state import DerivationTreeNode
from search_strategies.evolution import Individual
from search_strategies.utils import constrained_smith_waterman_crossover

def count_nodes(node):
    children = getattr(node, 'children', [])
    return 1 + sum(count_nodes(child) for child in children)

class DatabaseNotLoadedError(Exception):
    pass

class InsufficientCandidatesError(Exception):
    pass

# Database class with two load modes.
class DerivationTreeDatabase:
    def __init__(self, pkl_path="data/benchmark/benchmark.pkl", load_mode="cswx1"):
        self.pkl_path = pkl_path
        self.load_mode = load_mode  # "cswx1" or "cswx2"
        self.data = None
        self.groups = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return self
        with open(self.pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.groups = defaultdict(list)
        if self.load_mode == "cswx1":
            for bench_name, bench_dict in self.data.items():
                for seed, run in bench_dict.items():
                    for ind in run['population']:
                        n = count_nodes(ind.root)
                        self.groups[n].append(ind)
        elif self.load_mode == "cswx2":
            for bench_name, bench_dict in self.data.items():
                for seed, run in bench_dict.items():
                    for arch, _, _, _ in run['rewards']:
                        n = count_nodes(arch[0])
                        self.groups[n].append(arch[0])
        self.loaded = True
        return self

    def sample(self, n_nodes, samples=2, rng_seed=42):
        if not self.loaded:
            click.echo("Prefetch DB first!")
            raise DatabaseNotLoadedError("Prefetch DB first")
        if len(self.groups[n_nodes]) < samples:
            raise InsufficientCandidatesError(
                f"Not enough candidates with {n_nodes} nodes; found {len(self.groups[n_nodes])}"
            )
        rng = random.Random(rng_seed)
        return tuple(rng.sample(self.groups[n_nodes], samples))

    def get_sample_counts(self):
        return {n_nodes: len(ind_list) for n_nodes, ind_list in self.groups.items()}

# Global database instance.
DB = None

def run_trial(method, n_nodes, seed, cache_file):
    try:
        if method == "sepx":
            G1 = random_unlabeled_tree(n_nodes, seed=seed)
            G2 = random_unlabeled_tree(n_nodes, seed=seed + 1)
            t0 = time.time()
            # graph_edit_distance(G1, G2)
        elif method in ["cswx1", "cswx2"]:
            ind1, ind2 = DB.sample(n_nodes, samples=2, rng_seed=seed)
            if method == "cswx1":
                ind1, ind2 = ind1.root, ind2.root
            t0 = time.time()
            constrained_smith_waterman_crossover(ind1, ind2)
        else:
            raise ValueError("Unknown method")
        elapsed = time.time() - t0
        result = {"method": method, "n_nodes": n_nodes, "seed": seed, "time": elapsed}
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result
    except DatabaseNotLoadedError as e:
        logging.error(f'[DATABASE] {e}')
    except InsufficientCandidatesError as e:
        logging.error(f'[CANDIDATE] {e}')
    except Exception as e:
        logging.error(f'[GENERIC] {e}')
        raise

@click.group()
def cli():
    pass

@cli.command()
@click.argument("method", type=click.Choice(["sepx", "cswx1", "cswx2", "ged4py"]))
@click.argument("min_nodes", type=int)
@click.argument("max_nodes", type=int)
@click.option("--runs", default=10, help="Number of runs per node count")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle order")
@click.option("-j", "--jobs", default=-1, help="Max number of jobs; -1 means all cores")
def benchmark(method, min_nodes, max_nodes, runs, shuffle, jobs):
    """Run benchmark trials."""
    global DB
    cache_dir = "results/benchmark"
    os.makedirs(cache_dir, exist_ok=True)
    if method in ['cswx1', 'cswx2']:
        click.echo(f"Prefetching database in {method} mode...")
        DB = DerivationTreeDatabase(pkl_path="data/benchmark/benchmark.pkl", load_mode=method)
        DB.load()
        click.echo("Prefetched database")
    max_workers = jobs if jobs > 0 else os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed in range(runs):
            n_range = list(range(min_nodes, max_nodes + 1))
            if shuffle:
                random.shuffle(n_range)
            for n in n_range:
                cache_file = os.path.join(cache_dir, f"{method}_{n}_{seed}.json")
                if os.path.exists(cache_file):
                    click.echo(f"Skipping cached: {cache_file}")
                    continue
                futures.append(executor.submit(run_trial, method, n, seed, cache_file))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking"):
            try:
                _ = future.result()
            except Exception as e:
                if "terminated abruptly" in str(e):
                    click.echo("Warning: A process was terminated abruptly.")
                else:
                    click.echo(f"Error: {e}")
            del future
            gc.collect()
    click.echo("Benchmarking completed")

@cli.command()
def plot():
    """
    Extended plot:
      - SEPX as before (blue line and fill).
      - CSWX (CSWX1 in blue and CSWX2 in orange): 
          * Raw data as scatter ('+' markers).
          * Log-fit extrapolation over the full range.
          * Smoothed lower/upper bounds interpolated from residuals.
      - Bottom histogram & KDE as before.
    """
    import json, glob, pickle, os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import gaussian_kde
    from scipy.interpolate import UnivariateSpline
    import matplotlib as mpl

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
    plt.style.use('seaborn-v0_8-paper')

    # SEPX data remains unchanged.
    files = glob.glob("results/benchmark/*.json")
    sepx_data = defaultdict(list)
    cswx1_data = defaultdict(list)
    cswx2_data = defaultdict(list)
    for f in files:
        with open(f) as fin:
            d = json.load(fin)
        if d["method"] == "sepx":
            sepx_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx1":
            cswx1_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx2":
            cswx1_data[d["n_nodes"]].append(d["time"])

    def compute_geom_stats(times_dict):
        xs, means, lows, highs = [], [], [], []
        for k in sorted(times_dict.keys()):
            vals = np.array(times_dict[k])
            if len(vals) == 0 or np.any(vals <= 0):
                continue
            log_vals = np.log(vals)
            mu, sd = log_vals.mean(), log_vals.std()
            gm = np.exp(mu)
            gstd_factor = np.exp(sd)
            xs.append(k)
            means.append(gm)
            lows.append(gm / gstd_factor)
            highs.append(gm * gstd_factor)
        return np.array(xs, dtype=float), np.array(means), np.array(lows), np.array(highs)

    # Plot SEPX (unchanged)
    x_sepx, mean_sepx, low_sepx, high_sepx = compute_geom_stats(sepx_data)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(6,3.7),
        gridspec_kw={'height_ratios': [4,1]}
    )
    def plot_sepx(ax, data_dict, label, color):
        # Gather raw SEPX points.
        x_raw, y_raw = [], []
        for k, times in data_dict.items():
            x_raw.extend([k] * len(times))
            y_raw.extend(times)
        if not x_raw:
            return
        ax.scatter(x_raw, y_raw, s=7, marker='.', color=color, label='_nolegend')
        x_arr = np.array(x_raw, dtype=float)
        y_arr = np.array(y_raw, dtype=float)
        valid = (x_arr > 0) & (y_arr > 0)
        x_arr, y_arr = x_arr[valid], y_arr[valid]
        if len(x_arr) < 2:
            return
        # Global log–fit.
        log_x, log_y = np.log(x_arr), np.log(y_arr)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        x_fit = np.linspace(x_arr.min(), 200, 100)
        y_fit_log = intercept + slope * np.log(x_fit)
        # Global residual std.
        global_std = np.std(log_y - (intercept + slope * log_x))
        # Sliding window width.
        win = max(1, (x_arr.max() - x_arr.min()) / 20)
        local_std = []
        for x0 in x_fit:
            idx = (x_arr >= x0 - win/2) & (x_arr <= x0 + win/2)
            if np.sum(idx) >= 3:
                std_local = np.std(log_y[idx] - (intercept + slope * np.log(x_arr[idx])))
            else:
                std_local = global_std
            local_std.append(std_local)
        local_std = np.array(local_std)
        lower = np.exp(y_fit_log - local_std)
        upper = np.exp(y_fit_log + local_std)
        ax.fill_between(x_fit, lower, upper, color=color, alpha=0.3)
        ax.plot(x_fit, np.exp(y_fit_log), linestyle='-', color=color, label=label)
    plot_sepx(ax_top, sepx_data, 'SEPX', 'tab:red')

    # --- Revised CSWX plotting: scatter + full extrapolation + smoothed bounds ---
    def plot_cswx(ax, data_dict, label, color):
        # Gather scatter points.
        x_raw, y_raw = [], []
        for k, times in data_dict.items():
            x_raw.extend([k] * len(times))
            y_raw.extend(times)
        if not x_raw:
            return
        ax.scatter(x_raw, y_raw, s=7, marker='.', color=color, label='_nolegend')
        x_arr = np.array(x_raw, dtype=float)
        y_arr = np.array(y_raw, dtype=float)
        valid = (x_arr > 0) & (y_arr > 0)
        x_arr, y_arr = x_arr[valid], y_arr[valid]
        if len(x_arr) < 2:
            return
        # Global log-fit.
        log_x, log_y = np.log(x_arr), np.log(y_arr)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        # Create x grid.
        x_fit = np.linspace(x_arr.min(), 200, 100)
        y_fit_log = intercept + slope * np.log(x_fit)
        # Global residual std.
        global_std = np.std(log_y - (intercept + slope * log_x))
        # Define sliding window width.
        win = max(1, (x_arr.max() - x_arr.min()) / 20)
        local_std = []
        for x0 in x_fit:
            idx = (x_arr >= x0 - win/2) & (x_arr <= x0 + win/2)
            if np.sum(idx) >= 3:
                std_local = np.std(log_y[idx] - (intercept + slope * np.log(x_arr[idx])))
            else:
                std_local = global_std
            local_std.append(std_local)
        local_std = np.array(local_std)
        lower = np.exp(y_fit_log - local_std)
        upper = np.exp(y_fit_log + local_std)
        ax.fill_between(x_fit, lower, upper, color=color, alpha=0.3)
        ax.plot(x_fit, np.exp(y_fit_log), linestyle='-', color=color, label=label)



    plot_cswx(ax_top, cswx1_data, 'CSWX', 'tab:blue')
    plot_cswx(ax_top, cswx2_data, 'CSWX2', 'tab:orange')
    # --- End CSWX modifications ---

    # Bottom histogram & KDE (unchanged)
    hist_file = 'data/benchmark-hist.pkl'
    if os.path.exists(hist_file):
        with open(hist_file, 'rb') as f:
            counts = pickle.load(f)
    else:
        click.echo("Loading DB to create hist file")
        from __main__ import DerivationTreeDatabase
        temp_DB = DerivationTreeDatabase(pkl_path="data/benchmark/benchmark.pkl", load_mode="cswx1")
        temp_DB.load()
        counts = temp_DB.get_sample_counts()
        with open(hist_file, 'wb') as f:
            pickle.dump(counts, f)
    x_all = []
    for node_val, freq_val in counts.items():
        x_all.extend([node_val] * freq_val)

    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"\textbf{Runtime (s)}")
    ax_top.set_title(r"\textbf{Runtime comparison between methods}")
    ax_top.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    ax_top.set_xlim(left=0)
    ax_top.legend(loc='lower right')

    ref_yvals = [1, 60, 3600, 86400, 604800, 2592000, 31536000]
    ref_labels = ["1 second", "1 minute", "1 hour", "1 day", "1 week", "1 month", "1 year"]
    for y, label in zip(ref_yvals, ref_labels):
        ax_top.axhline(y, color='grey', linestyle='--', linewidth=.6)
        ax_top.text(
            1.01, y, label,
            va='center', ha='left', color='grey', fontsize=8,
            transform=ax_top.get_yaxis_transform()
        )

    ax_bot.set_xlabel(r"\textbf{Number of nodes}")
    ax_bot.set_ylabel(r"\textbf{Frequency}")
    ax_bot.set_title(r"\textbf{Distribution of number of nodes}")
    ax_bot.set_xlim(left=0)

    if x_all:
        x_min = 0
        x_max_val = max(x_all)
        bins = np.arange(x_min, x_max_val + 1, 1)
        ax_bot.hist(x_all, bins=bins, alpha=0.3,
                    color='green', edgecolor='none', label='Histogram')
        ax_kde = ax_bot.twinx()
        ax_kde.set_ylabel(r"\textbf{KDE Density}", color='black', fontsize=8)
        if len(set(x_all)) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(x_all)
            kde.covariance_factor = lambda: 0.4 * kde.scotts_factor()
            kde._compute_covariance()
            x_vals = np.linspace(x_min, x_max_val, 500)
            kde_pdf = kde(x_vals)
            ax_kde.plot(x_vals, kde_pdf, color='green', label='KDE')
        lines1, labels1 = ax_bot.get_legend_handles_labels()
        lines2, labels2 = ax_kde.get_legend_handles_labels()
        ax_bot.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    plt.tight_layout()
    plt.savefig("results/benchmark/benchmark-runtimes.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    cli()
