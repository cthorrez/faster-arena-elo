import argparse
import os
from contextlib import contextmanager
import time
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from data_utils import load_data
from original import get_bootstrap_result, compute_mle_elo as og_mle, compute_elo as original_compute_elo
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle
from original_style import construct_style_matrices, fit_mle_elo, get_bootstrap_result_style_control
from faster_style import construct_style_matrices as construct_style_matrices_fast, compute_style_mle, get_bootstrap_style_mle
from rating_systems import (
    compute_elo_ratings,
    compute_bootstrap_elo_ratings
)


@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name} duration (s): {end_time - start_time:.4f}")


def plot_bootstraps(label, models, original_means, original_stds, new_means, new_stds):
    x = np.arange(len(original_means))  # Model indices
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, original_means, yerr=original_stds, fmt='o-', label=f'Original {label}', capsize=5)
    plt.errorbar(x, new_means, yerr=new_stds, fmt='s-', label=f'New {label}', capsize=5)
    plt.xticks(ticks=x, labels=models, rotation=45, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Rating')
    plt.title(f'Bootstrap {label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/bootstrap_{label}')
    plt.show()


def bench_elo(df):
    with timer('original elo'):
        original_ratings = original_compute_elo(df)
    with timer('new elo'):
        new_ratings = compute_elo_ratings(df)
    diffs = [original_ratings[m] - new_ratings[m] for m in original_ratings.keys()]
    print(f"mean abs diff: {np.mean(np.abs(diffs))}")


def bench_bootstrap_elo(df, num_round):
    with timer('original elo'):
        original_ratings = get_bootstrap_result(df, original_compute_elo, num_round, seed=42)
    with timer('new elo'):
        new_ratings = compute_bootstrap_elo_ratings(df, num_round=num_round)
    
    original_means = original_ratings.values.mean(axis=0)
    new_means = new_ratings[original_ratings.columns].values.mean(axis=0)
    print(f'mean abs diff in bootstrap means: {np.mean(np.abs(original_means - new_means))}')

    original_stds = original_ratings.values.std(axis=0)
    new_stds = new_ratings[original_ratings.columns].values.std(axis=0)
    print(f'mean abs diff in bootstrap stds: {np.mean(np.abs(original_stds - new_stds))}')

    plot_bootstraps(
        label='Elo',
        models=original_ratings.columns,
        original_means=original_means,
        original_stds=original_stds,
        new_means=new_means,
        new_stds=new_stds
    )



def main():
    N = 200_000
    df = load_data(N=N, use_preprocessed=True)
    # df = load_data()

    bench_elo(df)
    bench_bootstrap_elo(df, num_round=100)



if __name__ == '__main__':
    main()
    # bench()