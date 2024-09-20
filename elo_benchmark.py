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
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo as og_mle, compute_elo
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle
from original_style import construct_style_matrices, fit_mle_elo, get_bootstrap_result_style_control
from faster_style import construct_style_matrices as construct_style_matrices_fast, compute_style_mle, get_bootstrap_style_mle
from rating_systems import get_elo_ratings, get_bootstrap_elo_ratings


@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name} duration (s): {end_time - start_time:.4f}")

FUNCTIONS = {
    'elo': (None, None),
    'boot_elo': (None, None),
    'bt': (og_mle, fast_mle),
    'boot_bt': (og_bootstrap, fast_bootstrap),
    'style_preprocess': (construct_style_matrices, construct_style_matrices_fast),
    'style': (fit_mle_elo, compute_style_mle),
    'boot_style': (get_bootstrap_result_style_control, get_bootstrap_style_mle),
}

def bench_elo(df):
    with timer('original elo'):
        original_ratings = compute_elo(df)
    with timer('new elo'):
        new_ratings = get_elo_ratings(df)
    diffs = [original_ratings[m] - new_ratings[m] for m in original_ratings.keys()]
    print(f"mean abs diff: {np.mean(np.abs(diffs))}")


def bench_bootstrap_elo(df, num_round):
    with timer('original elo'):
        original_ratings = og_bootstrap(df, compute_elo, num_round, seed=42)
    with timer('new elo'):
        new_ratings = get_bootstrap_elo_ratings(df, num_round=num_round)
    
    original_means = original_ratings.values.mean(axis=0)
    new_means = new_ratings[original_ratings.columns].values.mean(axis=0)
    print(f'mean abs diff in bootstrap means: {np.mean(np.abs(original_means - new_means))}')

    original_stds = original_ratings.values.std(axis=0)
    new_stds = new_ratings[original_ratings.columns].values.std(axis=0)
    print(f'mean abs diff in bootstrap stds: {np.mean(np.abs(original_stds - new_stds))}')

    model_names = original_ratings.columns
    x = np.arange(len(original_means))  # Model indices
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, original_means, yerr=original_stds, fmt='o-', label='Original Elo', capsize=5)
    plt.errorbar(x, new_means, yerr=new_stds, fmt='s-', label='New Elo', capsize=5)
    plt.xticks(ticks=x, labels=model_names, rotation=45, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Elo Rating')
    plt.title('Bootstrap Elo Ratings with Error Bars')
    plt.legend()
    plt.tight_layout()
    plt.show()


def bench_bootstrap_elo_seed(df, num_round):
    ratings_42 = og_bootstrap(df, compute_elo, num_round=num_round, seed=42)
    ratings_0 = og_bootstrap(df, compute_elo, num_round=num_round, seed=0)
    
    original_means = ratings_42.values.mean(axis=0)
    new_means = ratings_0[ratings_42.columns].values.mean(axis=0)
    print(f'mean abs diff in bootstrap means: {np.mean(np.abs(original_means - new_means))}')

    original_stds = ratings_42.values.std(axis=0)
    new_stds = ratings_0[ratings_42.columns].values.std(axis=0)
    print(f'mean abs diff in bootstrap stds: {np.mean(np.abs(original_stds - new_stds))}')

    model_names = ratings_42.columns
    x = np.arange(len(original_means))  # Model indices
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, original_means, yerr=original_stds, fmt='o-', label='Boot Elo Seed 42', capsize=5)
    plt.errorbar(x, new_means, yerr=new_stds, fmt='s-', label='Boot Elo Seed 0', capsize=5)
    plt.xticks(ticks=x, labels=model_names, rotation=45, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Elo Rating')
    plt.title('Bootstrap Elo Ratings with Error Bars')
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    N = 2_000_000
    df = load_data(N=N, use_preprocessed=True)
    # df = load_data()

    bench_elo(df)
    bench_bootstrap_elo(df, num_round=100)



if __name__ == '__main__':
    main()
    # bench()