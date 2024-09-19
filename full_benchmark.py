import argparse
import os
import io
import time
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from data_utils import load_data
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo as og_mle
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle
from original_style import construct_style_matrices, fit_mle_elo, get_bootstrap_result_style_control
from faster_style import construct_style_matrices as construct_style_matrices_fast, compute_style_mle, get_bootstrap_style_mle
from rating_systems import get_elo_ratings


FUNCTIONS = {
    'elo': (None, None),
    'boot_elo': (None, None),
    'bt': (og_mle, fast_mle),
    'boot_bt': (og_bootstrap, fast_bootstrap),
    'style_preprocess': (construct_style_matrices, construct_style_matrices_fast),
    'style': (fit_mle_elo, compute_style_mle),
    'boot_style': (get_bootstrap_result_style_control, get_bootstrap_style_mle),
}

def bench():
    pass

def main():
    df = load_data(N=100_000, use_preprocessed=True)
    start_time = time.time()
    ratings = get_elo_ratings(df)
    end_time = time.time()
    print(ratings.shape)
    print(ratings)
    print(f'elo duration (s): {end_time - start_time}')

    start_time = time.time()
    sample_indices = np.random.randint(low=0, high=len(df), size=(len(df), 100))
    ratings = get_elo_ratings(df, sample_indices)
    end_time = time.time()
    print(ratings.shape)
    print(ratings)
    print(f'bootstrap elo duration (s): {end_time - start_time}')


if __name__ == '__main__':
    main()
    # bench()