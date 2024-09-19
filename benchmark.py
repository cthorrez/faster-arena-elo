import argparse
import os
import io
import time
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo as og_mle
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle



def bench_function(df, refresh, function, **kwargs):
    if function == 'mle':
        og_function = og_mle
        fast_function = fast_mle
    elif function == 'bootstrap':
        og_function = partial(og_bootstrap, func_compute_elo=og_mle)
        fast_function = fast_bootstrap

    os.makedirs('ratings', exist_ok=True)
    og_ratings_path = f'ratings/orignal_{function}_ratings.csv'
    if (not os.path.exists(og_ratings_path)) or refresh:
        start_time = time.time()
        og_ratings = og_function(df, **kwargs)
        duration = time.time() - start_time
        print(f'og {function} fit time: {duration}')
        og_ratings.to_csv(og_ratings_path, index=False)
    else:
        og_ratings = pd.read_csv(og_ratings_path)

    start_time = time.time()
    fast_ratings = fast_function(df, **kwargs)
    duration = time.time() - start_time
    print(f'fast {function} fit time: {duration}')

    og_ratings_comp = og_ratings.values.squeeze()
    fast_ratings_comp = fast_ratings.values
    if function == 'bootstrap':
        og_ratings_comp = og_ratings_comp.mean(axis=0)
        fast_ratings_comp = fast_ratings_comp.mean(axis=0)
        compare_bootstrap_distributions(og_ratings, fast_ratings, 4)
    else:
        print(fast_ratings)

    diff = np.abs(og_ratings_comp - fast_ratings_comp).mean()
    print(f'mean diff: {diff}')



def compare_bootstrap_distributions(df1, df2, n):
    """
    Visualize the distribution of ratings for the first `n` models across two 
    dataframes with the same schema using KDE plots.

    Parameters:
    df1 (pd.DataFrame): First dataframe of model ratings from bootstrap samples.
    df2 (pd.DataFrame): Second dataframe of model ratings from bootstrap samples.
    n (int): Number of models to visualize (first `n` columns).
    """
    models = df1.columns[:n]
    
    plt.figure(figsize=(12, n * 4))
    
    for i, model in enumerate(models):
        plt.subplot(n, 1, i + 1)
        plt.hist(df1[model], bins=10, alpha=0.5, label='Set 1')
        plt.hist(df2[model], bins=10, alpha=0.5, label='Set 2')
        plt.title(f'Distribution of {model}')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true')
    parser.add_argument('--use_polars', action='store_true')
    parser.add_argument('-n', '--num_rows', type=int, default=2_000_000)
    parser.add_argument('-s', '--num_samples', type=int, default=100)
    args = parser.parse_args()
    df = load_data(use_polars=args.use_polars, N=args.num_rows)
    bench_function(df, refresh=args.refresh, function='mle')
    # bench_function(df, refresh=args.refresh, function='bootstrap', num_round=args.num_samples)