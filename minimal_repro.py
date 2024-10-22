import io
from contextlib import contextmanager
import time
import numpy as np
import pandas as pd
from data_utils import load_data
from original import get_bootstrap_result, compute_mle_elo
from rating_systems import compute_bt, compute_bootstrap_bt


@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name} duration (s): {end_time - start_time:.4f}")



def bench_bt(df):
    with timer('original bt'):
        original_ratings = compute_mle_elo(df)
    with timer('new bt'):
        new_ratings = compute_bt(df)
    diffs = [original_ratings[m] - new_ratings[m] for m in original_ratings.keys()]
    print(f"mean abs diff: {np.mean(np.abs(diffs))}")



def bench_bootstrap_bt(df, num_round, save_figs=False):
    with timer('original bootstrap bt'):
        original_ratings = get_bootstrap_result(df, compute_mle_elo, num_round)
    with timer('new bootstrap bt'):
        new_ratings = compute_bootstrap_bt(df, num_round=num_round)
    
    original_means = original_ratings.values.mean(axis=0)
    new_means = new_ratings[original_ratings.columns].values.mean(axis=0)
    print(f'mean abs diff in bootstrap means: {np.mean(np.abs(original_means - new_means))}')

    original_stds = original_ratings.values.std(axis=0)
    new_stds = new_ratings[original_ratings.columns].values.std(axis=0)
    print(f'mean abs diff in bootstrap stds: {np.mean(np.abs(original_stds - new_stds))}')




def main():
    df = pd.read_json('data/clean_battle_20240826_public.json').sort_values(ascending=True, by=["tstamp"])
    df = df[df["anony"] == True]
    df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]

    # df = load_data(use_preprocessed=True)
    
    num_round = 100

    with timer('original bt'):
        original_ratings = compute_mle_elo(df)

    with timer('original bootstrap bt'):
        original_boot_ratings = get_bootstrap_result(df, compute_mle_elo, num_round)

    # route data through parquet
    buffer = io.BytesIO()
    df = df.reset_index(drop=True)
    df.to_parquet(buffer)
    df = pd.read_parquet(buffer)

    with timer('new bt'):
        new_ratings = compute_bt(df)

    with timer('new bootstrap bt'):
        new_boot_ratings = compute_bootstrap_bt(df, num_round=num_round)

    diffs = [original_ratings[m] - new_ratings[m] for m in original_ratings.keys()]
    print(f"mean abs diff in bt ratings: {np.mean(np.abs(diffs))}")

    original_boot_means = original_boot_ratings.values.mean(axis=0)
    new_boot_means = new_boot_ratings[original_boot_ratings.columns].values.mean(axis=0)
    print(f'mean abs diff in bootstrap means: {np.mean(np.abs(original_boot_means - new_boot_means))}')

    original_boot_stds = original_boot_ratings.values.std(axis=0)
    new_boot_stds = new_boot_ratings[original_boot_ratings.columns].values.std(axis=0)
    print(f'mean abs diff in bootstrap stds: {np.mean(np.abs(original_boot_stds - new_boot_stds))}')

    

if __name__ == '__main__':
    main()