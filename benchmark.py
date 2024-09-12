import argparse
import os
import io
import time
import numpy as np
import polars as pl
import pandas as pd
from functools import partial
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo as og_mle
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle

def load_data(use_polars=False, N=2_000_000_000):
    if not use_polars:
        # pandas is slower but I want to make sure it works exactly like chatbot arena notebook
        df = pd.read_json('local_file_name.json').sort_values(ascending=True, by=["tstamp"])
        df = df[df["anony"] == True]
        df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]
        df = df.tail(N)
        df = df.reset_index(drop=True)
        # weird as hell but it makes the MLE stuff faster downstream...
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        df = pd.read_parquet(buffer)
    else:
        # use polars for a quick inner loop
        df = pl.read_json('local_file_name.json').filter(
           pl.col("anony") & (pl.col("dedup_tag").struct.field("sampled").fill_null(False))
        ).sort("tstamp").select("model_a", "model_b", "winner").tail(N).to_pandas()
    print(f"num matches: {len(df)}")
    return df


def bench_function(df, refresh, function, **kwargs):
    if function == 'mle':
        og_function = og_mle
        fast_function = fast_mle
    elif function == 'bootstrap':
        og_function = partial(og_bootstrap, func_compute_elo=og_mle)
        fast_function = fast_bootstrap

    os.makedirs('ratings', exist_ok=True)
    og_ratings_path = f'ratings/orignal_{function}_ratings.npz'
    if (not os.path.exists(og_ratings_path)) or refresh:
        start_time = time.time()
        og_ratings = og_function(df, **kwargs)
        duration = time.time() - start_time
        print(f'og {function} fit time: {duration}')
        og_ratings = og_ratings.values
        np.savez(open(og_ratings_path, 'wb'), og_ratings)
    else:
        og_ratings = np.load(open(og_ratings_path, 'rb'))['arr_0']

    start_time = time.time()
    fast_ratings = fast_function(df, **kwargs)
    duration = time.time() - start_time
    print(f'fast {function} fit time: {duration}')

    og_ratings_comp = og_ratings
    fast_ratings_comp = fast_ratings.values
    if function == 'bootstrap':
        og_ratings_comp = og_ratings_comp.mean(axis=0)
        fast_ratings_comp = fast_ratings_comp.mean(axis=0)

    diff = np.abs(og_ratings_comp - fast_ratings_comp).mean()
    print(f'mean diff: {diff}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true')
    parser.add_argument('--use_polars', action='store_true')

    args = parser.parse_args()
    df = load_data(use_polars=args.use_polars)
    bench_function(df, refresh=args.refresh, function='mle')
    bench_function(df, refresh=args.refresh, function='bootstrap', num_round=100)