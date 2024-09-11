import argparse
import os
import time
import numpy as np
import polars as pl
import pandas as pd
from functools import partial
from original import get_bootstrap_result as og_bootstrap, compute_mle_elo as og_mle
from faster import get_bootstrap_result as fast_bootstrap, compute_mle_elo as fast_mle


def bench_function(refresh, function, **kwargs):
    # N = 50_000
    N = 2_000_000
    # lol
    # df = pd.read_json('local_file_name.json').sort_values(ascending=True, by=["tstamp"])
    # df = df[df["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    # df = df.tail(N)

    df = pl.scan_parquet('data.parquet').filter(
        pl.col("dedup_tag").struct.field("sampled").fill_null(False)
    )
    df = df.tail(N).collect().to_pandas()
    print(f'num matches: {len(df)}')

    if function == 'mle':
        og_function = og_mle
        fast_function = fast_mle
    elif function == 'bootstrap':
        og_function = partial(og_bootstrap, func_compute_elo=og_mle)
        fast_function = fast_bootstrap

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
    print(fast_ratings)

    og_ratings_comp = og_ratings
    fast_ratings_comp = fast_ratings.values
    if function == 'bootstrap':
        og_ratings_comp = og_ratings_comp.mean(axis=0)
        print(og_ratings_comp)
        fast_ratings_comp = fast_ratings_comp.mean(axis=0)
        print(fast_ratings_comp)

    diff = np.abs(og_ratings_comp - fast_ratings_comp).mean()
    print(f'mean diff: {diff}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true')
    args = parser.parse_args()
    # bench_function(refresh=args.refresh, function='mle')
    bench_function(refresh=args.refresh, function='bootstrap', num_round=200)