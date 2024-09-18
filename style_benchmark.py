import time
import math
import numpy as np
import polars as pl
from benchmark import load_data
from original_style import construct_style_matrices, fit_mle_elo
from faster_style import construct_style_matrices as construct_style_matrices_fast, fit_contextual_bt

def main():
    N = 2_000_000
    # df = load_data(use_polars=True, N=N)
    df = pl.scan_parquet("processed_data.parquet").tail(N).collect().to_pandas()
    # print(df.head(1)['conv_metadata'].to_dict())

    # start_time = time.time()
    # X, y, models = construct_style_matrices(df)
    # mid_time = time.time()
    # original_ratings, original_params = fit_mle_elo(X, y, models)
    # end_time = time.time()
    # print(original_ratings)
    # print(original_params)
    # print(f'original preprocess: {mid_time - start_time}')
    # print(f'original fit: {end_time - start_time}')

    start_time = time.time()
    matchups, features, outcomes, models = construct_style_matrices_fast(df)
    mid_time = time.time()
    ratings, params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=math.log(10.0),
        reg=0.5,
        regularize_ratings=True,
        init_rating=1000.0,
        scale=400.0,
        tol=1e-6,
    )
    end_time = time.time()
    print(ratings)
    print(params)
    print(f'faster preprocess: {mid_time - start_time}')
    print(f'faster fit: {end_time - start_time}')

    # print(f'ratings mean abs diff: {np.mean(np.abs(original_ratings.values - ratings))}')
    # print(f'params mean abs diff: {np.mean(np.abs(original_params - params))}')





if __name__ == '__main__':
    main()