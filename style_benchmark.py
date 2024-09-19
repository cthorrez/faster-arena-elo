import os
import pickle
import time
import math
import numpy as np
import polars as pl
from benchmark import load_data
from original_style import construct_style_matrices, fit_mle_elo, get_bootstrap_result_style_control
from faster_style import construct_style_matrices as construct_style_matrices_fast, compute_style_mle, get_bootstrap_style_mle

def main():
    N = 2_000_000
    # df = load_data(use_polars=False, N=N)
    df = pl.scan_parquet("processed_data.parquet").tail(N).collect().to_pandas()

    if os.path.exists(f'style_cache.pkl'):
        print('loading original results from cache')
        X, y, models, original_ratings, original_params = pickle.load(open('style_cache.pkl', 'rb'))
    else:
        start_time = time.time()
        X, y, models = construct_style_matrices(df)
        mid_time = time.time()
        # original_ratings, original_params = fit_mle_elo(X, y, models)
        original_ratings, original_params = get_bootstrap_result_style_control(X, y, df, models, fit_mle_elo, num_round=100)
        end_time = time.time()
        pickle.dump((X, y, models, original_ratings, original_params), open('style_cache.pkl', 'wb'))
        print(original_ratings)
        print(original_params)
        print(f'original preprocess: {mid_time - start_time}')
        print(f'original boot fit: {end_time - start_time}')

    # start_time = time.time()
    # matchups, features, outcomes, models = construct_style_matrices_fast(df)
    # mid_time = time.time()
    # ratings, params = fit_contextual_bt(
    #     matchups,
    #     features,
    #     outcomes,
    #     models=models,
    #     alpha=math.log(10.0),
    #     reg=0.5,
    #     regularize_ratings=True,
    #     init_rating=1000.0,
    #     scale=400.0,
    #     tol=1e-6,
    # )
    # end_time = time.time()
    # print(ratings)
    # print(params)
    # print(f'faster preprocess: {mid_time - start_time}')
    # print(f'faster fit: {end_time - start_time}')

    # print(f'ratings mean abs diff: {np.mean(np.abs(original_ratings.values - ratings))}')
    # print(f'params mean abs diff: {np.mean(np.abs(original_params - params))}')

    # start_time = time.time()
    # ratings, params = get_bootstrap_style_mle(df, num_round=100)
    # end_time = time.time()
    # print(ratings)
    # print(params)
    # print(f'bootstrap duration: {end_time - start_time}')



if __name__ == '__main__':
    main()