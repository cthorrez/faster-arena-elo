import os
import pickle
import time
import argparse
import math
import numpy as np
from data_utils import load_data
from original_style import construct_style_matrices, fit_mle_elo, get_bootstrap_result_style_control
from faster_style import construct_style_matrices as construct_style_matrices_fast, compute_style_mle, get_bootstrap_style_mle

def main(num_rows, num_boot, refresh=False):
    df = load_data(num_rows, use_preprocessed=True)
    # df = load_data(num_rows, use_preprocessed=False)
    # df = load_data(num_rows, use_polars=True)


    os.makedirs('cache', exist_ok=True)
    if os.path.exists(f'cache/style.pkl') and not refresh:
        X, y, models, original_ratings, original_params = pickle.load(open('cache/style.pkl', 'rb'))
        print('loaded original results from cache')
    else:
        start_time = time.time()
        X, y, models = construct_style_matrices(df)
        original_ratings, original_params = fit_mle_elo(X, y, models)
        # original_ratings, original_params = get_bootstrap_result_style_control(X, y, df, models, fit_mle_elo, num_round=num_boot)
        end_time = time.time()
        pickle.dump((X, y, models, original_ratings, original_params), open('cache/style.pkl', 'wb'))
        print(original_ratings)
        # print(np.mean(np.array(original_params), axis=0))
        print(f'original boot fit: {end_time - start_time}')

    start_time = time.time()
    ratings, params = compute_style_mle(df, alpha=math.log(10.0))
    end_time = time.time()
    print(ratings)
    print(params)
    print(f'faster fit: {end_time - start_time}')

    print(f'ratings mean abs diff: {np.mean(np.abs(original_ratings.values - ratings))}')
    print(f'params mean abs diff: {np.mean(np.abs(original_params - params))}')

    # start_time = time.time()
    # ratings, params = get_bootstrap_style_mle(df, num_round=num_boot)
    # end_time = time.time()
    # print(ratings)
    # print(params)
    # print(f'faster bootstrap duration: {end_time - start_time}')

    # print(f'abs mean param diff: {np.mean(np.abs(original_params - params))}')
    # print(f'abs mean rating diff: {np.mean(np.abs(original_ratings.values - ratings.values))}')
    # print(f'abs mean rating mean diff: {np.mean(np.abs(original_ratings.values.mean(axis=0) - ratings.values.mean(axis=0)))}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true', default=False)
    parser.add_argument('-b', '--num_boot', type=int, default=100)
    parser.add_argument('-n', '--num_rows', type=int, default=2_000_000)

    args = parser.parse_args()
    main(**vars(args))