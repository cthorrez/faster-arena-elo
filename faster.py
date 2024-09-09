import math
import tqdm
import time
import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
import pandas as pd
import polars as pl

def preprocess_to_numpy(df):
    models = pd.unique(df[['model_a', 'model_b']].values.ravel())
    N, C = len(df), len(df)
    model_to_idx = {model:idx for idx,model in enumerate(models)}
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.empty((N,3), dtype=np.int32)
    schedule[:,[0,1]] = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values
    # map outcomes to integers
    # model_a win -> 0
    # tie         -> 1
    # model_b win -> 2
    schedule[:,2] = np.select(
        [
            df['winner'] == 'model_a',
            df['winner'] == 'model_b',
        ],
        [0, 2],
        default=1
    )
    matchups, weights = np.unique(schedule, return_counts=True, axis=0)
    weights = weights.astype(np.float32)

    print(matchups, weights)
    print(matchups.shape)

    return matchups, weights, models, model_to_idx


def bt_loss_and_grad(ratings, matchups, weights, alpha=1.0):
    N = weights.sum()
    matchup_ratings = ratings[matchups[:,[0,1]]]
    # this maps 0 -> 1.0, 1 -> 0.5, 2 -> 1.0 
    outcomes = matchups[:,2].astype(np.float32) / 2.0
    probs = sigmoid(alpha * (matchup_ratings[:,0] - matchup_ratings[:,1]))
    loss = -(((np.log(probs) * outcomes) + (np.log(1.0 - probs) * (1.0 - outcomes))) * weights).sum() / N
    matchups_grads = (outcomes - probs) * weights / N
    model_grad = np.zeros_like(ratings)
    np.add.at(model_grad, matchups[:, 0], -matchups_grads)
    np.add.at(model_grad, matchups[:, 1], matchups_grads)
    return loss, model_grad

 
    
def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    start_time = time.time()
    matchups, weights, models, model_to_idx = preprocess_to_numpy(df)
    duration = time.time() - start_time
    print(f'preprocess duration (s): {duration}')
    alpha = math.log(BASE) / SCALE

    start_time = time.time()
    initial_ratings = np.zeros(len(model_to_idx))
    ratings = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, weights),
        jac=True,
    )
    duration = time.time() - start_time
    print(f'fit duration (s): {duration}')
    # print(ratings)
    # sort_idx = np.argsort(ratings['x'])
    # for rank_idx, sort_idx in enumerate(sort_idx):
    #     print(models[sort_idx], ratings['x'][sort_idx])


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]

def main():
    N = 20_000_000
    df = pl.scan_parquet('data.parquet').tail(N).collect().to_pandas()
    print(df[['model_a', 'model_b', 'winner']])
 
    start_time = time.time()
    compute_mle_elo(df)
    duration = time.time() - start_time
    print(f'duration (s): {duration}')

if __name__ == '__main__':
    main()
