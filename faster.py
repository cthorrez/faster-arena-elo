import math
import tqdm
import time
import numpy as np
from scipy.special import expit, log_expit
from scipy.optimize import minimize
import multiprocessing as mp
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
    # model_a win -> 2
    # tie         -> 1
    # model_b win -> 0
    schedule[:,2] = np.select(
        [
            df['winner'] == 'model_a',
            df['winner'] == 'model_b',
        ],
        [2, 0],
        default=1
    )
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:,[0,1]]
    # maps 2.0 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used during optimization
    outcomes = matchups_outcomes[:,2].astype(np.float32) / 2.0
    weights = weights.astype(np.float32)
    # normalize the weights
    weights = weights / weights.sum()
    return matchups, outcomes, weights, models


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:,0] - matchup_ratings[:,1])
    
    probs = expit(logits)  # Still needed for gradient calculation
    # Numerically stable log probability calculation
    # log_probs = log_expit(logits)
    # log_1_minus_probs = log_expit(-logits)  # log(1-sigmoid(x)) = log_expit(-x)
    # loss = -((log_probs * outcomes + log_1_minus_probs * (1.0 - outcomes)) * weights).sum()
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights

    model_grad = np.zeros_like(ratings)
    # np.add.at(model_grad, matchups[:, 0], matchups_grads)
    # np.add.at(model_grad, matchups[:, 1], -matchups_grads)
    np.add.at(model_grad, matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1, -1]))
    return loss, model_grad

 
    
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    matchups, outcomes, weights, models = preprocess_to_numpy(df)
    return fit_bt(matchups, outcomes, weights, models, BASE, SCALE, INIT_RATING)


def fit_bt_wrapper(args):
    return fit_bt(*args)

def fit_bt(matchups, outcomes, weights, n_models, alpha):
    initial_ratings = np.zeros(n_models)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        # method='SLSQP',
        method='L-BFGS-B',
        options={'disp' : False},
    )
    loss = result['fun']
    ratings = result['x']
    return ratings, loss


def get_bootstrap_result(battles, num_round, BASE=10.0, SCALE=400.0, INIT_RATING=100.0):
    matchups, outcomes, weights, models = preprocess_to_numpy(battles)
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles),
        pvals=weights,
        size=(num_round)
    )
    print(idxs.shape)
    boot_matchups = np.tile(matchups[None,:], (num_round,1,1))
    boot_outcomes = np.tile(outcomes[None,:], (num_round,1))
    boot_weights = idxs.astype(np.float32) / len(battles)
    print(boot_matchups.shape)
    print(boot_outcomes.shape)

    n_models = len(models)
    alpha = np.log(BASE)
    args_list = [(boot_matchups[i], boot_outcomes[i], boot_weights[i], n_models, alpha) 
                 for i in range(num_round)]

    with mp.Pool(8) as pool:
        results = pool.map(fit_bt_wrapper, args_list)


    ratings = np.stack([result[0] for result in results])
    losses = np.array([result[1] for result in results])

    ratings = np.median(ratings, axis=0)
    scaled_ratings = (ratings * SCALE) + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models:
        scaled_ratings += 1114 - scaled_ratings[models.tolist().index("mixtral-8x7b-instruct-v0.1")]
    sort_idxs = np.argsort(-scaled_ratings)
    for place_idx, sort_idx in enumerate(sort_idxs):
        print(models[sort_idx], scaled_ratings[sort_idx])

    return ratings, losses



def main():
    N = 100_000_000
    df = pl.scan_parquet('data.parquet').tail(N).collect().to_pandas()
 
    # start_time = time.time()
    # compute_mle_elo(df)

    start_time = time.time()
    # matchups, outcomes, weights, models = preprocess_to_numpy(df)
    ratings, losses = get_bootstrap_result(df, num_round=100)
    duration = time.time() - start_time
    print(f'duration (s): {duration}')
    print(ratings.shape)
    print(losses.mean())

if __name__ == '__main__':
    main()
