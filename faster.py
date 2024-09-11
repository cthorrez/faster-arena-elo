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
    outcomes = matchups_outcomes[:,2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # normalize the weights
    weights = weights / weights.sum()
    return matchups, outcomes, weights, models


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:,0] - matchup_ratings[:,1])
    probs = expit(logits)
    
    # Numerically stable log probability calculation
    # based on grad check it's BAD though!!!
    # log_probs = log_expit(logits)
    # log_1_minus_probs = log_expit(-logits)  # log(1-sigmoid(x)) = log_expit(-x)
    # loss = -((log_probs * outcomes + log_1_minus_probs * (1.0 - outcomes)) * weights).sum()

    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
    
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    np.add.at(model_grad, matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64))
    return loss, model_grad

 
    
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, tol=1e-6):
    matchups, outcomes, weights, models = preprocess_to_numpy(df)
    ratings, loss = fit_bt(matchups, outcomes, weights, len(models), np.log(BASE), tol)
    scaled_ratings = (ratings * SCALE) + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models:
        scaled_ratings += 1114 - scaled_ratings[models.tolist().index("mixtral-8x7b-instruct-v0.1")]
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def fit_bt_wrapper(args):
    return fit_bt(*args)

def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        # method='BFGS',
        method='L-BFGS-B',
        # method='SLSQP',
        options={'disp' : False, 'maxiter': 100, 'gtol':tol},
    )
    loss = result['fun']
    # print(f'loss: {loss}')
    ratings = result['x']
    return ratings, loss


def get_bootstrap_result(battles, num_round, BASE=10.0, SCALE=400.0, INIT_RATING=1000.0):
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

    ratings = np.empty((num_round, n_models))
    losses = np.empty(num_round)
    for idx, result in enumerate(results):
        ratings[idx,:] = result[0] 
        losses[idx] = result[1]

    scaled_ratings = (ratings * SCALE) + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models:
        baseline_idx = models.tolist().index("mixtral-8x7b-instruct-v0.1")
        scaled_ratings = scaled_ratings + (1114 - scaled_ratings[:,baseline_idx][:,None])

    df = pd.DataFrame(scaled_ratings, columns=models)
    print(df.columns)

    return df[df.median().sort_values(ascending=False).index]


def main():
    N = 100_000_000
    df = pl.scan_parquet('data.parquet').tail(N).collect().to_pandas()
 
    # start_time = time.time()
    # compute_mle_elo(df)
    # duration = time.time() - start_time
    # print(f'mle duration (s): {duration}')

    start_time = time.time()
    ratings, losses = get_bootstrap_result(df, num_round=10000)
    duration = time.time() - start_time
    print(f'bootstrap duration (s): {duration}')
    print(ratings.shape)
    print(losses.mean())

if __name__ == '__main__':
    main()
