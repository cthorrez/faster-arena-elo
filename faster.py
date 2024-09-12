import os
from functools import partial
import multiprocessing as mp
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas as pd


def preprocess_to_numpy(df):
    models = pd.unique(df[['model_a', 'model_b']].values.ravel()).tolist()
    model_to_idx = {model:idx for idx,model in enumerate(models)}
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.empty((len(df), 3), dtype=np.int32)
    # set the two model cols by mapping the model names to their int ids
    schedule[:,[0,1]] = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values
    # map outcomes to integers (must be same dtype as model ids so it can be in the same array)
    # model_a win -> 2, tie -> 1, model_b win -> 0
    schedule[:,2] = np.select(
        condlist=[df['winner'] == 'model_a', df['winner'] == 'model_b'],
        choicelist=[2, 0],
        default=1
    )
    # count the number of occurances of each observed result
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:,[0,1]]
    # map 2 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used as labels during optimization
    outcomes = matchups_outcomes[:,2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # each possible result is weighted according to number of times it occured in the dataset
    weights = weights / weights.sum()
    return matchups, outcomes, weights, models


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:,0] - matchup_ratings[:,1])
    probs = expit(logits)
    # this form naturally counts a draw as half a win and half a loss
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    # aggregate gradients at the model level using the indices in matchups
    np.add.at(model_grad, matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64))
    return loss, model_grad

def scale_and_offset(ratings, models, scale=400, init_rating=1000, baseline_model="mixtral-8x7b-instruct-v0.1", baseline_rating=1114):
    scaled_ratings = (ratings * scale) + init_rating
    baseline_idx = models.index(baseline_model)
    scaled_ratings += (baseline_rating - scaled_ratings[..., [baseline_idx]])
    return scaled_ratings
    
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, tol=1e-6):
    matchups, outcomes, weights, models = preprocess_to_numpy(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), np.log(BASE), tol)
    scaled_ratings = scale_and_offset(ratings, models, SCALE, INIT_RATING)
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method='L-BFGS-B',
        # method='SLSQP',
        options={'disp' : False, 'maxiter': 100, 'gtol': tol},
    )
    return result["x"]


def get_bootstrap_result(battles, num_round, BASE=10.0, SCALE=400.0, INIT_RATING=1000.0, tol=1e-6):
    matchups, outcomes, weights, models = preprocess_to_numpy(battles)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles),
        pvals=weights,
        size=(num_round)
    )
    # only the distribution over their occurance counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(battles)

    # the only thing different across samples is the distribution of weights
    bt_fn = partial(fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(BASE), tol=tol)

    with mp.Pool(os.cpu_count()) as pool:
        results = pool.map(bt_fn, boot_weights)

    ratings = np.array(results)
    scaled_ratings = scale_and_offset(ratings, models, SCALE, INIT_RATING)
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]
