import os
import math
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas  as pd
from original_style import STYLE_CONTROL_ELEMENTS_V1
from faster import bt_loss_and_grad, scale_and_offset


DIFF_MASK = np.array([1.0, -1.0], dtype=np.float64)
def contextual_bt_loss_and_grad(
        params,
        n_competitors,
        matchups,
        features,
        outcomes,
        alpha=1.0,
        reg=1.0,
        half_reg=0.5,
        regularize_ratings=False,
    ):
    # Split params into ratings and feature parameters
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:,0] - matchup_ratings[:,1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
    reg_loss = half_reg * np.inner(feature_params, feature_params)
    reg_loss += regularize_ratings * half_reg * np.inner(ratings, ratings)

    loss += reg_loss
    error = (outcomes - probs)
    grad = np.zeros_like(params)
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors],
        matchups[:, [0, 1]],
        matchups_grads[:, None] * DIFF_MASK
    )
    grad[:n_competitors] += regularize_ratings * reg * ratings
    grad[n_competitors:] = -np.dot(features.T, error) + (reg * feature_params)
    return loss, grad


def construct_style_matrices(
    df,
    apply_ratio=[1, 1, 1, 1],
    style_elements=STYLE_CONTROL_ELEMENTS_V1,
    add_one=True,
):
    models = np.unique(df[['model_a', 'model_b']].values)
    model_to_idx = {model:idx for idx,model in enumerate(models)}

    # set two model cols by mapping the model names to their int ids
    matchups = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values

    n = matchups.shape[0]
    k = int(len(style_elements) / 2)

    def style_fn(x, element):
        val = x[element]
        if isinstance(val, int):
            return val
        else:
            return sum(x[element].values())

    style_vector = np.zeros(shape=(2*k,n), dtype=np.int32)
    for idx, element in enumerate(style_elements):
        style_vector[idx,:] = df.conv_metadata.map(partial(style_fn, element=element)).values

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    # Apply ratio where necessary (length, etc)
    style_diff[apply_ratio] /= style_sum[apply_ratio]

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    outcomes = np.full(shape=(n,), fill_value=0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0

    return matchups, features, outcomes, models

def fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models,
        idxs=None,
        alpha=1.0,
        reg=0.5,
        regularize_ratings=True,
        init_rating=1000.0,
        scale=400.0, 
        tol=1e-6
    ):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]
    
    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg, regularize_ratings),
        jac=True,
        method='L-BFGS-B',
        options={'disp': False, 'maxiter': 100, 'gtol': tol},
    )
    
    ratings = result["x"][:n_models]
    feature_params = result["x"][n_models:]
    scaled_ratings = scale_and_offset(
        ratings,
        models=models,
        scale=scale,
        init_rating=init_rating,
    )
    scaled_ratings = pd.Series(scaled_ratings, index=models).sort_values(ascending=False)
    return scaled_ratings, feature_params



def get_bootstrap_result_style_control(
    battles, num_round=1000
):
    matchups, features, outcomes, models = construct_style_matrices(battles)

    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=np.log(10.0),
        tol=1e-6)

    boot_idxs = np.random.randint(
        low=0, high=matchups.shape[0], size=(num_round, matchups.shape[0])
    )

    # with mp.Pool(os.cpu_count()) as pool:
    with mp.Pool(8) as pool:
        results = pool.map(contextual_bt_fn, boot_idxs)

    print(results)


    # df = pd.DataFrame(elos)
    # return df[df.median().sort_values(ascending=False).index], coefs