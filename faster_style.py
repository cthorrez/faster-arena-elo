import math
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas  as pd
from original_style import STYLE_CONTROL_ELEMENTS_V1
from faster import bt_loss_and_grad, scale_and_offset



def contextual_bt_loss_and_grad(
        params,
        n_competitors,
        matchups,
        features,
        outcomes,
        alpha=1.0,
        reg=1.0,
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
    reg_loss = 0.5 * reg * np.inner(feature_params, feature_params)
    reg_loss += regularize_ratings * 0.5 * reg * np.inner(ratings, ratings)

    loss += reg_loss
    error = (outcomes - probs)
    grad = np.zeros_like(params)
    
    matchups_grads = -alpha * error
    np.add.at(grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64))
    
    grad[n_competitors:] = -np.dot(features.T, error) + (reg * feature_params)
    grad[:n_competitors] += regularize_ratings * reg * ratings

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

    # features = np.empty(shape=(n,k))
    style_vector = np.array(
        [
            df.conv_metadata.map(
                lambda x: x[element]
                if type(x[element]) is int
                else sum(x[element].values())
            ).tolist()
            for element in style_elements
        ]
    )
    print(style_vector.shape)

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio = np.flatnonzero(apply_ratio)

    style_diff[apply_ratio] /= style_sum[
        apply_ratio
    ]  # Apply ratio where necessary (length, etc)

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)


    outcomes = np.full(shape=(n,), fill_value=0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0

    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T
    print(f'{unq_x.shape=}')

    return matchups, features, outcomes, models

def fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models,
        alpha,
        reg,
        regularize_ratings,
        init_rating,
        scale=400.0, 
        tol=1e-6
    ):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    
    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, regularize_ratings),
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
    X, Y, battles, models, func_compute_elo, num_round=1000
):
    elos = []
    coefs = []
    assert X.shape[0] % 2 == 0 and X.shape[0] == Y.shape[0]
    k = int(
        X.shape[0] / 2
    )  # Since we duplicate the battles when constructing X and Y, we don't want to sample the duplicates

    battles_tie_idx = (battles["winner"] == "tie") | (
        battles["winner"] == "tie (bothbad)"
    )
    for _ in tqdm(range(num_round), desc="bootstrap"):
        indices = np.random.choice(list(range(k)), size=(k), replace=True)

        index2tie = np.zeros(k, dtype=bool)
        index2tie[battles_tie_idx] = True

        nontie_indices = indices[~index2tie[indices]]
        tie_indices = np.concatenate(
            [indices[index2tie[indices]], indices[index2tie[indices]] + k]
        )

        _X = np.concatenate([X[nontie_indices], X[nontie_indices], X[tie_indices]])
        _Y = np.concatenate([Y[nontie_indices], Y[nontie_indices], Y[tie_indices]])

        assert _X.shape == X.shape and _Y.shape == Y.shape

        states = ~_X[:, : len(models)].any(axis=0)

        elo, coef = func_compute_elo(_X, _Y, models=models[~states])
        elos.append(elo)
        coefs.append(coef)

    df = pd.DataFrame(elos)
    return df[df.median().sort_values(ascending=False).index], coefs