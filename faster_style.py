import math
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas  as pd
from original_style import STYLE_CONTROL_ELEMENTS_V1
from faster import bt_loss_and_grad



def contextual_bt_loss_and_grad(
        params,
        n_competitors,
        matchups,
        features,
        outcomes,
        weights,
        alpha=1.0,
        reg=1.0,
    ):
    # Split params into ratings and feature parameters
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:,0] - matchup_ratings[:,1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
    reg_loss = 0.5 * reg * np.inner(feature_params, feature_params)
    loss += reg_loss
    print(loss)

    error = (outcomes - probs) * weights
    grad = np.zeros_like(params)
    
    matchups_grads = -alpha * error
    np.add.at(grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64))
    
    grad[n_competitors:] = -np.dot(features.T, error) + reg * feature_params
    
    return loss, grad



def fit_contextual_bt(matchups, features, outcomes, weights, n_competitors, alpha, reg, tol=1e-6):
    n_features = features.shape[1]
    initial_params = np.zeros(n_competitors + n_features, dtype=np.float64)
    
    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_competitors, matchups, features, outcomes, weights, alpha, reg),
        jac=True,
        method='L-BFGS-B',
        options={'disp': False, 'maxiter': 100, 'gtol': tol},
    )
    
    fitted_ratings = result["x"][:n_competitors]
    fitted_feature_params = result["x"][n_competitors:]
    
    return fitted_ratings, fitted_feature_params

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

    return matchups, features, outcomes, models
