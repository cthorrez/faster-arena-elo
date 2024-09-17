import math
from tqdm import tqdm
import pandas as pd
import numpy as np


STYLE_CONTROL_ELEMENTS_V1 = [
    "sum_assistant_a_tokens",
    "header_count_a",
    "list_count_a",
    "bold_count_a",
    "sum_assistant_b_tokens",
    "header_count_b",
    "list_count_b",
    "bold_count_b",
]

def construct_style_matrices(
    df,
    BASE=10,
    apply_ratio=[1, 1, 1, 1],
    style_elements=STYLE_CONTROL_ELEMENTS_V1,
    add_one=True,
):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]
    assert len(style_elements) % 2 == 0
    k = int(len(style_elements) / 2)

    X = np.zeros([n, p + k])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # creates turn each of the specified column in "conv_metadata" into a vector
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

    X[:, -k:] = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    return X, Y, models


def fit_mle_elo(X, Y, models, indices=None, SCALE=400, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression

    p = len(models.index)

    lr = LogisticRegression(fit_intercept=False)
    if indices:
        lr.fit(X[indices], Y[indices])
    else:
        lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-13b to 800 if applicable
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return (
        pd.Series(elo_scores[:p], index=models.index).sort_values(ascending=False),
        lr.coef_[0][p:],
    )

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