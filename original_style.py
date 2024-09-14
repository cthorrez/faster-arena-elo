import math
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