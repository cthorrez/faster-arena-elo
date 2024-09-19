import os
import math
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas  as pd
from faster import bt_loss_and_grad, scale_and_offset

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


def preprocess_to_numpy(df):
    models = np.unique(df[['model_a', 'model_b']].values)
    model_to_idx = {model:idx for idx,model in enumerate(models)}
    matchups = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values
    outcomes = np.full(shape=(matchups.shape[0],), fill_value=0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def fit_vectorized_elo(
    matchups,
    outcomes,
    num_models,
    sample_indices=None,
    k=4.0,
    base=10.0,
    scale=400.0,
    ):
    """fit multiple sets of Elo ratings on different samples of the data at the same time"""
    alpha = math.log(base) / scale
    n = matchups.shape[0]
    # if no sample indices are provided, just go over the results in order once
    if sample_indices is None:
        sample_indices = np.arange(n)[:,np.newaxis]
    num_samples = sample_indices.shape[1]
    ratings = np.zeros(shape=(num_samples, num_models), dtype=np.float64)
    # iterate over the rows of sample_indices, each column is an index into a match in the input arrays
    sample_range = np.arange(num_samples)
    for matchup_indices in sample_indices:
        model_a_indices = matchups[matchup_indices,0] # [num_samples,]
        model_b_indices = matchups[matchup_indices,1] # [num_samples,]
        model_a_ratings = ratings[sample_range, model_a_indices] # [num_samples,]
        model_b_ratings = ratings[sample_range, model_b_indices] # [num_samples,]
        sample_outcomes = outcomes[matchup_indices] # [num_samples,]
        probs = expit(alpha * (model_a_ratings - model_b_ratings)) # [num_samples,]
        updates = k * (sample_outcomes - probs) # [num_samples,]
        ratings[sample_range, model_a_indices] += updates
        ratings[sample_range, model_b_indices] -= updates
    return np.squeeze(ratings)

def get_elo_ratings(df, sample_indices=None):
    matchups, outcomes, models = preprocess_to_numpy(df)
    ratings = fit_vectorized_elo(matchups, outcomes, len(models), sample_indices=sample_indices)
    return ratings


def get_bootstrap_elo_ratings(df, num_boot=100):
    sample_indices = np.random.randint(low=0, high=len(df), size=(len(df), num_boot))
    ratings = get_elo_ratings(df, sample_indices)
    return ratings

