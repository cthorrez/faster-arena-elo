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


# def preprocess_to_numpy(df):
#     models = np.unique(df[['model_a', 'model_b']].values)
#     model_to_idx = {model:idx for idx,model in enumerate(models)}
#     matchups = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values
#     outcomes = np.full(shape=(matchups.shape[0],), fill_value=0.5)
#     outcomes[df["winner"] == "model_a"] = 1.0
#     outcomes[df["winner"] == "model_b"] = 0.0
#     return matchups, outcomes, models

# def preprocess_to_numpy(df):
#     # Use pd.factorize() for both model columns at once
#     models, matchups = pd.factorize(df[['model_a', 'model_b']].values.ravel())
#     matchups = matchups.reshape(-1, 2)
#     print(matchups)
#     # Pre-allocate outcomes array with correct values
#     outcomes = np.where(df["winner"] == "model_a", 1.0, 
#                         np.where(df["winner"] == "model_b", 0.0, 0.5))
    
#     return matchups, outcomes, pd.unique(models)

def preprocess_to_numpy(df):
    # Combine model_a and model_b, then factorize
    all_models = pd.concat([df['model_a'], df['model_b']])
    models, model_indices = pd.factorize(all_models)
    
    # Split the factorized indices back into model_a and model_b
    n_rows = len(df)
    model_a_indices = models[:n_rows]
    model_b_indices = models[n_rows:]
    
    # Create matchups array
    matchups = np.column_stack([model_a_indices, model_b_indices])
    
    # Create outcomes array
    outcomes = np.where(df["winner"] == "model_a", 1.0, 
                        np.where(df["winner"] == "model_b", 0.0, 0.5))
    
    return matchups, outcomes, model_indices.to_numpy()


def fit_vectorized_elo(
    matchups,
    outcomes,
    sample_indices,
    num_models,
    k=4.0,
    base=10.0,
    init_rating=1000.0,
    scale=400.0,
    ):
    """fit multiple sets of Elo ratings on different samples of the data at the same time"""
    alpha = math.log(base) / scale
    num_samples = sample_indices.shape[1]
    ratings = np.zeros(shape=(num_samples, num_models), dtype=np.float64)
    # iterate over the rows of sample_indices, each column is an index into a match in the input arrays
    sample_range = np.arange(num_samples)
    for matchup_indices in sample_indices:
        model_a_indices = matchups[matchup_indices,0]
        model_b_indices = matchups[matchup_indices,1]
        model_a_ratings = ratings[sample_range, model_a_indices]
        model_b_ratings = ratings[sample_range, model_b_indices]
        sample_outcomes = outcomes[matchup_indices]
        probs = expit(alpha * (model_a_ratings - model_b_ratings))
        updates = k * (sample_outcomes - probs)
        ratings[sample_range, model_a_indices] += updates
        ratings[sample_range, model_b_indices] -= updates
    return ratings + init_rating

def compute_elo_ratings(df, k=4.0, base=10.0, init_rating=1000.0, scale=400.0):
    matchups, outcomes, models = preprocess_to_numpy(df)
    alpha = math.log(base) / scale
    ratings = np.full(shape=(len(models),), fill_value=init_rating)
    for (model_a_idx, model_b_idx), outcome in zip(matchups, outcomes):
        prob = 1.0 / (1.0 + math.exp(alpha * (ratings[model_b_idx] - ratings[model_a_idx])))
        update = k * (outcome - prob)
        ratings[model_a_idx] += update
        ratings[model_b_idx] -= update
    return {model:ratings[idx] for idx,model in enumerate(models)}

def compute_bootstrap_elo_ratings(df, num_round=100, k=4.0, base=10.0, init_rating=1000.0, scale=400.0):
    matchups, outcomes, models = preprocess_to_numpy(df)
    sample_indices = np.random.randint(low=0, high=len(df), size=(len(df), num_round))
    ratings = fit_vectorized_elo(matchups, outcomes, sample_indices, len(models), k, base, init_rating, scale)
    df = pd.DataFrame(data=ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]

