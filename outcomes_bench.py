import numpy as np
from data_utils import load_data
import timeit

def outcomes_a(df):
    """Calculate outcomes using nested np.where"""
    return np.where(df["winner"] == "model_a", 1.0, 
                    np.where(df["winner"] == "model_b", 0.0, 0.5))

def outcomes_b(df):
    """Calculate outcomes using np.select"""
    conditions = [
        (df["winner"] == "model_a"),
        (df["winner"] == "model_b")
    ]
    choices = [1.0, 0.0]
    return np.select(conditions, choices, default=0.5)

def outcomes_c(df):
    """Calculate outcomes using array initialization and indexing"""
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return outcomes

df = load_data(use_preprocessed=True)

number = 100


time_a = timeit.timeit('outcomes_a(df)', globals=globals(), number=number)
time_b = timeit.timeit('outcomes_b(df)', globals=globals(), number=number)
time_c = timeit.timeit('outcomes_c(df)', globals=globals(), number=number)

print(f"\nMethod a (nested np.where) average time: {time_a/number:.6f} seconds")
print(f"Method b (np.select) average time: {time_b/number:.6f} seconds")
print(f"Method c (array init and indexing) average time: {time_c/number:.6f} seconds")


