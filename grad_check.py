import numpy as np
from faster import bt_loss_and_grad

def generate_synthetic_data(C, N):
    ratings = np.random.randn(C)
    matchups = np.random.randint(0, C, size=(N, 2))
    outcomes = np.random.choice([0.0, 1.0], size=N)
    weights = np.random.rand(N)
    return ratings, matchups, outcomes, weights

def finite_difference_grad(f, x, eps=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def grad_check():
    C, N = 139, 50_000
    ratings, matchups, outcomes, weights = generate_synthetic_data(C, N)
    alpha = np.log(10.0)

    def loss_func(r):
        return bt_loss_and_grad(r, matchups, outcomes, weights, alpha)[0]

    analytic_loss, analytic_grad = bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha)
    numeric_grad = finite_difference_grad(loss_func, ratings)

    rel_error = np.abs(analytic_grad - numeric_grad) / (np.abs(analytic_grad) + np.abs(numeric_grad))
    mean_rel_error = np.mean(rel_error)

    print(f"Mean relative error: {mean_rel_error}")
    print(f"Gradient check {'passed' if mean_rel_error < 1e-5 else 'failed'}")

if __name__ == "__main__":
    grad_check()