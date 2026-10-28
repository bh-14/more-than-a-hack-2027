"""A small L2-regularised logistic regression, in numpy, saved as JSON.

Small on purpose: a dozen collective features and a linear decision are enough to carry the
signal, train in seconds on the open games, and stay inspectable -- the weights file is readable
and its training script is in the repository (organisers' answer in issue #16). Fitting uses
Newton's method, so the result does not depend on a learning rate or a random seed, and inference
is a dot product on CPU, so the submission stays byte-identical.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def fit(X, y, l2=1.0, intercept=True, sample_weight=None, iterations=30):
    """Weights for P(y=1 | X). Features are standardised; the stats are stored with the model."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    if intercept:
        Z = np.column_stack([Z, np.ones(len(Z))])
    w = np.ones(len(Z)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    beta = np.zeros(Z.shape[1])
    penalty = np.full(Z.shape[1], l2)
    if intercept:
        penalty[-1] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
        gradient = Z.T @ (w * (p - y)) + penalty * beta
        hessian = (Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"coef": beta[:X.shape[1]].tolist(), "bias": float(beta[-1]) if intercept else 0.0,
            "mean": mean.tolist(), "scale": scale.tolist()}


def logit(model, X):
    """Log-odds per row, float64."""
    X = np.asarray(X, dtype=np.float64)
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return Z @ np.asarray(model["coef"]) + model["bias"]


def save(model, name, **info):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name + ".json")
    with open(path, "w", newline="\n") as handle:
        json.dump(dict(model, **info), handle, indent=1, sort_keys=True)
        handle.write("\n")
    return path


def load(name):
    """The model, or None when it has not been trained yet."""
    path = os.path.join(MODELS_DIR, name + ".json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)
