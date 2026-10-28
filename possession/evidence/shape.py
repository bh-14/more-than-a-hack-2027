"""M4 -- possession from the shape of both teams, mostly for frames where the ball is not seen.

The side with the ball pushes up and spreads out; the side without it drops and narrows. A
logistic regression on home-minus-away shape differences (possession/features.py) turns that into
log-odds of home possession. It has no intercept, so it is antisymmetric: it cannot carry a
home-side bias from the open games.

Trained by scripts/train_models.py on the open games' train split (degraded by the noise model);
weights in models/shape_v1.json. Where the ball is seen, M2 knows better, so the vote is scaled
down there by seen_factor.

Next steps for the owner: pressure features around the last known ball position, a short
temporal context (the shape lags the hand-over by a second or two), and a small sequence model if
the linear one saturates.
"""

import numpy as np

from .. import features as collective
from .. import models
from ..contract import A, H, Evidence

MODEL = "shape_v1"


def compute(match, features, params):
    model = models.load(MODEL)
    if model is None:
        return Evidence.empty("shape", match.n_frames)
    names, X, _, _ = collective.cached(match, features)
    if names != model["features"]:
        raise ValueError("{} was trained on other features; retrain it".format(MODEL))
    odds = np.clip(models.logit(model, X), -params["clip"], params["clip"])
    odds *= np.where(features.ball_seen, params["seen_factor"], 1.0)
    scores = np.zeros((match.n_frames, 3), dtype=np.float32)
    scores[:, H] = 0.5 * odds
    scores[:, A] = -0.5 * odds
    return Evidence("shape", scores)
