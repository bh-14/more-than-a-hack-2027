import numpy as np

from possession import features as collective
from possession import geometry, models
from possession.contract import AWAY, HOME, Match
from synthetic import make_match


def test_fit_recovers_direction_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] - 2 * X[:, 1] + 0.3 * rng.normal(size=2000)) > 0
    first = models.fit(X, y)
    assert first == models.fit(X, y)
    assert first["coef"][0] > 0 > first["coef"][1]
    assert np.mean((models.logit(first, X) > 0) == y) > 0.9


def test_shape_features_flip_when_sides_swap():
    match, _ = make_match(seed=1)
    swapped = Match(name="swapped", fps=match.fps, pitch=match.pitch,
                    players_xy=match.players_xy,
                    players_team=np.where(match.players_team == HOME, AWAY, HOME).astype(np.int8),
                    ball_xy=match.ball_xy, halves=match.halves)
    _, shape, _, dead = collective.compute(match, geometry.compute(match))
    _, shape_swapped, _, dead_swapped = collective.compute(swapped, geometry.compute(swapped))
    assert np.allclose(shape, -shape_swapped, atol=1e-4)
    assert np.allclose(dead[:, :2], dead_swapped[:, :2], atol=1e-4)  # pitch pace is side-blind
