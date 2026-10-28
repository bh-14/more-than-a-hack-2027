import numpy as np

from possession.external import noise
from synthetic import make_match


def test_ball_missing_share_close_to_target():
    match, _ = make_match(seed=2, half_s=600)
    noisy = noise.degrade(match, seed=2)
    inside = np.concatenate([np.arange(a, b) for a, b in match.halves])
    on_pitch = inside[(match.ball_xy[inside, 0] >= 0) & (match.ball_xy[inside, 0] <= match.pitch[0])]
    missing = np.isnan(noisy.ball_xy[on_pitch, 0]).mean()
    assert 0.33 < missing < 0.49


def test_same_seed_same_output():
    match, _ = make_match(seed=4)
    first = noise.degrade(match, seed=7)
    second = noise.degrade(match, seed=7)
    assert np.array_equal(first.ball_xy, second.ball_xy, equal_nan=True)
    assert np.array_equal(first.players_xy, second.players_xy, equal_nan=True)
