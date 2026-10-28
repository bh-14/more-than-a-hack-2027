import numpy as np

from possession.contract import A, D, H
from possession.decode import decode

PARAMS = {"switch_cost": 5.0, "to_dead_cost": 5.0, "from_dead_cost": 5.0,
          "min_possession_frames": 25, "min_dead_frames": 25}


def test_no_evidence_gives_one_state():
    codes = decode(np.zeros((500, 3)), PARAMS)
    assert len(set(codes.tolist())) == 1


def test_switch_lands_where_evidence_changes():
    scores = np.zeros((400, 3))
    scores[:200, H] = 1.0
    scores[200:, A] = 1.0
    codes = decode(scores, PARAMS)
    assert (codes[:200] == H).all() and (codes[200:] == A).all()


def test_short_blip_is_absorbed():
    scores = np.zeros((300, 3))
    scores[:, H] = 0.5
    scores[140:150, A] = 2.0  # 10 frames, shorter than the 25-frame minimum
    codes = decode(scores, PARAMS)
    assert (codes == H).all()


def test_dead_ball_splits_possession():
    scores = np.zeros((300, 3))
    scores[:, H] = 1.0
    scores[100:160, D] = 3.0
    codes = decode(scores, PARAMS)
    assert (codes[100:160] == D).all()
    assert (codes[:100] == H).all() and (codes[160:] == H).all()


def test_deterministic():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(1000, 3))
    assert np.array_equal(decode(scores, PARAMS), decode(scores, PARAMS))
