import numpy as np

from possession import baselines, pipeline
from possession.contract import X
from possession.external import noise
from possession.scoring import score
from synthetic import make_match


def test_labels_only_inside_halves():
    match, gt = make_match()
    codes = pipeline.label_match(match, pipeline.load_params())
    assert len(codes) == match.n_frames
    inside = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        inside[start:stop] = True
    assert (codes[~inside] == X).all()
    assert (codes[inside] != X).all()


def test_beats_baseline_on_noisy_synthetic():
    match, gt = make_match(seed=3)
    noisy = noise.degrade(match, seed=3)
    ours = score(gt, pipeline.label_match(noisy, pipeline.load_params()))
    floor = score(gt, baselines.nearest_player(noisy, 50))
    assert ours["spell_f1_mean"] > floor["spell_f1_mean"]


def test_reproducible():
    match, _ = make_match(seed=5)
    noisy = noise.degrade(match, seed=5)
    params = pipeline.load_params()
    assert np.array_equal(pipeline.label_match(noisy, params), pipeline.label_match(noisy, params))
