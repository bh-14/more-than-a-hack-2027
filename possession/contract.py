"""The two objects every part of the pipeline agrees on.

A loader turns some tracking source into a Match. An evidence module reads a Match (through
geometry.Features) and returns an Evidence. The decoder sums the Evidence it is given and labels
each half. Keeping that contract identical for the provided game and for the open datasets is what
lets every threshold be chosen on other games and applied unchanged here.

Codes are the scorer's (evaluation/possession_metrics.py): H=0, A=1, D=2, X=3.
"""

from dataclasses import dataclass, field

import numpy as np

H, A, D, X = 0, 1, 2, 3
STATES = ("H", "A", "D")
N_STATES = len(STATES)
LETTERS = np.array(["H", "A", "D", "X"])

HOME, AWAY = 0, 1


@dataclass
class Match:
    """Tracking for one match, on its own frame index, in metres with the origin at a corner.

    players_xy:   (n_frames, n_slots, 2) float32, NaN where a slot is not tracked.
    players_team: (n_slots,) int8, HOME or AWAY. A slot is a tracklet, not a person.
    ball_xy:      (n_frames, 2) float32, NaN where the ball is not seen.
    halves:       [(start, stop)] half-open spans that get a label; everything else is written X.
    ball_z:       optional (n_frames,) height in metres. The provided ATD has none.
    events:       optional Smart Tagging instances, frames already on this match's index.
    """
    name: str
    fps: int
    pitch: tuple
    players_xy: np.ndarray
    players_team: np.ndarray
    ball_xy: np.ndarray
    halves: list
    ball_z: np.ndarray = None
    events: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.ball_xy)
        if self.ball_xy.shape != (n, 2):
            raise ValueError("ball_xy must be (n_frames, 2), got {}".format(self.ball_xy.shape))
        if self.players_xy.ndim != 3 or self.players_xy.shape[0] != n or self.players_xy.shape[2] != 2:
            raise ValueError("players_xy must be (n_frames, n_slots, 2), got {}".format(
                self.players_xy.shape))
        if self.players_team.shape != (self.players_xy.shape[1],):
            raise ValueError("players_team must have one entry per slot")
        for start, stop in self.halves:
            if not 0 <= start < stop <= n:
                raise ValueError("half ({}, {}) outside 0..{}".format(start, stop, n))

    @property
    def n_frames(self):
        return len(self.ball_xy)

    @property
    def n_slots(self):
        return self.players_xy.shape[1]


@dataclass
class Evidence:
    """Per-frame additive scores for H, A and D, in log-likelihood units.

    A row of zeros means "no opinion". Positive pushes the decoder towards that state. Modules do
    not need to normalise: the decoder weights each Evidence by name (configs/params.toml), which
    is where the relative trust between signals is set -- on other games, never on this one.
    """
    name: str
    scores: np.ndarray

    def __post_init__(self):
        if self.scores.ndim != 2 or self.scores.shape[1] != N_STATES:
            raise ValueError("{}: scores must be (n_frames, 3), got {}".format(
                self.name, self.scores.shape))
        if not np.isfinite(self.scores).all():
            raise ValueError("{}: scores contain NaN or inf".format(self.name))

    @classmethod
    def empty(cls, name, n_frames):
        return cls(name, np.zeros((n_frames, N_STATES), dtype=np.float32))
