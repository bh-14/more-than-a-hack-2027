"""The official baseline, re-run on any Match, so the open datasets get a floor to beat too.

Same algorithm as starter/baseline.py (nearest tracked player, D where the ball is unseen, centred
majority vote) and the same smoothing function, imported from it.
"""

import os
import sys

import numpy as np

from .contract import AWAY, HOME, X
from .geometry import nearest_by_team

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import NOBODY, smooth  # noqa: E402


def nearest_player(match, window=50):
    """Codes for the baseline with a +/- window majority vote (50 = baseline_smooth50)."""
    dist_home, dist_away = nearest_by_team(match.players_xy, match.players_team, match.ball_xy)
    home = np.nan_to_num(dist_home, nan=np.inf)
    away = np.nan_to_num(dist_away, nan=np.inf)
    raw = np.where(home <= away, HOME, AWAY).astype(np.int8)
    raw[~np.isfinite(np.fmin(home, away))] = NOBODY

    letters = smooth(raw, window)
    codes = np.full(match.n_frames, X, dtype=np.int8)
    lookup = {"H": 0, "A": 1, "D": 2}
    for start, stop in match.halves:
        codes[start:stop] = [lookup[letter] for letter in letters[start:stop]]
    return codes
