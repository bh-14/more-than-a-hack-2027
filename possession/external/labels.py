"""Bring open-data labels to the challenge's definition of possession.

DFL BallPossession changes hands on any touch, so its spells are short: ~23 per 5 minutes, median
3-4 s on the seven open games. The challenge only changes possession on *controlled* possession;
a touch, a deflection or a won header without control does not count, and in loose phases the
ball stays with the last team that had it (CHALLENGE.md). Scored against raw DFL labels, a
decoder would be rewarded for reporting touches.

Rule, set a priori from that text: a team spell shorter than min_control_s that follows a spell of
the other side never became control, so it is handed back to the side before it. Stoppages are
untouched. The threshold is a prior, not fitted; DATA.md's published summary (17 possessions in
the public five minutes, mean 14.2 s) is only a sanity check on the result.
"""

import numpy as np

from ..contract import D


def controlled_possession(gt, runs, min_control_frames):
    """Copy of gt where short H/A spells after the other side's spell become that side's."""
    out = gt.copy()
    for start, stop in runs:
        window = out[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        previous_team = None
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            code = window[a]
            if code == D:
                previous_team = None  # a stoppage resets: the restart is a fresh possession
                continue
            if previous_team is not None and code != previous_team and b - a < min_control_frames:
                window[a:b] = previous_team
                continue
            previous_team = code
        out[start:stop] = window
    return out


def spell_summary(gt, runs, fps):
    """(spells per 5 minutes of graded play, mean spell length in s)."""
    from ..scoring import pm
    spells = pm.possession_spells(gt, runs)
    graded_min = sum(stop - start for start, stop in runs) / fps / 60.0
    lengths = np.array([stop - start for start, stop, _ in spells]) / fps
    return len(spells) / graded_min * 5.0, float(lengths.mean()) if len(lengths) else 0.0
