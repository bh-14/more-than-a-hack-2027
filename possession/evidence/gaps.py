"""M3 -- what the ball probably did while the tracking lost it.

The ball is missing on 41% of graded frames, mostly in crowds, duels and the air (DATA.md). Two
cheap ideas, both deliberately weaker than seeing the ball:

  * Short gaps: the ball barely changes course in under a second, so a straight line between the
    last and next sighting is a usable position, and control is scored on it at reduced weight.
  * Bridges: if the ball vanished at the feet of one side and reappears at the feet of the same
    side, the possession very likely did not change in between. A small vote for that side over
    the whole gap stops the decoder inventing a hand-over out of noise.

Next steps for the owner of this module: extrapolate a pass from the last sighting to estimate
when and to whom it arrives, and use that to place a transition inside a gap.
"""

import numpy as np

from .. import geometry
from ..contract import A, H, Evidence
from .control import team_scores


def gap_bounds(seen):
    """For every frame, the index of the last seen frame at or before it and the next at or after.

    -1 / n where there is none.
    """
    n = len(seen)
    index = np.arange(n)
    prev_seen = np.maximum.accumulate(np.where(seen, index, -1))
    next_seen = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    return prev_seen, next_seen


def interpolate_ball(ball_xy, max_gap):
    """Linear fill across gaps of at most max_gap frames. Returns (xy, filled_mask)."""
    seen = np.isfinite(ball_xy[:, 0])
    out = ball_xy.copy()
    filled = np.zeros(len(seen), dtype=bool)
    if seen.sum() < 2:
        return out, filled
    prev_seen, next_seen = gap_bounds(seen)
    length = next_seen - prev_seen - 1
    filled = (~seen) & (prev_seen >= 0) & (next_seen < len(seen)) & (length <= max_gap)
    index = np.flatnonzero(seen)
    for axis in (0, 1):
        out[filled, axis] = np.interp(np.flatnonzero(filled), index, ball_xy[index, axis])
    return out, filled


def compute(match, features, params):
    scores = np.zeros((match.n_frames, 3), dtype=np.float32)

    # Short gaps: score control on the interpolated ball.
    ball, filled = interpolate_ball(match.ball_xy, params["max_interp_frames"])
    if filled.any():
        dist_home, dist_away = geometry.nearest_by_team(
            match.players_xy[filled], match.players_team, ball[filled])
        vote, _ = team_scores(dist_home, dist_away, np.full(filled.sum(), np.nan), params)
        vote *= params["interp_weight"]
        scores[filled, H] += vote
        scores[filled, A] -= vote

    # Bridges: same side at both ends of a longer gap.
    seen = features.ball_seen
    vote_seen, _ = team_scores(features.dist_home, features.dist_away, features.ball_speed,
                               params)
    prev_seen, next_seen = gap_bounds(seen)
    n = match.n_frames
    length = next_seen - prev_seen - 1
    bridge = ((~seen) & ~filled & (prev_seen >= 0) & (next_seen < n)
              & (length <= params["max_bridge_frames"]))
    if bridge.any():
        before = vote_seen[prev_seen[bridge]]
        after = vote_seen[np.minimum(next_seen[bridge], n - 1)]
        agree = (np.sign(before) == np.sign(after)) & (np.abs(before) >= params["bridge_min_vote"]) \
            & (np.abs(after) >= params["bridge_min_vote"])
        vote = np.where(agree, np.sign(before) * params["bridge_score"], 0.0).astype(np.float32)
        scores[bridge, H] += vote
        scores[bridge, A] -= vote
    return Evidence("gaps", scores)
