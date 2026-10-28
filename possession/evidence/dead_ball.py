"""M1 -- stoppages and restarts.

Every missed stoppage merges two possessions and every invented one splits a possession in two,
so this module moves spellF1 more than any other. The baseline's only notion of a dead ball is
"the tracking lost the ball", which is wrong on 41% of graded frames.

v1 signals:
  * ball seen outside the pitch
  * almost nobody tracked (before kickoff, halftime, after the whistle)
  * the whole pitch slows down (whistle -> players walk); a fast pitch is evidence of live play
  * the ball sits still for a while (placed for a free kick, corner, goal kick)

Next steps for the owner: restart templates (kickoff, corner, goal kick, throw-in) from player
and ball positions, Smart Tagging SET PIECES as restart anchors, the long stoppage after a goal,
and which side restarts (throw-ins and goal kicks usually go to the side that did not have it).
"""

import numpy as np

from ..contract import D, Evidence
from ..geometry import rolling_fraction, rolling_nanmean


def prev_seen_index(seen):
    """Index of the last frame at or before each frame where the ball was seen, -1 if none."""
    return np.maximum.accumulate(np.where(seen, np.arange(len(seen)), -1))


def compute(match, features, params):
    scores = np.zeros((match.n_frames, 3), dtype=np.float32)
    length, width = match.pitch
    x, y = match.ball_xy[:, 0], match.ball_xy[:, 1]
    margin = params["out_margin"]

    with np.errstate(invalid="ignore"):
        outside = features.ball_seen & ((x < -margin) | (x > length + margin)
                                        | (y < -margin) | (y > width + margin))
    scores[outside, D] += params["out_score"]

    # The tactical camera frames the pitch, so a ball that crosses a line usually just vanishes.
    # A ball last seen at the edge and not seen since is probably out.
    last = prev_seen_index(features.ball_seen)
    unseen = ~features.ball_seen & (last >= 0)
    lx, ly = x[np.maximum(last, 0)], y[np.maximum(last, 0)]
    edge = params["edge_margin"]
    with np.errstate(invalid="ignore"):
        at_edge = (lx < edge) | (lx > length - edge) | (ly < edge) | (ly > width - edge)
    scores[unseen & at_edge, D] += params["edge_score"]

    empty = features.n_players < params["min_players"]
    scores[empty, D] += params["empty_score"]

    pace = rolling_nanmean(features.median_speed, params["slow_window"])
    slow = np.clip((params["slow_speed"] - pace) / params["slow_scale"], -1.0, 1.0)
    scores[:, D] += params["slow_score"] * np.nan_to_num(slow, nan=0.0).astype(np.float32)

    still = features.ball_seen & (np.nan_to_num(features.ball_speed, nan=np.inf)
                                  < params["still_speed"])
    share = rolling_fraction(still, params["still_window"])
    scores[:, D] += params["still_score"] * np.clip(2.0 * share - 1.0, 0.0, 1.0).astype(np.float32)
    return Evidence("dead_ball", scores)
