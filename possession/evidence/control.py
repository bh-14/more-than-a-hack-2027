"""M2 -- who controls the ball, on frames where the tracking sees it.

The baseline asks "who is nearest". This asks two more things: is anyone near enough to be
playing the ball at all, and is the ball travelling (a pass or a shot, so the nearest player is
just the one it is flying past)? Both shrink the vote rather than flip it, and the decoder's
switching cost does the rest: one close frame is a touch, a run of them is control.
"""

import numpy as np

from ..contract import A, H, Evidence


def team_scores(dist_home, dist_away, ball_speed, params):
    """Signed per-frame vote: positive for home, negative for away, 0 where there is no ball.

    Also returns how sure the vote is about *anyone* playing the ball (0..1), which dead_ball
    reads as evidence that the ball is alive.
    """
    home = np.where(np.isnan(dist_home), np.inf, dist_home)
    away = np.where(np.isnan(dist_away), np.inf, dist_away)
    near = np.fmin(home, away)

    closeness = np.clip(1.0 - (near - params["control_radius"]) / params["falloff"], 0.0, 1.0)
    closeness[~np.isfinite(near)] = 0.0
    in_flight = np.nan_to_num(ball_speed, nan=0.0) > params["flight_speed"]
    trust = closeness * np.where(in_flight, params["flight_factor"], 1.0)

    with np.errstate(invalid="ignore"):
        side = np.tanh((away - home) / params["contest_margin"])
    side = np.nan_to_num(side, nan=0.0, posinf=1.0, neginf=-1.0)
    return (trust * side).astype(np.float32), closeness.astype(np.float32)


def compute(match, features, params):
    vote, closeness = team_scores(features.dist_home, features.dist_away, features.ball_speed,
                                  params)
    scores = np.zeros((match.n_frames, 3), dtype=np.float32)
    scores[:, H] = vote
    scores[:, A] = -vote
    return Evidence("control", scores)
