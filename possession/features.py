"""Per-frame collective features for the learned parts of M1 (dead ball) and M4 (team shape).

Everything is per side or for the whole pitch, never per slot, and everything is computed from
positions, so the provided ATD and the degraded open games produce the same quantities.

Positions are first mirrored so that each side attacks towards +x. Which goal a side defends is
read from the data, half by half: the side whose centroid sits lower on average defends x = 0.

Two families:
  * SHAPE   home-minus-away differences only. A model on these with no intercept is antisymmetric
            by construction -- swap the sides and the vote flips -- so it cannot learn "the home
            side has the ball more" from seven Bundesliga games.
  * DEAD    the whole pitch plus the ball, for "is the ball in play?".
"""

import warnings

import numpy as np

from .contract import AWAY, HOME
from .geometry import finite_speed, rolling_fraction, rolling_nanmean, MAX_PLAYER_SPEED

# Params
SHORT = 12   # +/- frames, ~0.5 s
LONG = 50    # +/- frames, ~2 s
GAP_CAP = 250  # frames; "ball unseen for 10 s" and "for a minute" are the same information
BOX_DEPTH, BOX_WIDTH = 16.5, 40.3  # m, penalty area


def attack_sign(match):
    """(n_frames, 2) +1 where that side attacks towards +x, -1 where towards 0. 0 outside halves."""
    sign = np.zeros((match.n_frames, 2), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start, stop in match.halves:
            mean_x = [np.nanmean(match.players_xy[start:stop, match.players_team == team, 0])
                      for team in (HOME, AWAY)]
            home_low = np.nan_to_num(mean_x[0], nan=0.0) <= np.nan_to_num(mean_x[1], nan=0.0)
            sign[start:stop, HOME] = 1.0 if home_low else -1.0
            sign[start:stop, AWAY] = -1.0 if home_low else 1.0
    return sign


def _side_stats(x_attack, y, fps):
    """Shape of one side, x already oriented so the side attacks towards +x."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid = np.nanmean(x_attack, axis=1)
        ordered = np.sort(x_attack, axis=1)  # NaN sort last
        count = np.isfinite(x_attack).sum(axis=1)
        second = np.where(count >= 2, ordered[:, 1], np.nan)  # deepest outfield line (GK first)
        top = np.nanmax(x_attack, axis=1)
        stats = {
            "centroid": centroid,
            "spread_x": np.nanstd(x_attack, axis=1),
            "spread_y": np.nanstd(y, axis=1),
            "length": top - second,
            "width": np.nanmax(y, axis=1) - np.nanmin(y, axis=1),
            "last_line": second,
            "front": top,
            "advance": rolling_nanmean(np.gradient(rolling_nanmean(centroid, SHORT)) * fps, SHORT),
        }
    return stats


def compute(match, features):
    """(shape_names, shape_X, dead_names, dead_X), float32 arrays of shape (n_frames, k)."""
    n = match.n_frames
    length, width = match.pitch
    sign = attack_sign(match)
    xy = match.players_xy
    sides = {}
    for team in (HOME, AWAY):
        cols = match.players_team == team
        x = xy[:, cols, 0]
        x_attack = np.where(sign[:, team:team + 1] > 0, x, length - x)
        sides[team] = _side_stats(x_attack, xy[:, cols, 1], match.fps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            speed = finite_speed(xy[:, cols], match.fps, max_speed=MAX_PLAYER_SPEED)
            sides[team]["pace"] = rolling_nanmean(np.nanmedian(speed, axis=1), SHORT)
            sides[team]["in_attack_half"] = rolling_nanmean(
                (x_attack > length / 2).sum(axis=1).astype(float), SHORT)

    shape_names, shape_cols = [], []
    for key in sides[HOME]:
        shape_names.append("diff_" + key)
        shape_cols.append(sides[HOME][key] - sides[AWAY][key])
    shape = np.nan_to_num(np.column_stack(shape_cols), nan=0.0).astype(np.float32)

    seen = features.ball_seen
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    nxt = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    since = np.where(last >= 0, np.minimum(index - last, GAP_CAP), GAP_CAP) / GAP_CAP
    until = np.where(nxt < n, np.minimum(nxt - index, GAP_CAP), GAP_CAP) / GAP_CAP
    bx = match.ball_xy[np.maximum(last, 0), 0]
    by = match.ball_xy[np.maximum(last, 0), 1]
    edge = np.fmin(np.fmin(bx, length - bx), np.fmin(by, width - by))
    edge = np.where(last >= 0, np.clip(edge, -5.0, 20.0), 20.0) / 20.0
    ball_speed = np.nan_to_num(features.ball_speed, nan=0.0)
    still = seen & (ball_speed < 0.5)

    with np.errstate(invalid="ignore"):
        x, y = xy[..., 0], xy[..., 1]
        lateral = np.abs(y - width / 2) < BOX_WIDTH / 2
        in_box = ((x < BOX_DEPTH) | (x > length - BOX_DEPTH)) & lateral
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        all_x_spread = np.nanstd(x, axis=1)
        all_y_spread = np.nanstd(y, axis=1)

    dead_cols = {
        "pace_short": rolling_nanmean(features.median_speed, SHORT),
        "pace_long": rolling_nanmean(features.median_speed, LONG),
        "pace_home": sides[HOME]["pace"],
        "pace_away": sides[AWAY]["pace"],
        "players": features.n_players / 22.0,
        "ball_seen_share": rolling_fraction(seen, LONG),
        "ball_speed": rolling_nanmean(ball_speed, SHORT // 2),
        "ball_still_share": rolling_fraction(still, LONG),
        "ball_since": since,
        "ball_until": until,
        "ball_edge": edge,
        "in_box": rolling_nanmean(in_box.sum(axis=1).astype(float), SHORT) / 22.0,
        "spread_x": all_x_spread / length,
        "spread_y": all_y_spread / width,
        "centroid_gap": np.sqrt(((features.centroid[:, HOME] - features.centroid[:, AWAY]) ** 2)
                                .sum(axis=1)) / length,
    }
    dead_names = list(dead_cols)
    dead = np.nan_to_num(np.column_stack([dead_cols[k] for k in dead_names]),
                         nan=0.0).astype(np.float32)
    return shape_names, shape, dead_names, dead


def cached(match, features):
    """compute(), once per Features object: M1 and M4 both read it."""
    if getattr(features, "_collective", None) is None:
        features._collective = compute(match, features)
    return features._collective
