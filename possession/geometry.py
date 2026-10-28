"""Quantities several evidence modules read, computed once per match.

Everything here is collective or per-team, never per-player: slots are tracklets that appear,
vanish and swap identity (DATA.md), so nothing may assume a slot follows one person. Speeds are
recomputed from positions for every source, provided or open, so that a threshold chosen on an
open dataset means the same thing on the provided game.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .contract import AWAY, HOME

# Params
# Anything faster is a tracking artefact (an identity switch or a re-detection), not running.
MAX_PLAYER_SPEED = 12.0  # m/s
# Hardest shots are ~35 m/s; above that the "ball" jumped between two detections.
MAX_BALL_SPEED = 40.0  # m/s
# Centred difference over +/- this many frames. 2 frames at 25 fps = 0.16 s, enough to damp
# per-frame jitter without smearing a pass.
SPEED_LAG = 2


def rolling_nanmean(values, half_window):
    """Centred mean over +/- half_window frames, ignoring NaN. NaN where the window is empty.

    Cumulative sums, so the cost does not depend on the window.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    out = np.full(len(values), np.nan)
    ok = n > 0
    out[ok] = (total[high] - total[low])[ok] / n[ok]
    return out


def rolling_fraction(mask, half_window):
    """Share of True in a centred window of +/- half_window frames."""
    mask = np.asarray(mask, dtype=np.float64)
    total = np.concatenate(([0.0], np.cumsum(mask)))
    index = np.arange(len(mask))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(mask), index + half_window + 1)
    return (total[high] - total[low]) / (high - low)


def finite_speed(xy, fps, lag=SPEED_LAG, max_speed=None):
    """Speed in m/s from a centred difference over +/- lag frames.

    xy is (n_frames, ..., 2). NaN where either end is missing or the speed is implausible.
    """
    n = len(xy)
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if n > 2 * lag:
        step = xy[2 * lag:] - xy[:-2 * lag]
        speed[lag:n - lag] = np.sqrt((step ** 2).sum(axis=-1)) * fps / (2.0 * lag)
    if max_speed is not None:
        speed[speed > max_speed] = np.nan
    return speed


def nearest_by_team(players_xy, players_team, point_xy):
    """(dist_home, dist_away): distance from point_xy to the closest tracked player of each side.

    players_xy is (n, slots, 2), point_xy is (n, 2). NaN where the point or the whole side is
    missing.
    """
    distance = np.sqrt(((players_xy - point_xy[:, None, :]) ** 2).sum(axis=-1))
    distance = np.where(np.isnan(distance), np.inf, distance)
    out = []
    for team in (HOME, AWAY):
        best = distance[:, players_team == team].min(axis=1) if (players_team == team).any() \
            else np.full(len(point_xy), np.inf)
        best = best.astype(np.float32)
        best[~np.isfinite(best)] = np.nan
        out.append(best)
    return out[0], out[1]


@dataclass
class Features:
    ball_seen: np.ndarray       # (n,) bool
    ball_speed: np.ndarray      # (n,) m/s, NaN where unknown
    dist_home: np.ndarray       # (n,) m, nearest home player to the ball
    dist_away: np.ndarray       # (n,) m, nearest away player to the ball
    n_players: np.ndarray       # (n,) tracked slots on the frame
    median_speed: np.ndarray    # (n,) m/s, median over tracked slots
    centroid: np.ndarray        # (n, 2, 2) per side (HOME, AWAY), NaN where a side is absent


def compute(match):
    """Features for one Match. The expensive step shared by every evidence module."""
    players = match.players_xy
    ball = match.ball_xy
    tracked = np.isfinite(players[..., 0])

    with warnings.catch_warnings():
        # All-NaN rows (halftime, empty frames) are expected; they come back as NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        player_speed = finite_speed(players, match.fps, max_speed=MAX_PLAYER_SPEED)
        median_speed = np.nanmedian(player_speed, axis=1).astype(np.float32)
        centroid = np.stack([np.nanmean(players[:, match.players_team == team, :], axis=1)
                             for team in (HOME, AWAY)], axis=1).astype(np.float32)

    dist_home, dist_away = nearest_by_team(players, match.players_team, ball)
    return Features(
        ball_seen=np.isfinite(ball[:, 0]),
        ball_speed=finite_speed(ball, match.fps, max_speed=MAX_BALL_SPEED),
        dist_home=dist_home,
        dist_away=dist_away,
        n_players=tracked.sum(axis=1).astype(np.int16),
        median_speed=median_speed,
        centroid=centroid,
    )
