"""Make clean tracking look like the provided automatic tracking (ATD).

The open datasets are corrected, identity-preserving tracking with the ball almost always present.
The provided ATD is not (DATA.md): no identities, slots that vanish and swap, ~20.5 tracked objects
instead of 22, and the ball missing on 41% of graded frames -- concentrated in crowds, duels and
the air. A threshold tuned on clean data would be tuned for a different problem, so every external
evaluation runs on a degraded copy.

Every number below is a prior taken from DATA.md or set by hand, not measured on the provided
labels. Calibrating them on unlabelled statistics of the provided tracking (gap lengths, object
counts) is a question for the organisers before anyone does it.

Deterministic for a given seed.
"""

import numpy as np

from ..contract import Match

DEFAULTS = {
    "ball_missing_share": 0.41,   # DATA.md, graded frames
    "ball_gap_mean_s": 1.5,       # prior
    "crowd_radius": 3.0,          # m
    "crowd_count": 3,             # players within crowd_radius make a crowd
    "crowd_boost": 4.0,           # relative hazard of losing the ball in a crowd
    "air_height": 2.0,            # m; above this the ball is "in the air"
    "air_boost": 4.0,
    "outside_missing": 0.9,       # the tactical camera frames the pitch, not the touchline
    "player_missing_share": 0.07, # 22 -> ~20.5 tracked
    "player_gap_mean_s": 2.0,
    "swap_rate_per_min": 2.0,     # identity swaps between same-side slots, per minute
    "jitter_m": 0.15,
}


def _markov_gaps(rng, n, hazard, mean_gap):
    """Boolean 'missing' mask from a two-state chain: enter a gap with per-frame probability
    hazard[t], leave it with probability 1/mean_gap."""
    leave = 1.0 / max(mean_gap, 1.0)
    draws = rng.random(n)
    missing = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        state = (draws[t] >= leave) if state else (draws[t] < hazard[t])
        missing[t] = state
    return missing


def _calibrated_hazard(base, share, mean_gap):
    """Scale a relative hazard so the stationary missing share is about `share`."""
    # Stationary share of a two-state chain = h / (h + 1/mean_gap). Solve for the mean h.
    target_mean = share / (1.0 - share) / mean_gap
    return np.clip(base * (target_mean / max(base.mean(), 1e-9)), 0.0, 1.0)


def degrade(match, seed=0, **overrides):
    p = dict(DEFAULTS, **overrides)
    rng = np.random.default_rng(seed)
    n, slots = match.n_frames, match.n_slots
    players = match.players_xy.copy()
    ball = match.ball_xy.copy()

    # Jitter.
    players += rng.normal(0.0, p["jitter_m"], players.shape).astype(np.float32)
    ball += rng.normal(0.0, p["jitter_m"], ball.shape).astype(np.float32)

    # Ball dropout, likelier in crowds and in the air.
    distance = np.sqrt(((match.players_xy - match.ball_xy[:, None, :]) ** 2).sum(-1))
    crowd = (np.nan_to_num(distance, nan=np.inf) < p["crowd_radius"]).sum(1) >= p["crowd_count"]
    relative = 1.0 + (p["crowd_boost"] - 1.0) * crowd
    if match.ball_z is not None:
        relative = relative + (p["air_boost"] - 1.0) * (np.nan_to_num(match.ball_z) > p["air_height"])
    gap = p["ball_gap_mean_s"] * match.fps
    hazard = _calibrated_hazard(relative, p["ball_missing_share"], gap)
    lost = _markov_gaps(rng, n, hazard, gap)
    x, y = match.ball_xy[:, 0], match.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        outside = (x < 0) | (x > match.pitch[0]) | (y < 0) | (y > match.pitch[1])
    lost |= outside & (rng.random(n) < p["outside_missing"])
    ball[lost] = np.nan

    # Player dropout.
    gap = p["player_gap_mean_s"] * match.fps
    hazard = np.full(n, p["player_missing_share"] / (1.0 - p["player_missing_share"]) / gap)
    for slot in range(slots):
        players[_markov_gaps(rng, n, hazard, gap), slot] = np.nan

    # Identity swaps: from a random frame on, two same-side slots trade trajectories, which is
    # what an ATD identity switch looks like to anything that differentiates positions per slot.
    n_swaps = rng.poisson(p["swap_rate_per_min"] * n / (60.0 * match.fps))
    for frame in np.sort(rng.integers(0, n, n_swaps)):
        side = match.players_team[rng.integers(0, slots)]
        same = np.flatnonzero(match.players_team == side)
        if len(same) < 2:
            continue
        a, b = rng.choice(same, 2, replace=False)
        players[frame:, [a, b]] = players[frame:, [b, a]]

    return Match(name=match.name + "+noise", fps=match.fps, pitch=match.pitch,
                 players_xy=players, players_team=match.players_team.copy(), ball_xy=ball,
                 halves=list(match.halves), ball_z=None, events=list(match.events))
