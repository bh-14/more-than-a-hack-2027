"""A small fake match with known labels, so the pipeline can be tested without any real data.

Two halves of play with a scripted sequence of possessions and stoppages. In possession, the ball
sits at one player's feet and is passed to a team-mate; a hand-over is an interception. In a
stoppage the ball goes out of play and everybody slows to a walk. Crude, but it has every
property the pipeline relies on, and its labels are exact.
"""

import numpy as np

from possession.contract import AWAY, HOME, A, D, H, Match, X

FPS = 25
PITCH = (105.0, 68.0)


def make_match(seed=0, half_s=180, gap_s=20, lead_s=10):
    rng = np.random.default_rng(seed)
    half = half_s * FPS
    lead = lead_s * FPS
    gap = gap_s * FPS
    n = lead + half + gap + half + lead
    halves = [(lead, lead + half), (lead + half + gap, lead + 2 * half + gap)]

    teams = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    home_xy = np.column_stack([rng.uniform(10, 60, 11), rng.uniform(5, 63, 11)])
    away_xy = np.column_stack([rng.uniform(45, 95, 11), rng.uniform(5, 63, 11)])
    position = np.vstack([home_xy, away_xy])
    players = np.full((n, 22, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    gt = np.full(n, X, dtype=np.int8)

    for start, stop in halves:
        t = start
        side = HOME
        velocity = np.zeros((22, 2))
        while t < stop:
            dead = rng.random() < 0.25
            length = int(rng.uniform(4, 9) * FPS) if dead else int(rng.uniform(5, 20) * FPS)
            end = min(stop, t + length)
            carrier = rng.choice(np.flatnonzero(teams == side))
            hold = 0
            flight = None
            for f in range(t, end):
                pace = 0.6 if dead else 3.0
                velocity = 0.9 * velocity + 0.1 * rng.normal(0, pace, (22, 2))
                position = np.clip(position + velocity / FPS, [1, 1], [PITCH[0] - 1, PITCH[1] - 1])
                players[f] = position
                if dead:
                    ball[f] = (-3.0, position[carrier, 1])  # out over the line
                    gt[f] = D
                    continue
                gt[f] = H if side == HOME else A
                if flight is not None:
                    origin, target, t0, duration = flight
                    share = (f - t0) / duration
                    if share >= 1.0:
                        carrier, flight, hold = target, None, 0
                    else:
                        ball[f] = (1 - share) * position[origin] + share * position[target]
                        continue
                ball[f] = position[carrier] + 0.3
                hold += 1
                if hold > rng.uniform(1.0, 2.5) * FPS:
                    mates = np.flatnonzero((teams == side) & (np.arange(22) != carrier))
                    target = rng.choice(mates)
                    distance = np.linalg.norm(position[target] - position[carrier])
                    flight = (carrier, target, f, max(int(distance / 15.0 * FPS), 3))
            t = end
            if not dead and rng.random() < 0.7:
                side = AWAY if side == HOME else HOME

    match = Match(name="synthetic", fps=FPS, pitch=PITCH, players_xy=players,
                  players_team=teams, ball_xy=ball, halves=halves)
    return match, gt
