"""Load the provided match (tactical source) into a Match, with a local cache.

Parsing the raw ATD takes a minute or two, so the arrays are cached as an .npz inside the data
directory -- gitignored, and never published: frame-level tracking is derived data that the terms
forbid redistributing (RULES.md, "Data handling").

Raw ATD line (starter/baseline.py): "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed"
normalised 0-1, NaN where missing. frame_id counts from 1; the video, and so the submission, from 0.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from ..contract import AWAY, HOME, Match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import games  # noqa: E402
import load_smart_tagging  # noqa: E402

CACHE_VERSION = 1


def slot_teams(meta_path):
    """HOME/AWAY per track slot, in file order. The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team == team_ids[0] else AWAY for team in slots], dtype=np.int8)


def _xy(group):
    if not group or group[0] == "N":
        return np.nan, np.nan
    parts = group.split(",")
    return float(parts[0]), float(parts[1])


def parse_raw(raw_path, n_frames, n_slots, pitch):
    """(players_xy, ball_xy) in metres, origin at a corner, indexed by video frame."""
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                players[frame, slot] = _xy(group)
            ball[frame] = _xy(parts[2])
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def load(game=games.GAME, data_dir=None, use_cache=True):
    """The tactical source of the provided match as a Match."""
    meta_path, raw_path, tagging_path = games.require(
        ["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n_frames = games.source_frames("tactical", game)
    pitch = tuple(games.GAMES[game]["pitch"])
    teams = slot_teams(meta_path)

    cache_dir = os.path.join(data_dir or games.DATA_DIR, "cache")
    cache = os.path.join(cache_dir, "{}_tactical_v{}.npz".format(game, CACHE_VERSION))
    if use_cache and os.path.isfile(cache):
        stored = np.load(cache)
        players, ball = stored["players_xy"], stored["ball_xy"]
    else:
        players, ball = parse_raw(raw_path, n_frames, len(teams), pitch)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, players_xy=players, ball_xy=ball)

    halves = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    return Match(
        name=game,
        fps=games.GAMES[game]["fps"],
        pitch=pitch,
        players_xy=players,
        players_team=teams,
        ball_xy=ball,
        halves=halves,
        events=load_smart_tagging.load(game, data_dir),
    )
