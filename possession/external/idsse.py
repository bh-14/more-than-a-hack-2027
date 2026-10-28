"""The open DFL / IDSSE matches: seven Bundesliga games with corrected tracking at 25 fps.

Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data, 12, 195. CC BY 4.0 -- cite it in the README and the
technical document. Using it to fit the pipeline was confirmed as permitted by the organisers (issue #16).

Why this dataset: every ball frame carries BallStatus (alive/dead) and BallPossession (home/away),
which is exactly the label we predict, on other games.

Files come from figshare through its public API (the springernature.figshare.com links kloppy
uses sit behind a browser challenge). They are parsed with a line scanner rather than a full XML
parser -- each file is ~370 MB with one <Frame> per line -- and cached as .npz next to them.

    python -m possession.external.idsse            # download all seven, build the caches
"""

import os
import re
import sys
import urllib.request

import numpy as np

from ..contract import AWAY, HOME, A, D, H, Match, X

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("IDSSE_DIR") or os.path.join(ROOT, "external_data", "idsse")
URL = "https://api.figshare.com/v2/file/download/{}"
CACHE_VERSION = 1

# match id -> figshare file ids, as listed by kloppy (kloppy._providers.sportec.get_IDSSE_url).
FILES = {
    "J03WPY": {"meta": 51643487, "tracking": 51643526},
    "J03WN1": {"meta": 51643472, "tracking": 51643517},
    "J03WMX": {"meta": 51643475, "tracking": 51643514},
    "J03WOH": {"meta": 51643478, "tracking": 51643520},
    "J03WQQ": {"meta": 51643484, "tracking": 51643529},
    "J03WOY": {"meta": 51643481, "tracking": 51643523},
    "J03WR9": {"meta": 51643490, "tracking": 51643532},
}
MATCH_IDS = tuple(FILES)

FPS = 25
# NaN frames inserted between the halves, so nothing temporal runs across halftime.
HALFTIME_GAP = 250
HALF_ORDER = ("firstHalf", "secondHalf")

_FRAMESET = re.compile(r'GameSection="(\w+)".*?TeamId="([^"]+)".*?PersonId="([^"]+)"')
_FRAME = re.compile(r'N="(\d+)".*? X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="(-?[\d.]+)")?')
_BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
_PITCH = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')
_TEAM = re.compile(r'<Team TeamId="([^"]+)"[^>]*Role="(\w+)"')


def path(match_id, kind):
    return os.path.join(DATA_DIR, "{}_{}.xml".format(match_id, kind))


def download(match_id):
    """Fetch the meta and tracking files of one match, unless they are already there."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for kind, file_id in FILES[match_id].items():
        target = path(match_id, kind)
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            continue
        partial = target + ".part"
        urllib.request.urlretrieve(URL.format(file_id), partial)
        os.replace(partial, target)


def home_team(meta_path):
    with open(meta_path, encoding="utf-8") as handle:
        roles = dict(_TEAM.findall(handle.read()))
    home = [team for team, role in roles.items() if role == "home"]
    if len(home) != 1:
        raise ValueError("{}: expected one home team, found {}".format(meta_path, roles))
    return home[0]


def parse(match_id):
    """Scan one tracking file into per-object frame series. Returns a dict of raw arrays."""
    home = home_team(path(match_id, "meta"))
    objects = {}          # (section, person) -> lists
    teams = {}            # person -> team id
    pitch = (105.0, 68.0)
    current = None
    with open(path(match_id, "tracking"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                found = _FRAME.search(line)
                current["n"].append(int(found.group(1)))
                current["x"].append(float(found.group(2)))
                current["y"].append(float(found.group(3)))
                if current["ball"]:
                    current["z"].append(float(found.group(4) or "nan"))
                    possession, status = _BALL.search(line).groups()
                    current["possession"].append(int(possession))
                    current["status"].append(int(status))
            elif line.startswith("<FrameSet"):
                section, team, person = _FRAMESET.search(line).groups()
                ball = team == "BALL"
                current = {"n": [], "x": [], "y": [], "ball": ball, "team": team,
                           "z": [], "possession": [], "status": []}
                objects[(section, person)] = current
                if not ball:
                    teams[person] = team
            elif line.startswith("<PitchSize"):
                found = _PITCH.search(line)
                pitch = (float(found.group(1)), float(found.group(2)))
    return home, objects, teams, pitch


def build(match_id):
    """Match and ground-truth codes for one open game, from the raw files."""
    home, objects, teams, pitch = parse(match_id)
    people = sorted(teams)
    slot_of = {person: slot for slot, person in enumerate(people)}
    players_team = np.array([HOME if teams[p] == home else AWAY for p in people], dtype=np.int8)

    ball_of = {section: series for (section, _), series in objects.items() if series["ball"]}
    spans, offset = {}, 0
    for section in HALF_ORDER:
        first, last = min(ball_of[section]["n"]), max(ball_of[section]["n"])
        spans[section] = (first, offset)
        offset += last - first + 1 + (HALFTIME_GAP if section != HALF_ORDER[-1] else 0)
    n = offset

    players = np.full((n, len(people), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    ball_z = np.full(n, np.nan, dtype=np.float32)
    gt = np.full(n, X, dtype=np.int8)
    halves = []
    for (section, person), series in objects.items():
        first, start = spans[section]
        index = np.asarray(series["n"]) - first + start
        xy = np.column_stack([np.asarray(series["x"]) + pitch[0] / 2,
                              np.asarray(series["y"]) + pitch[1] / 2]).astype(np.float32)
        if series["ball"]:
            ball_xy[index] = xy
            ball_z[index] = series["z"]
            status = np.asarray(series["status"])
            possession = np.asarray(series["possession"])
            gt[index] = np.where(status == 0, D, np.where(possession == 1, H,
                                                          np.where(possession == 2, A, X)))
            halves.append((int(index.min()), int(index.max()) + 1))
        else:
            players[index, slot_of[person]] = xy

    match = Match(name=match_id, fps=FPS, pitch=pitch, players_xy=players,
                  players_team=players_team, ball_xy=ball_xy, halves=sorted(halves),
                  ball_z=ball_z)
    return match, gt


def cache_path(match_id):
    return os.path.join(DATA_DIR, "{}_v{}.npz".format(match_id, CACHE_VERSION))


def load(match_id, use_cache=True):
    """(Match, gt_codes) for one open game, downloading and caching on first use."""
    cache = cache_path(match_id)
    if use_cache and os.path.isfile(cache):
        stored = np.load(cache, allow_pickle=False)
        match = Match(name=match_id, fps=FPS, pitch=tuple(stored["pitch"].tolist()),
                      players_xy=stored["players_xy"], players_team=stored["players_team"],
                      ball_xy=stored["ball_xy"], ball_z=stored["ball_z"],
                      halves=[tuple(h) for h in stored["halves"].tolist()])
        return match, stored["gt"]
    download(match_id)
    match, gt = build(match_id)
    np.savez_compressed(cache, players_xy=match.players_xy, players_team=match.players_team,
                        ball_xy=match.ball_xy, ball_z=match.ball_z, gt=gt,
                        halves=np.array(match.halves), pitch=np.array(match.pitch))
    return match, gt


if __name__ == "__main__":
    for match_id in (sys.argv[1:] or MATCH_IDS):
        match, gt = load(match_id)
        graded = gt != X
        print("{}  frames {:>7}  slots {:>2}  alive {:5.1%}  ball seen {:5.1%}".format(
            match_id, match.n_frames, match.n_slots, np.mean(gt[graded] != D),
            np.mean(np.isfinite(match.ball_xy[graded, 0]))))
