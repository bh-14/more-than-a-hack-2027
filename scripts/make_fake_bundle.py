"""Write an open game in the provided bundle's file formats, for a full rehearsal without the data.

    python scripts/make_fake_bundle.py                       # J03WPY -> external_data/fake_bundle
    $env:MTAH_DATA_DIR = "external_data/fake_bundle"         # PowerShell (export ... in bash)
    python scripts/make_submission.py --team rehearsal
    python evaluation/run_evaluation.py --gt external_data/fake_bundle/fake_gt.csv

The open game's halves are placed on the provided match's frame layout (199,319 frames, the same
graded windows), its tracking written as ATD text + metadata XML, a one-instance Smart Tagging
file is added, and its labels (brought to the challenge's possession rule) become fake_gt.csv.
So every step of the real run -- parsing, cache, pipeline, CSV, validator, official scorer --
runs on real football, and the score means something.

The tracking is written clean; make_submission.py does not add noise, so expect a higher score
than on the real ATD. The noise model is for validate_external.py.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import games  # noqa: E402
from possession.contract import LETTERS, X  # noqa: E402
from possession.external import idsse, labels  # noqa: E402
from possession.scoring import pm  # noqa: E402
from validate_external import MIN_CONTROL_S  # noqa: E402

SLOTS = 46


def group(x, y):
    return "NaN,NaN,NaN" if np.isnan(x) else "{:.5f},{:.5f},0.00".format(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", default="J03WPY", choices=idsse.MATCH_IDS)
    parser.add_argument("--out", default=os.path.join(ROOT, "external_data", "fake_bundle"))
    args = parser.parse_args()

    match, gt = idsse.load(args.match)
    gt = labels.controlled_possession(gt, pm.valid_runs(gt), int(MIN_CONTROL_S * match.fps))
    n = games.source_frames("tactical")
    length, width = games.GAMES[games.GAME]["pitch"]
    players = np.full((n, SLOTS, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    fake_gt = np.full(n, X, dtype=np.int8)
    used = min(match.n_slots, SLOTS)
    team = np.arange(SLOTS) % 2
    team[:used] = match.players_team[:used]

    for (start, stop), (_, first, last) in zip(match.halves, games.half_video_blocks(games.GAME)):
        count = min(stop - start, last - first + 1)
        players[first:first + count, :used] = match.players_xy[start:start + count, :used]
        ball[first:first + count] = match.ball_xy[start:start + count]
        fake_gt[first:first + count] = gt[start:start + count]
        # Graded frames the open half does not reach count as dead ball, as halftime run-in would.
        fake_gt[first + count:last + 1] = 2
    players[..., 0] /= length
    players[..., 1] /= width
    ball[:, 0] /= length
    ball[:, 1] /= width

    folder = os.path.join(args.out, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, games.SOURCES["tactical"]["stem"])
    with open(stem + "_atd.xml", "w", newline="\n") as handle:
        handle.write('<?xml version="1.0"?>\n<Root><Teams><Team id="T1"><Name>Home</Name></Team>'
                     '<Team id="T2"><Name>Away</Name></Team></Teams><Players>\n')
        for slot in range(SLOTS):
            handle.write('<Player id="P{}" teamId="T{}"/>\n'.format(slot + 1, team[slot] + 1))
        handle.write("</Players></Root>\n")
    with open(stem + "_atd.txt", "w", newline="\n") as handle:
        for frame in range(n):
            body = ";".join(group(*players[frame, slot]) for slot in range(SLOTS))
            handle.write("{}:{}:{}\n".format(frame + 1, body, group(*ball[frame])))
    with open(stem + "_smart_tagging.xml", "w", newline="\n") as handle:
        handle.write("<file><ALL_INSTANCES><instance><ID>1</ID><start>447.0</start>"
                     "<end>452.0</end><code>SET PIECES</code><label><group>Type</group>"
                     "<text>Kick Off (Start)</text></label><label><group>Half</group>"
                     "<text>1st Half</text></label></instance></ALL_INSTANCES></file>\n")
    gt_path = os.path.join(args.out, "fake_gt.csv")
    with open(gt_path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, LETTERS[c]) for f, c in enumerate(fake_gt))
    print("bundle   {}".format(folder))
    print("truth    {}".format(gt_path))


if __name__ == "__main__":
    main()
