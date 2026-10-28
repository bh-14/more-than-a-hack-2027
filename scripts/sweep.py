"""Grid search over params on the open games (noise applied). Prints combinations by spellF1.

    python scripts/sweep.py --grid decoder.switch_cost=6,12,24 --grid decoder.min_possession_frames=25,50

Matches are loaded and degraded once and reused for every combination. Use the tuning split to
choose, then confirm the winner once on --split all; a gap under ~0.04 spellF1 is noise.
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import pipeline  # noqa: E402
from possession.external import idsse, labels, noise  # noqa: E402
from possession.scoring import pm, score  # noqa: E402
from validate_external import MIN_CONTROL_S, SPLITS, override  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=sorted(SPLITS), default="tune")
    parser.add_argument("--grid", action="append", required=True, metavar="SECTION.KEY=V1,V2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    games = []
    for index, match_id in enumerate(SPLITS[args.split]):
        match, gt = idsse.load(match_id)
        gt = labels.controlled_possession(gt, pm.valid_runs(gt), int(MIN_CONTROL_S * match.fps))
        games.append((noise.degrade(match, seed=args.seed + index), gt))

    keys = [item.split("=", 1)[0] for item in args.grid]
    values = [item.split("=", 1)[1].split(",") for item in args.grid]
    base = pipeline.load_params()
    rows = []
    for combo in itertools.product(*values):
        params = override(base, ["{}={}".format(k, v) for k, v in zip(keys, combo)])
        results = [score(gt, pipeline.label_match(match, params)) for match, gt in games]
        rows.append((np.mean([r["spell_f1_mean"] for r in results]),
                     np.mean([r["macro_f1"] for r in results]),
                     np.mean([r["n_pred_spells"] - r["n_gt_spells"] for r in results]), combo))
        print("spellF1 {:.4f}  macroF1 {:.4f}  spells {:+6.1f}  {}".format(
            rows[-1][0], rows[-1][1], rows[-1][2], dict(zip(keys, combo))), flush=True)

    print("\nbest first:")
    for spell, macro, count, combo in sorted(rows, key=lambda r: -r[0])[:10]:
        print("spellF1 {:.4f}  macroF1 {:.4f}  spells {:+6.1f}  {}".format(
            spell, macro, count, dict(zip(keys, combo))))


if __name__ == "__main__":
    main()
