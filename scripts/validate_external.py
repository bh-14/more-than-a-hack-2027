"""Score the pipeline on the open games, with the noise model applied -- the team's own leaderboard.

    python scripts/validate_external.py                  # tuning split, noisy, vs baseline
    python scripts/validate_external.py --split all --bootstrap 200
    python scripts/validate_external.py --set decoder.switch_cost=15 --set weights.gaps=0

Every number in configs/params.toml is chosen with this script, on these games. Nothing is chosen
on the provided match. Write the run you based a change on into docs/DECISIONS.md.

Labels are brought to the challenge's definition first (external/labels.py), and the official
baseline is re-run on the same degraded tracking, so "beats the floor" means the same thing here
as on the board.
"""

import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import baselines, pipeline  # noqa: E402
from possession.external import idsse, labels, noise  # noqa: E402
from possession.scoring import pm, score  # noqa: E402

# Params
# Four games to train models on, three to choose thresholds on. Fixed, so a result can be repeated.
SPLITS = {
    "train": ("J03WPY", "J03WN1", "J03WMX", "J03WOH"),
    "tune": ("J03WQQ", "J03WOY", "J03WR9"),
}
SPLITS["all"] = SPLITS["train"] + SPLITS["tune"]
MIN_CONTROL_S = 2.0  # external/labels.py: prior from CHALLENGE.md, not fitted
COLUMNS = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells")


def override(params, assignments):
    """Apply 'section.key=value' strings to a copy of params, keeping each value's type."""
    params = copy.deepcopy(params)
    for item in assignments:
        key, text = item.split("=", 1)
        section, name = key.split(".", 1)
        current = params.setdefault(section, {}).get(name, 0.0)
        if isinstance(current, bool):
            value = text.lower() == "true"
        elif isinstance(current, int):
            value = int(float(text))
        else:
            value = float(text)
        params[section][name] = value
    return params


def evaluate_match(match_id, params, use_noise=True, seed=0, bootstrap=0):
    match, gt = idsse.load(match_id)
    gt = labels.controlled_possession(gt, pm.valid_runs(gt), int(MIN_CONTROL_S * match.fps))
    if use_noise:
        match = noise.degrade(match, seed=seed)
    ours = score(gt, pipeline.label_match(match, params), bootstrap)
    floor = score(gt, baselines.nearest_player(match, 50))
    return ours, floor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=sorted(SPLITS), default="tune")
    parser.add_argument("--params", help="params TOML (default configs/params.toml)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--clean", action="store_true", help="skip the noise model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    params = override(pipeline.load_params(args.params), args.set)
    rows = []
    print("{:<8} | {:>7} {:>7} {:>5} {:>5} | baseline_smooth50 {:>7} {:>7} {:>5}".format(
        "match", "spellF1", "macroF1", "gt", "pred", "spellF1", "macroF1", "pred"))
    for index, match_id in enumerate(SPLITS[args.split]):
        ours, floor = evaluate_match(match_id, params, not args.clean, args.seed + index,
                                     args.bootstrap)
        rows.append((ours, floor))
        extra = "  [{:.3f}, {:.3f}]".format(ours["spell_low"], ours["spell_high"]) \
            if args.bootstrap else ""
        print("{:<8} | {:7.4f} {:7.4f} {:5d} {:5d} | {:>25.4f} {:7.4f} {:5d}{}".format(
            match_id, ours["spell_f1_mean"], ours["macro_f1"], ours["n_gt_spells"],
            ours["n_pred_spells"], floor["spell_f1_mean"], floor["macro_f1"],
            floor["n_pred_spells"], extra))
    mean = {key: np.mean([r[0][key] for r in rows]) for key in ("spell_f1_mean", "macro_f1")}
    base = {key: np.mean([r[1][key] for r in rows]) for key in ("spell_f1_mean", "macro_f1")}
    print("{:<8} | {:7.4f} {:7.4f} {:>11} | {:>25.4f} {:7.4f}".format(
        "mean", mean["spell_f1_mean"], mean["macro_f1"], "", base["spell_f1_mean"],
        base["macro_f1"]))


if __name__ == "__main__":
    main()
