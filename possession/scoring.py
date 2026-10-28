"""The official scorer, imported from evaluation/ rather than copied, so it can never drift."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

HEADLINE = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "alive_dead_f1",
            "team_acc_gt_alive")


def score(gt_codes, pred_codes, bootstrap_n=0):
    """pm.evaluate, reduced to the figures worth printing in a table."""
    result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    out = {key: result[key] for key in HEADLINE}
    if bootstrap_n:
        out["spell_low"] = result["spell_bootstrap_low"]
        out["spell_high"] = result["spell_bootstrap_high"]
    return out
