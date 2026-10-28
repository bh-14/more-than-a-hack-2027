"""Train the learned parts of M1 and M4 on the open games' train split. Writes models/*.json.

    python scripts/train_models.py

  models/dead_v1.json    P(dead | pitch and ball features)        -> M1 (evidence/dead_ball.py)
  models/shape_v1.json   P(home | alive, team-shape differences)  -> M4 (evidence/shape.py)

Only the train split (scripts/validate_external.py SPLITS["train"]) is used, degraded by the noise
model, with labels brought to the challenge's possession rule. The tune split stays unseen, so
validate_external.py on it is an honest test. The provided match is never read here.

Deterministic: fixed noise seeds, Newton fitting, sorted JSON. The weights are committed, so
reproducing the submission does not need this script; it is here to show how they were built.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import features as collective, geometry, models  # noqa: E402
from possession.contract import A, D, H, X  # noqa: E402
from possession.external import idsse, labels, noise  # noqa: E402
from possession.scoring import pm  # noqa: E402
from validate_external import MIN_CONTROL_S, SPLITS  # noqa: E402

# Params
STRIDE = 5          # use every 5th frame: neighbours are near-duplicates
NOISE_SEED = 100    # different from the evaluation seeds
L2 = 1.0


def rows(match_ids):
    shape_x, shape_y, dead_x, dead_y = [], [], [], []
    for index, match_id in enumerate(match_ids):
        match, gt = idsse.load(match_id)
        gt = labels.controlled_possession(gt, pm.valid_runs(gt), int(MIN_CONTROL_S * match.fps))
        noisy = noise.degrade(match, seed=NOISE_SEED + index)
        shape_names, shape, dead_names, dead = collective.compute(noisy, geometry.compute(noisy))
        pick = np.zeros(match.n_frames, dtype=bool)
        pick[::STRIDE] = True
        graded = pick & (gt != X)
        alive = graded & (gt != D)
        dead_x.append(dead[graded])
        dead_y.append(gt[graded] == D)
        shape_x.append(shape[alive])
        shape_y.append(gt[alive] == H)
        print("{}: {} graded rows, {} alive".format(match_id, graded.sum(), alive.sum()))
    return (shape_names, np.vstack(shape_x), np.concatenate(shape_y),
            dead_names, np.vstack(dead_x), np.concatenate(dead_y))


def accuracy(model, X, y):
    return float(np.mean((models.logit(model, X) > 0) == y))


def main():
    train = SPLITS["train"]
    shape_names, sx, sy, dead_names, dx, dy = rows(train)
    info = {"train_matches": list(train), "noise_seed": NOISE_SEED, "stride": STRIDE, "l2": L2,
            "label_rule": "controlled_possession, min_control_s={}".format(MIN_CONTROL_S)}

    dead = models.fit(dx, dy, l2=L2, intercept=True)
    print("dead : train accuracy {:.3f} (always-alive {:.3f})".format(
        accuracy(dead, dx, dy), 1 - dy.mean()))
    print(models.save(dead, "dead_v1", features=dead_names, **info))

    shape = models.fit(sx, sy, l2=L2, intercept=False)
    print("shape: train accuracy {:.3f} (home share {:.3f})".format(
        accuracy(shape, sx, sy), sy.mean()))
    print(models.save(shape, "shape_v1", features=shape_names, **info))
    for name, coef in sorted(zip(shape_names, shape["coef"]), key=lambda t: -abs(t[1])):
        print("  {:<22} {:+.3f}".format(name, coef))


if __name__ == "__main__":
    main()
