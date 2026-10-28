# Technical document — draft

Structure required by SUBMISSION.md. Each section has an owner; keep it short ("a few pages is
plenty"). Exported to `<team_name>.pdf` or submitted as `.md` at the end.

## 1. Approach and why

Possession is a segmentation problem, not a per-frame classification problem: the ranking metric
(spellF1) matches whole possessions by temporal IoU, so every invented possession is a false
positive and every missed stoppage merges two real ones. The pipeline therefore has two layers:

1. **Evidence modules** turn tracking into per-frame scores for H, A and D (`possession/evidence/`).
2. **A semi-Markov decoder** (Viterbi over H/A/D with switching costs and minimum durations) turns
   the summed evidence into possessions, looking at the whole half at once
   (`possession/decode/viterbi.py`).

Alternatives considered: _fill in (e.g. per-frame classifier + smoothing, which is the baseline's
family; end-to-end sequence model)._

## 2. Assets used, and assets dropped

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (players + ball) | yes | all evidence modules |
| Smart Tagging | planned | M5, weak prior; weight set a priori |
| Homography | _tbd_ | |
| Tactical video | _tbd_ | |
| Broadcast ATD / video | _tbd_ | extra E1: alignment to recover ball positions |

## 3. Training data

No labels of any kind from the provided match are used, for training, validation or choosing
parameters.

| Dataset | Source | Labelled by | Amount | Used for |
|---|---|---|---|---|
| IDSSE / DFL open data | Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195, CC BY 4.0, figshare | DFL official data (BallStatus, BallPossession) | 7 Bundesliga matches, ~94 min graded each | 4 matches train the two logistic models (M1 dead ball, M4 shape; `scripts/train_models.py`); 3 matches choose every threshold and weight |

Labels from the open games are brought to the challenge's definition of possession with one a
priori rule (`possession/external/labels.py`): a spell shorter than 2 s that follows the other
side's spell is handed back, because a touch without control is not a change of possession
(CHALLENGE.md). The tracking is degraded with a noise model before every evaluation
(`possession/external/noise.py`) so it resembles the provided automatic tracking.

## 4. Possession rules implemented

- **Loose balls and challenges:** _owner of M2/M3._
- **Restarts:** _owner of M1._
- **Dead ball:** _owner of M1._

## 5. Limitations

_What fails and what we would fix with more time._

## 6. Reproduction

```bash
python -m pip install -r requirements-team.txt
python scripts/make_submission.py --team <team_name>
```

CPU only and no randomness: byte-identical output for the same data and commit. See SOLUTION.md.
