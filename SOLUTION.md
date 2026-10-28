# Team solution — quick start

Everything the organisers' files describe (README.md, CHALLENGE.md, RULES.md…) still applies. This
file covers only what the team added on top: the `possession/` package, `scripts/` and `configs/`.

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-team.txt   # Windows
.venv/bin/python -m pip install -r requirements-team.txt       # macOS / Linux
python -m pytest -q                                             # synthetic matches only
```

Put the match bundle in `./data` or anywhere else with `MTAH_DATA_DIR` pointing to it (see
`games.py`). Keeping it **outside** the repository is safer. `data/` is gitignored either way.

## Reproduce the submission

```bash
python scripts/make_submission.py --team <team_name>
```

This writes `evaluation/results/<team_name>.csv`, runs the official validator on it and prints the
SHA-256. It runs on CPU only and uses no randomness, so the same data and the same commit always
give the same bytes. The first run parses the raw tracking and caches it under `<data dir>/cache/`.

## Rehearse without the match data

```bash
python scripts/make_fake_bundle.py                 # an open game in the provided file formats
export MTAH_DATA_DIR=external_data/fake_bundle     # PowerShell: $env:MTAH_DATA_DIR = "..."
python scripts/make_submission.py --team rehearsal
python evaluation/run_evaluation.py --gt external_data/fake_bundle/fake_gt.csv
```

Ignore the `baseline_*` rows there: they were computed on the real match, not on the fake one.

## Choose parameters (only on other games)

```bash
python -m possession.external.idsse                     # downloads 7 open games (~2.6 GB), once
python scripts/validate_external.py                     # tuning split, noise applied, vs baseline
python scripts/sweep.py --grid decoder.switch_cost=6,12,24
```

Every value in `configs/params.toml` comes from these scripts or is set a priori. **None** is
chosen on the provided match, including the five public minutes (RULES.md and the organisers'
answer in issue #10). Log each change in `docs/DECISIONS.md`.

## Layout

| Path | What it holds |
|---|---|
| `possession/contract.py` | `Match` (tracking in metres) and `Evidence` (per-frame scores for H/A/D) |
| `possession/geometry.py` | distances, speeds and counts shared by every module |
| `possession/evidence/` | M1 `dead_ball`, M2 `control`, M3 `gaps`, M4 `shape`, M5 `tagging` |
| `possession/decode/viterbi.py` | M6: semi-Markov Viterbi with switching costs and minimum durations |
| `possession/io/bar_ath.py` | loader for the provided match (with a cache) |
| `possession/external/` | open games (`idsse.py`), label conversion (`labels.py`), ATD noise model (`noise.py`) |
| `possession/pipeline.py` | `Match` + params → one code per frame |
| `possession/features.py`, `possession/models.py` | collective features and the logistic models of M1 and M4 |
| `models/` | committed weights, trained on the open games only (`scripts/train_models.py`) |
| `configs/params.toml` | every threshold and weight |
| `scripts/` | `make_submission.py`, `validate_external.py`, `sweep.py`, `train_models.py`, `make_fake_bundle.py` |
| `docs/` | team plan (`PLANO.md`), technical document (`TECHNICAL.md`), decision log (`DECISIONS.md`) |

## Data and licences

- Match data: under the terms accepted at enrolment. Never committed, never redistributed.
- Open games: IDSSE / DFL open data. Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An
  integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12, 195.
  CC BY 4.0.
