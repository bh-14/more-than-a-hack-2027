"""Regenerate the submission from the provided data. The one command the organisers run.

    python scripts/make_submission.py --team <team_name>

Writes evaluation/results/<team_name>.csv, checks it with the official validator, and prints its
SHA-256. CPU only, no randomness: the same data and the same commit give the same bytes
(SUBMISSION.md, organisers' answer in issue #11).

The data directory is ./data or MTAH_DATA_DIR (games.py). The parsed tracking is cached under
<data dir>/cache/ on the first run; --no-cache re-parses it.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from possession import pipeline, submission  # noqa: E402
from possession.contract import LETTERS  # noqa: E402
from possession.io import bar_ath  # noqa: E402
from possession.scoring import pm  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; also the file name")
    parser.add_argument("--params", help="params TOML (default configs/params.toml)")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    match = bar_ath.load(use_cache=not args.no_cache)
    codes = pipeline.label_match(match, pipeline.load_params(args.params))

    os.makedirs(args.out, exist_ok=True)
    path = submission.write(os.path.join(args.out, "{}.csv".format(args.team)), codes)
    problem = submission.validate(path, games.source_frames("tactical"))
    if problem:
        sys.exit("INVALID: {}".format(problem))

    counts = {str(LETTERS[c]): int(np.sum(codes == c)) for c in range(4)}
    runs = [(start, stop) for start, stop in match.halves]
    spells = pm.possession_spells(codes, runs)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("labels   {}".format(counts))
    print("spells   {} predicted possessions over {} graded frames".format(
        len(spells), sum(stop - start for start, stop in runs)))
    print("sha256   {}".format(submission.sha256(path)))


if __name__ == "__main__":
    main()
