"""Match + params -> one label code per frame.

    features  = geometry.compute(match)
    evidence  = [module.compute(match, features, params[name]) for each enabled module]
    total     = sum(weight[name] * evidence.scores)
    labels    = decode(total) inside each half, X everywhere else

X outside the halves costs nothing: those frames are never graded. Inside them the decoder always
commits to H, A or D, because the scorer fills an X with the previous label anyway (spellF1) or
counts it as a miss (macroF1) -- an explicit label is never worse for the ranking metric.
"""

import os
import tomllib

import numpy as np

from . import geometry
from .contract import X
from .decode import decode
from .evidence import MODULES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARAMS = os.path.join(ROOT, "configs", "params.toml")


def load_params(path=None):
    with open(path or DEFAULT_PARAMS, "rb") as handle:
        return tomllib.load(handle)


def evidence_for(match, params, features=None):
    """Every enabled module's Evidence, in the order of [weights]."""
    features = features or geometry.compute(match)
    out = []
    for name, weight in params["weights"].items():
        if weight == 0:
            continue
        out.append(MODULES[name].compute(match, features, params.get(name, {})))
    return out


def combine(evidence, params, n_frames):
    total = np.zeros((n_frames, 3), dtype=np.float64)
    for item in evidence:
        total += params["weights"][item.name] * item.scores
    return total


def label_match(match, params, evidence=None):
    """Label codes (H/A/D inside the halves, X outside) for the whole match."""
    evidence = evidence if evidence is not None else evidence_for(match, params)
    total = combine(evidence, params, match.n_frames)
    codes = np.full(match.n_frames, X, dtype=np.int8)
    for start, stop in match.halves:
        codes[start:stop] = decode(total[start:stop], params["decoder"])
    return codes
