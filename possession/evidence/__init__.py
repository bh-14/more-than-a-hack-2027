"""Evidence modules. Each exposes compute(match, features, params) -> Evidence.

    dead_ball  M1  stoppages and restarts
    control    M2  who controls the ball while the tracking sees it
    gaps       M3  what the ball probably did while the tracking lost it
    shape      M4  possession from the shape of both teams, for frames without a ball
    tagging    M5  Smart Tagging as a weak prior

A module that has nothing to say about a frame leaves its row at zero. Relative trust between
modules is not set here: the decoder weights each one by name from configs/params.toml.
"""

from . import control, dead_ball, gaps, shape, tagging

MODULES = {
    "dead_ball": dead_ball,
    "control": control,
    "gaps": gaps,
    "shape": shape,
    "tagging": tagging,
}
