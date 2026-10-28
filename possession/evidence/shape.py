"""M4 -- possession from the shape of both teams, for frames where the ball is not seen.

Not implemented yet: returns no opinion. The plan, for its owner:

  * Collective features per frame, per side: centroid along the attacking direction, length,
    width, compactness, height of the last line, spread of speeds, how many players are behind
    the last known ball position, pressure (opponents within a radius of that position).
  * The attacking direction of each side per half comes from the data (the side whose deepest
    player sits near a goal line defends that goal), never from a constant.
  * A logistic regression or gradient-boosted model trained on the open datasets with the noise
    model applied (external/), predicting P(home | alive) and P(dead). Weights are committed under
    models/ together with the script that trains them; inference runs on CPU.
  * Output: log-odds into the H / A columns, scaled by the model's confidence.
"""

from ..contract import Evidence


def compute(match, features, params):
    return Evidence.empty("shape", match.n_frames)
