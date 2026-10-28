"""M5 -- Smart Tagging as a weak prior.

Not implemented yet: returns no opinion. The plan, for its owner:

  * Attacking phases (BUILD UP, PROGRESSION, CREATING CHANCES, BALL IN FINAL THIRD, ATTACKING
    TRANSITION) vote for the tagged side; DEFENDING IN ... phases vote for the other side.
  * SET PIECES mark a restart: dead before it, alive after it, and the Type tag says which kind.
  * Team is N/A on 36 instances, including every throw-in, so those vote for nothing.
  * The open datasets carry no Smart Tagging, so this weight cannot be fitted on other games. It
    is set a priori, kept small, and documented in docs/DECISIONS.md.

match.events holds the instances as loaded by io/bar_ath.py: dicts with code, start_frame,
end_frame (half-open) and tags.
"""

from ..contract import Evidence


def compute(match, features, params):
    return Evidence.empty("tagging", match.n_frames)
