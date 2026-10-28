"""M6 -- turn summed evidence into labels, one half at a time.

A three-state Viterbi over H, A and D in which every state is a chain of min_frames sub-states:
once entered, a state lasts at least that long. That is the simplest semi-Markov model, and it is
what spellF1 asks for -- every spurious one-second possession is a false positive, so the decoder
has to be told that possessions have a length, not just a label.

The decoder sees the whole half at once, so it places a hand-over where the evidence for the new
side *starts*, even when it only becomes convincing seconds later. Look-ahead is allowed: the task
is post-match (organisers, issue #15).

Costs are in the same log units as the evidence. They are chosen on the open datasets
(scripts/validate_external.py), never on the provided game.
"""

import numpy as np

from ..contract import A, D, H, N_STATES


def transition_costs(params):
    """3x3 cost of leaving row state for column state. The diagonal is never used."""
    cost = np.zeros((N_STATES, N_STATES))
    cost[H, A] = cost[A, H] = params["switch_cost"]
    cost[H, D] = cost[A, D] = params["to_dead_cost"]
    cost[D, H] = cost[D, A] = params["from_dead_cost"]
    np.fill_diagonal(cost, np.inf)
    return cost


def decode(scores, params):
    """Most likely H/A/D code per frame for one half. scores is (n_frames, 3)."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    durations = np.array([params["min_possession_frames"], params["min_possession_frames"],
                          params["min_dead_frames"]], dtype=int)
    durations = np.maximum(durations, 1)
    first = np.concatenate(([0], np.cumsum(durations)[:-1]))
    last = first + durations - 1
    n_sub = int(durations.sum())
    state_of = np.repeat(np.arange(N_STATES), durations)
    inner = np.setdiff1d(np.arange(n_sub), first)  # sub-states reached from the one before
    cost = transition_costs(params)
    columns = np.arange(N_STATES)

    emissions = np.asarray(scores, dtype=np.float64)
    # A half may open mid-state, so the first frame can sit at the start or the end of a chain.
    delta = np.full(n_sub, -np.inf)
    delta[first] = emissions[0, state_of[first]]
    delta[last] = emissions[0, state_of[last]]
    back = np.zeros((n, n_sub), dtype=np.int16)

    for t in range(1, n):
        best = np.full(n_sub, -np.inf)
        arg = np.zeros(n_sub, dtype=np.int16)

        best[inner] = delta[inner - 1]
        arg[inner] = inner - 1

        stay = delta[last]
        keep = stay > best[last]
        best[last] = np.where(keep, stay, best[last])
        arg[last] = np.where(keep, last, arg[last])

        enter = delta[last][:, None] - cost
        source = np.argmax(enter, axis=0)
        value = enter[source, columns]
        take = value > best[first]
        best[first] = np.where(take, value, best[first])
        arg[first] = np.where(take, last[source], arg[first])

        delta = best + emissions[t, state_of]
        back[t] = arg

    path = np.empty(n, dtype=np.int16)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return state_of[path].astype(np.int8)
