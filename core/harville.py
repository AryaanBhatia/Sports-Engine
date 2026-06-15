"""
harville.py -- Exotic probabilities from win probabilities.

The win market only tells you P(i finishes 1st). Exotic pools -- place, exacta,
trifecta, first-four -- need the joint distribution over finishing *orders*. The
Harville (1973) model gives it from the win probabilities alone, by assuming a
runner drawn from the remaining field with probability proportional to its
strength at each stage:

    P(a, b, c, ... in order) = v_a/V * v_b/(V - v_a) * v_c/(V - v_a - v_b) * ...

With strengths v_i = p_i (the win probabilities) this is exactly the
Plackett-Luce model, and it is internally consistent with the softmax win model
in ratings.py.

Harville is known to be over-confident about the favourite's place chances
(it implicitly assumes independent exponential running times). A standard fix is
a *discount exponent* lambda < 1 applied to the strengths used in the conditional
(2nd, 3rd, ...) draws, which flattens the field at the back end and tracks the
Henery/Stern models more closely. lambda = 1 recovers plain Harville.
"""
from __future__ import annotations

from itertools import permutations
from typing import List, Sequence


def order_prob(win_probs: Sequence[float], order: Sequence[int],
               lam: float = 1.0) -> float:
    """Probability of a specific finishing order (tuple of runner indices).

    ``lam`` is the discount exponent on the conditional draws.
    """
    strengths = [p ** lam for p in win_probs]
    # first place uses the untouched win prob; later places use discounted ones
    prob = win_probs[order[0]]
    remaining = sum(strengths) - strengths[order[0]]
    for idx in order[1:]:
        if remaining <= 0:
            return 0.0
        prob *= strengths[idx] / remaining
        remaining -= strengths[idx]
    return prob


def exacta_prob(win_probs: Sequence[float], i: int, j: int,
                lam: float = 1.0) -> float:
    """P(i finishes 1st AND j finishes 2nd)."""
    if i == j:
        return 0.0
    strengths = [p ** lam for p in win_probs]
    V = sum(strengths)
    return win_probs[i] * (strengths[j] / (V - strengths[i]))


def finish_position_probs(win_probs: Sequence[float], i: int, places: int = 3,
                          lam: float = 1.0) -> List[float]:
    """P(runner i finishes in each of positions 1..places).

    Returns a list of length ``places``; element k is P(i is (k+1)-th).
    Computed by summing Plackett-Luce probabilities over all ordered prefixes
    of the field that put i in that slot. Field sizes in racing are small
    enough (<= ~24) that the recursion is cheap for places <= 4.
    """
    n = len(win_probs)
    strengths = [p ** lam for p in win_probs]
    out = [0.0] * places

    def recurse(prefix_sum_strength: float, depth: int, used: int,
                path_prob: float, target_slot: int):
        # depth is 0-indexed position we are about to fill
        remaining = sum(s for k, s in enumerate(strengths)
                        if not (used >> k) & 1)
        for cand in range(n):
            if (used >> cand) & 1:
                continue
            if depth == 0:
                step = win_probs[cand]
            else:
                step = strengths[cand] / remaining if remaining > 0 else 0.0
            new_prob = path_prob * step
            if cand == i:
                out[target_slot] += new_prob if depth == target_slot else 0.0
            if depth < target_slot:
                recurse(0.0, depth + 1, used | (1 << cand), new_prob,
                        target_slot)

    for slot in range(places):
        recurse(0.0, 0, 0, 1.0, slot)
    return out


def place_prob(win_probs: Sequence[float], i: int, places: int = 3,
               lam: float = 1.0) -> float:
    """P(runner i finishes in the top ``places`` (i.e. 'places')."""
    return sum(finish_position_probs(win_probs, i, places, lam))


def full_order_distribution(win_probs: Sequence[float], lam: float = 1.0):
    """All finishing orders and their probabilities (only for tiny fields).

    Returns a dict mapping order-tuple -> probability. Enumerates n!
    permutations, so keep n small (<= 8 or so) -- intended for tests and demos.
    """
    n = len(win_probs)
    dist = {}
    for perm in permutations(range(n)):
        dist[perm] = order_prob(win_probs, perm, lam)
    return dist
