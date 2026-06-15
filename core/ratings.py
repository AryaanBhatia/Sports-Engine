"""
ratings.py -- Turning a proprietary rating into win probabilities.

The edge in a syndicate does not come from reading the market; it comes from
having an *independent* opinion that is better than the market's. That opinion
usually starts as a per-runner ability rating r_i -- a speed figure, an Elo, a
regression output, the score of a gradient-boosted model, whatever. To bet it
we need to map ratings to a probability distribution over the field.

The natural map is the multinomial-logit / Luce choice model (a softmax):

    p_i = exp(beta * r_i) / sum_j exp(beta * r_j)

beta is a *consensus* (inverse-temperature) parameter. beta -> 0 makes the field
uniform (no opinion); large beta concentrates all probability on the top-rated
runner (maximum conviction). It is the same object as the strength vector in a
Plackett-Luce model, which is what makes the exotics in harville.py consistent
with the win market here.
"""
from __future__ import annotations

import math
from typing import List, Sequence


def softmax_probs(ratings: Sequence[float], beta: float = 1.0) -> List[float]:
    """Win probabilities from ratings via tempered softmax (Luce choice)."""
    m = max(ratings)  # subtract max for numerical stability
    exps = [math.exp(beta * (r - m)) for r in ratings]
    s = sum(exps)
    return [e / s for e in exps]


def blend(model_probs: Sequence[float], market_probs: Sequence[float],
          w: float = 0.5) -> List[float]:
    """Geometric blend of model and market opinions, renormalised.

    Pure model opinions are noisy; the market is an informative prior. A
    log-space (geometric) blend with weight ``w`` on the model is a principled
    way to shrink the model toward the crowd, which is exactly what a desk does
    before it fires a bet.
    """
    raw = [(m ** w) * (q ** (1.0 - w))
           for m, q in zip(model_probs, market_probs)]
    s = sum(raw)
    return [x / s for x in raw]
