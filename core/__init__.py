"""Edge Engine -- a quantitative wagering research core.

A small, tested reference implementation of the mathematics a proprietary
betting syndicate uses to find and size an edge:

    devig      -- strip the bookmaker margin (multiplicative/additive/power/Shin)
    ratings    -- map proprietary ratings to win probabilities (multinomial logit)
    harville   -- exotic probabilities from win probabilities (Plackett-Luce)
    kelly      -- edge, EV and growth-optimal stake sizing
    parimutuel -- pool price-impact and game-theoretic optimal stake
"""
from . import devig, ratings, harville, kelly, parimutuel  # noqa: F401

__all__ = ["devig", "ratings", "harville", "kelly", "parimutuel"]
__version__ = "1.0.0"
