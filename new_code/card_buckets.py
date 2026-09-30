"""
Board-relative postflop card buckets (the card part of v5 infoset keys).

A hand's bucket depends on how it plays on this board, not on its name:
mid pair on K-8-2 rainbow and mid pair on J-T-9 two-tone land in different
buckets. Two measures, both against a uniformly random opponent hand:

  strength  share of opponent hands the hand beats right now (ties count half),
            exact over every opponent hand
  equity    chance to win at showdown after the remaining board cards,
            by simulation

Flop and turn buckets combine an equity level with a potential class from
equity - strength: "drawing" hands (outs to improve) gain equity from the
cards to come, "vulnerable" made hands lose it. On the river no cards are
left, so buckets are strength levels alone.

The simulation is seeded from the cards themselves, so the same hand and
board always get the same bucket in every process and after every restart.
Buckets are cached on a suit-canonical form of the cards.
"""

import itertools
import random
import zlib
from bisect import bisect_right
from functools import lru_cache

import eval7

_RANKS = "23456789TJQKA"
_CARDS = {r + s: eval7.Card(r + s) for r in _RANKS for s in "cdhs"}

EQUITY_SAMPLES = 600

# Bucket boundaries, fitted on 8,000 random deals per street so each level
# holds a similar share of hands (tools/analysis/fit_card_buckets.py, seed 0).
# Changing them changes every v5 key, so trained v5 tables depend on them.
EQUITY_EDGES = {  # 10 equity levels
    3: (0.2458, 0.3117, 0.3675, 0.4258, 0.4833, 0.545, 0.6067, 0.6983, 0.8033),
    4: (0.195, 0.27, 0.3381, 0.4083, 0.4829, 0.5575, 0.6453, 0.7483, 0.8468),
}
POTENTIAL_EDGES = {  # vulnerable (bottom 25%), neutral, drawing (top 25%)
    3: (-0.1014, 0.087),
    4: (-0.0624, 0.0465),
}
RIVER_STRENGTH_EDGES = (  # 15 strength levels
    0.0614, 0.1217, 0.1823, 0.251, 0.329, 0.396, 0.453, 0.5273,
    0.5965, 0.6727, 0.7343, 0.8025, 0.8758, 0.9384,
)


def hand_strength(hole, board):
    """Share of opponent hands that `hole` beats on `board` now (ties = 1/2)."""
    dead = set(hole) | set(board)
    rest = [card for name, card in _CARDS.items() if name not in dead]
    board_cards = [_CARDS[c] for c in board]
    mine = eval7.evaluate([_CARDS[c] for c in hole] + board_cards)
    score = 0.0
    total = 0
    for a, b in itertools.combinations(rest, 2):
        theirs = eval7.evaluate([a, b] + board_cards)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        total += 1
    return score / total


def equity(hole, board, samples=EQUITY_SAMPLES, rng=None):
    """Chance `hole` wins at showdown vs a random hand, by simulation."""
    rng = rng or random.Random(0)
    dead = set(hole) | set(board)
    rest = [card for name, card in _CARDS.items() if name not in dead]
    hole_cards = [_CARDS[c] for c in hole]
    board_cards = [_CARDS[c] for c in board]
    to_come = 5 - len(board)
    score = 0.0
    for _ in range(samples):
        drawn = rng.sample(rest, 2 + to_come)
        full = board_cards + drawn[2:]
        mine = eval7.evaluate(hole_cards + full)
        theirs = eval7.evaluate(drawn[:2] + full)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return score / samples


_SUIT_PERMUTATIONS = [dict(zip("cdhs", p)) for p in itertools.permutations("cdhs")]


def _sorted_cards(cards):
    return tuple(sorted(cards, key=lambda c: (_RANKS.index(c[0]), c[1]), reverse=True))


def canonical(hole, board):
    """Suit-isomorphic form, so equivalent deals share one key: of the 24 ways
    to rename suits, the one giving the smallest (hole, board), each sorted."""
    return min(
        (
            _sorted_cards(c[0] + perm[c[1]] for c in hole),
            _sorted_cards(c[0] + perm[c[1]] for c in board),
        )
        for perm in _SUIT_PERMUTATIONS
    )


def measures(hole, board):
    """(strength, equity) for a canonical deal, deterministic per deal."""
    seed = zlib.crc32(("".join(hole) + "|" + "".join(board)).encode())
    strength = hand_strength(hole, board)
    if len(board) == 5:
        return strength, strength
    return strength, equity(hole, board, rng=random.Random(seed))


@lru_cache(maxsize=1_000_000)
def _bucket_canonical(hole, board):
    strength, eq = measures(hole, board)
    street = len(board)
    if street == 5:
        return f"[R{bisect_right(RIVER_STRENGTH_EDGES, strength)}]"
    level = bisect_right(EQUITY_EDGES[street], eq)
    potential = bisect_right(POTENTIAL_EDGES[street], eq - strength)
    return f"[{'F' if street == 3 else 'T'}{level}p{potential}]"


@lru_cache(maxsize=200_000)
def card_bucket(hole, board):
    """Bucket key such as [F7p2] (flop, equity level 7, drawing) or [R12].

    hole and board are tuples of card names like ("Ah", "Kd"). A player's
    cards stay fixed while training explores their actions, so this outer
    cache on the raw cards saves the canonical-form work at repeat nodes."""
    return _bucket_canonical(*canonical(hole, board))
