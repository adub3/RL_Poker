"""
Board-relative postflop card buckets (the card part of v5 infoset keys).

A hand's bucket depends on how it plays on this board, not on its name:
mid pair on K-8-2 rainbow and mid pair on J-T-9 two-tone land in different
buckets, and so do second pair and a set, which both crush a random hand.

Features, all for this hand on this board:

  equity         chance to win at showdown against a random opponent hand
                 after the remaining board cards (simulated)
  strong equity  the same against a strong range: opponent hands whose made
                 hand right now is in the top 40% on this board. This spreads
                 out the strong end, where strategy differs most.
  potential      equity minus current strength (the share of opponent hands
                 beaten right now, exact): positive for draws with outs,
                 negative for made hands the next cards can beat

On the river no cards are left: the features are current strength against a
random hand and against the strong range, both exact.

Hands are grouped by k-means on these features (flop and turn 150 buckets,
river 100), fitted once on random deals by tools/analysis/fit_card_buckets.py
and stored in card_bucket_centroids.json. A bucket's number is its rank by
strong equity, so higher numbers are stronger hands. Changing the centroids
changes every v5 key, so trained v5 tables depend on that file.

The simulation is seeded from the cards themselves, so the same hand and
board always get the same bucket in every process and after every restart.
Buckets are cached on a suit-canonical form of the cards.
"""

import hashlib
import itertools
import json
import warnings
import zlib
from functools import lru_cache
from pathlib import Path

import eval7
import numpy as np

_RANKS = "23456789TJQKA"
_CARDS = {r + s: eval7.Card(r + s) for r in _RANKS for s in "cdhs"}

EQUITY_SAMPLES = 500
STRONG_RANGE_SHARE = 0.40
CENTROIDS_PATH = Path(__file__).with_name("card_bucket_centroids.json")
# Every flop bucket precomputed (tools/analysis/build_flop_bucket_table.py).
FLOP_TABLE_PATH = Path(__file__).with_name("flop_bucket_table.npz")
_CARD_INDEX = {name: i for i, name in enumerate(_CARDS)}
_STREET_LETTER = {3: "F", 4: "T", 5: "R"}


@lru_cache(maxsize=4096)
def _board_hands(board):
    """Every two-card hand on `board`, shared by all hole cards on that board:
    (cards not on the board, hands as index pairs into them in
    itertools.combinations order, each hand's made-hand value right now)."""
    left = [name for name in _CARDS if name not in board]
    board_cards = [_CARDS[c] for c in board]
    pairs = np.array(list(itertools.combinations(range(len(left)), 2)))
    values = np.array([eval7.evaluate([_CARDS[left[i]], _CARDS[left[j]]] + board_cards)
                       for i, j in pairs.tolist()])
    return left, pairs, values


def features(hole, board, rng):
    """Feature vector for a deal: (equity, strong equity, potential) on the
    flop and turn, (strength, strong strength) on the river."""
    dead = set(hole) | set(board)
    rest = [name for name in _CARDS if name not in dead]
    board_cards = [_CARDS[c] for c in board]
    hole_cards = [_CARDS[c] for c in hole]

    # Every opponent hand's made hand right now: current strength, and the
    # strong range as the top share of those values. Opponent hands are the
    # board's hands without our cards, in itertools.combinations(rest, 2)
    # order, with indices shifted from the board's card list into `rest`.
    left, board_pairs, board_values = _board_hands(tuple(board))
    h1, h2 = sorted(left.index(card) for card in hole)
    keep = ~((board_pairs == h1) | (board_pairs == h2)).any(axis=1)
    values = board_values[keep]
    pairs = board_pairs[keep]
    pairs = pairs - (pairs > h1) - (pairs > h2)
    mine_now = eval7.evaluate(hole_cards + board_cards)
    strength = float(np.mean((mine_now > values) + 0.5 * (mine_now == values)))
    cutoff = np.quantile(values, 1 - STRONG_RANGE_SHARE)
    strong = np.flatnonzero(values >= cutoff)

    if len(board) == 5:
        strong_values = values[strong]
        strong_strength = float(np.mean((mine_now > strong_values) + 0.5 * (mine_now == strong_values)))
        return (strength, strong_strength)

    # One shared simulation: each sample deals the rest of the board and one
    # random and one strong opponent hand that don't collide with it. The
    # random draws are made up front with numpy; the loop only evaluates.
    to_come = 5 - len(board)
    rest_cards = [_CARDS[c] for c in rest]
    randoms = pairs[rng.integers(len(pairs), size=EQUITY_SAMPLES)]
    strongs = pairs[strong[rng.integers(len(strong), size=EQUITY_SAMPLES)]]
    # A random card order per sample; the runout is its first cards that
    # neither opponent holds (at most 4 are skipped).
    orders = np.argsort(rng.random((EQUITY_SAMPLES, len(rest))), axis=1)[:, : to_come + 4]
    score = strong_score = 0.0
    for (a, b), (s1, s2), order in zip(randoms.tolist(), strongs.tolist(), orders.tolist()):
        taken = (a, b, s1, s2)
        runout = [c for c in order if c not in taken][:to_come]
        full = board_cards + [rest_cards[c] for c in runout]
        mine = eval7.evaluate(hole_cards + full)
        theirs = eval7.evaluate([rest_cards[a], rest_cards[b]] + full)
        strong_theirs = eval7.evaluate([rest_cards[s1], rest_cards[s2]] + full)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        strong_score += 1.0 if mine > strong_theirs else 0.5 if mine == strong_theirs else 0.0
    equity = score / EQUITY_SAMPLES
    return (equity, strong_score / EQUITY_SAMPLES, equity - strength)


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


def canonical_board(board):
    """One representative per suit-symmetry class of boards."""
    return min(_sorted_cards(c[0] + perm[c[1]] for c in board) for perm in _SUIT_PERMUTATIONS)


def encode_deal(hole, board):
    """A canonical deal as one integer (its cards in base 52)."""
    key = 0
    for card in (*hole, *board):
        key = key * 52 + _CARD_INDEX[card]
    return key


def centroids_hash():
    return hashlib.sha256(CENTROIDS_PATH.read_bytes()).hexdigest()


def canonical_features(hole, board):
    """Features for a canonical deal, seeded from its cards (deterministic)."""
    seed = zlib.crc32(("".join(hole) + "|" + "".join(board)).encode())
    return features(hole, board, np.random.default_rng(seed))


@lru_cache(maxsize=None)
def _centroids():
    data = json.loads(CENTROIDS_PATH.read_text())
    return {int(street): np.array(rows) for street, rows in data["centroids"].items()}


def bucket_index(hole, board):
    """Bucket number of a canonical deal, computed from its features."""
    centroids = _centroids()[len(board)]
    point = np.array(canonical_features(hole, board))
    return int(np.argmin(((centroids - point) ** 2).sum(axis=1)))


@lru_cache(maxsize=None)
def _flop_table():
    """(sorted deal keys, buckets) if a table built from the current centroids
    exists, else None (flop buckets are then computed on the fly)."""
    if not FLOP_TABLE_PATH.exists():
        return None
    data = np.load(FLOP_TABLE_PATH)
    if str(data["centroids_hash"]) != centroids_hash():
        warnings.warn(f"{FLOP_TABLE_PATH.name} was built from other centroids; ignoring it")
        return None
    return data["keys"], data["buckets"]


@lru_cache(maxsize=1_000_000)
def _bucket_canonical(hole, board):
    street = len(board)
    table = _flop_table() if street == 3 else None
    if table is not None:
        keys, buckets = table
        key = encode_deal(hole, board)
        position = int(np.searchsorted(keys, key))
        if position < len(keys) and keys[position] == key:
            return f"[F{int(buckets[position])}]"
    return f"[{_STREET_LETTER[street]}{bucket_index(hole, board)}]"


@lru_cache(maxsize=200_000)
def card_bucket(hole, board):
    """Bucket key such as [F87] (flop bucket 87 of 150) or [R12].

    hole and board are tuples of card names like ("Ah", "Kd"). A player's
    cards stay fixed while training explores their actions, so this outer
    cache on the raw cards saves the canonical-form work at repeat nodes."""
    return _bucket_canonical(*canonical(hole, board))
