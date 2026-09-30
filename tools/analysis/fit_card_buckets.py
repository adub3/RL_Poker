"""
Fit the bucket boundaries in new_code/card_buckets.py from random deals.

Equity and river-strength levels are quantiles, so each level holds a similar
share of hands. Potential (equity - strength) splits into the bottom 25%
(vulnerable made hands), the middle 50%, and the top 25% (drawing hands).

    .venv/bin/python tools/analysis/fit_card_buckets.py --deals 8000
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_code"))

import card_buckets as cb  # noqa: E402

EQUITY_LEVELS = 10
RIVER_LEVELS = 15


def sample(street, deals, rng):
    deck = list(cb._CARDS)
    rows = []
    for _ in range(deals):
        cards = rng.sample(deck, 2 + street)
        rows.append(cb.measures(*cb.canonical(tuple(cards[:2]), tuple(cards[2:]))))
    return np.array(rows)


def quantile_edges(values, levels):
    return tuple(round(float(x), 4) for x in np.quantile(values, np.linspace(0, 1, levels + 1)[1:-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for street, name in ((3, "flop"), (4, "turn")):
        data = sample(street, args.deals, rng)
        strength, equity = data[:, 0], data[:, 1]
        print(f"EQUITY_EDGES[{street}] = {quantile_edges(equity, EQUITY_LEVELS)}  # {name}")
        potential = equity - strength
        print(f"POTENTIAL_EDGES[{street}] = {tuple(round(float(x), 4) for x in np.quantile(potential, (0.25, 0.75)))}  # {name}")
    river = sample(5, args.deals, rng)
    print(f"RIVER_STRENGTH_EDGES = {quantile_edges(river[:, 0], RIVER_LEVELS)}")


if __name__ == "__main__":
    main()
