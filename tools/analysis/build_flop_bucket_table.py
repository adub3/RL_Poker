"""
Precompute every flop bucket (new_code/flop_bucket_table.npz).

Each distinct flop deal (hole cards + flop, up to suit symmetry) gets the
bucket card_buckets would compute for it on the fly, so the table changes
nothing, it only makes flop buckets a lookup for training and LBR. The table
records a hash of card_bucket_centroids.json; card_buckets ignores a table
built from other centroids.

    .venv/bin/python tools/analysis/build_flop_bucket_table.py
"""

import argparse
import itertools
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_code"))

import card_buckets as cb  # noqa: E402

CARDS = list(cb._CARDS)


def _deals_for_board(board):
    """Canonical deals for every hole-card pair on one flop."""
    left = [card for card in CARDS if card not in board]
    return {cb.canonical(hole, board) for hole in itertools.combinations(left, 2)}


def _buckets(deals):
    return [(cb.encode_deal(hole, board), cb.bucket_index(hole, board)) for hole, board in deals]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--out", default=str(cb.FLOP_TABLE_PATH))
    args = parser.parse_args()

    t0 = time.time()
    # One flop per suit-symmetry class covers every canonical deal.
    boards = {cb.canonical_board(flop) for flop in itertools.combinations(CARDS, 3)}
    with multiprocessing.Pool(args.workers) as pool:
        deals = sorted(set().union(*pool.map(_deals_for_board, sorted(boards), chunksize=16)))
        print(f"{len(boards)} distinct flops, {len(deals):,} distinct flop deals ({time.time() - t0:.0f}s)")
        chunks = [deals[i:i + 2000] for i in range(0, len(deals), 2000)]
        rows = [row for part in pool.imap(_buckets, chunks) for row in part]
    rows.sort()
    keys = np.array([key for key, _ in rows], dtype=np.int64)
    buckets = np.array([bucket for _, bucket in rows], dtype=np.uint8)
    np.savez_compressed(args.out, keys=keys, buckets=buckets,
                        centroids_hash=np.array(cb.centroids_hash()))
    print(f"wrote {args.out}: {len(keys):,} deals ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
