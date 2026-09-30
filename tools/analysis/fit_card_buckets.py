"""
Fit the v5 card-bucket centroids (new_code/card_bucket_centroids.json).

Samples random deals per street, computes each deal's features (see
new_code/card_buckets.py), and groups them with k-means. Centroids are sorted
by strong equity (the second feature), so bucket numbers rise with strength.

    .venv/bin/python tools/analysis/fit_card_buckets.py --deals 30000

Refitting changes every v5 key: tables trained with the old file no longer
line up. Only refit before starting a new run.
"""

import argparse
import json
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_code"))

import card_buckets as cb  # noqa: E402

BUCKETS = {3: 150, 4: 150, 5: 100}


def _sample(job):
    street, count, seed = job
    rng = random.Random(seed)
    deck = list(cb._CARDS)
    rows = []
    for _ in range(count):
        cards = rng.sample(deck, 2 + street)
        rows.append(cb.canonical_features(*cb.canonical(tuple(cards[:2]), tuple(cards[2:]))))
    return rows


def sample_features(street, deals, seed, workers):
    chunk = -(-deals // workers)
    jobs = [(street, min(chunk, deals - i * chunk), seed * 1000 + street * 100 + i)
            for i in range(workers) if deals - i * chunk > 0]
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(_sample, jobs)
    return np.array([row for part in parts for row in part])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--out", default=str(cb.CENTROIDS_PATH))
    args = parser.parse_args()

    centroids = {}
    for street in (3, 4, 5):
        data = sample_features(street, args.deals, args.seed, args.workers)
        found, labels = kmeans2(data, BUCKETS[street], minit="++", seed=args.seed + street, iter=50)
        used = np.unique(labels)
        found = found[used]  # drop any empty clusters
        found = found[np.argsort(found[:, 1], kind="stable")]
        centroids[str(street)] = [[round(float(x), 5) for x in row] for row in found]
        print(f"street {street}: {len(found)} buckets from {len(data)} deals")

    Path(args.out).write_text(json.dumps({
        "features": {"3": "equity, strong equity, potential",
                     "4": "equity, strong equity, potential",
                     "5": "strength, strong strength"},
        "deals_per_street": args.deals,
        "seed": args.seed,
        "centroids": centroids,
    }, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
