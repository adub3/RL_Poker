"""
Pre-export a checkpoint's average strategy to a compact SQLite database.

This is a one-time step per checkpoint. The resulting .db file is used by
mc_exploitability.py via SQLiteBlueprint — only queried infosets are loaded
into Python memory (LRU-cached), keeping RAM usage ~20 MB instead of ~800 MB.

Schema:
    strategy(infoset TEXT PRIMARY KEY, actions TEXT, probs BLOB)
    - infoset: the infoset key string
    - actions: comma-joined action names (e.g. "fold,call,jam")
    - probs:   float32 little-endian binary blob (same order as actions)

Usage:
    # Export one checkpoint:
    .venv/bin/python tools/evaluation/export_strategy_db.py \\
        checkpoints/fullgame_100bb/mccfr_table_iter_16815000.json.gz

    # Export all checkpoints in a directory:
    .venv/bin/python tools/evaluation/export_strategy_db.py \\
        --all checkpoints/fullgame_100bb/
"""

import argparse
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

STRATEGIES_DIR = Path("checkpoints/fullgame_100bb/strategies")


def export_checkpoint(ckpt_path: Path, out_db: Path, batch_size: int = 10_000):
    out_db.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {ckpt_path.name} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    with gzip.open(ckpt_path, "rt") as f:
        data = json.load(f)
    print(f"{len(data):,} nodes  ({time.perf_counter()-t0:.1f}s)")

    print(f"Writing {out_db.name} ...", end=" ", flush=True)
    t1 = time.perf_counter()

    conn = sqlite3.connect(out_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS strategy "
        "(infoset TEXT PRIMARY KEY, actions TEXT, probs BLOB)"
    )
    conn.execute("DELETE FROM strategy")

    batch = []
    written = 0
    skipped = 0

    for infoset_key, node in data.items():
        ss = node.get("strategy_sum", {})
        total = sum(float(v) for v in ss.values())
        if total <= 0:
            skipped += 1
            continue

        actions_list = list(ss.keys())
        probs_f32 = np.array([float(ss[a]) / total for a in actions_list],
                             dtype=np.float32)
        batch.append((
            infoset_key,
            ",".join(actions_list),
            probs_f32.tobytes(),
        ))

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO strategy VALUES (?,?,?)", batch)
            conn.commit()
            written += len(batch)
            batch = []

    if batch:
        conn.executemany("INSERT OR REPLACE INTO strategy VALUES (?,?,?)", batch)
        conn.commit()
        written += len(batch)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_infoset ON strategy(infoset)")
    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - t1
    size_mb = out_db.stat().st_size / 1e6
    print(f"{written:,} infosets written, {skipped} skipped  "
          f"({elapsed:.1f}s, {size_mb:.1f} MB)")


def db_path_for(ckpt_path: Path) -> Path:
    iters = ckpt_path.stem.split("iter_")[1].split(".")[0]
    return STRATEGIES_DIR / f"strategy_{iters}.db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Checkpoint .json.gz file or directory (with --all)")
    parser.add_argument("--all", action="store_true",
                        help="Export every checkpoint in the given directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: checkpoints/fullgame_100bb/strategies/)")
    args = parser.parse_args()

    global STRATEGIES_DIR
    if args.out_dir:
        STRATEGIES_DIR = Path(args.out_dir)

    if args.all:
        ckpts = sorted(Path(args.path).glob("mccfr_table_iter_*.json.gz"))
        if not ckpts:
            sys.exit(f"No checkpoints found in {args.path}")
        for ckpt in ckpts:
            out = db_path_for(ckpt)
            if out.exists():
                print(f"[skip] {ckpt.name}  →  {out.name} already exists")
                continue
            export_checkpoint(ckpt, out)
    else:
        ckpt = Path(args.path)
        export_checkpoint(ckpt, db_path_for(ckpt))


if __name__ == "__main__":
    main()
