"""
Write TensorBoard logs for a run that predates live logging.

Reads <run dir>/metrics.csv and, if present, exploitability_greedy_br.json,
and writes them to <run dir>/tensorboard_backfill.

Usage:
    .venv/bin/python tools/viz/metrics_to_tensorboard.py checkpoints/fullgame_100bb
    tensorboard --logdir checkpoints/fullgame_100bb/tensorboard_backfill
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_code"))

from tb_logging import log_exploitability, log_training_metrics, open_writer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default="checkpoints/fullgame_100bb")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logdir = run_dir / "tensorboard_backfill"
    writer = open_writer(logdir)

    rows = 0
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open() as metrics_file:
            for row in csv.DictReader(metrics_file):
                # Older runs only have "iteration".
                row.setdefault("effective_iteration", row.get("iteration"))
                if row.get("effective_iteration"):
                    log_training_metrics(writer, row)
                    rows += 1

    results = 0
    results_path = run_dir / "exploitability_greedy_br.json"
    if results_path.exists():
        for iteration, result in json.loads(results_path.read_text()).items():
            log_exploitability(writer, int(iteration), result)
            results += 1

    writer.close()
    if not rows and not results:
        sys.exit(f"Nothing to log: no metrics.csv or exploitability results in {run_dir}")
    print(f"Logged {rows} training rows and {results} exploitability results to {logdir}")


if __name__ == "__main__":
    main()
