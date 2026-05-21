import argparse
import csv
import json
import math
import os
import queue
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NEW_CODE = ROOT / "new_code"
sys.path.insert(0, str(NEW_CODE))

from ai import (  # noqa: E402
    LinearMCCFRTrainer,
    PreflopActionAbstractor,
    StrategyTable,
    merge_strategy_tables,
    save_table,
    table_metrics,
)
from const import game_config  # noqa: E402
from preflop import preflop_coverage_metrics  # noqa: E402


def default_worker_count():
    cpus = os.cpu_count() or 2
    return max(1, min(12, cpus - 2))


def multiprocessing_context():
    try:
        return get_context("fork")
    except ValueError:
        return get_context()


def game_config_for_stack(stack_bb):
    config = dict(game_config)
    blinds = [int(value) for value in str(config["blind"]).split()]
    big_blind = max(blinds)
    stack = int(stack_bb) * big_blind
    config["stack"] = f"{stack} {stack}"
    return config, big_blind


def write_run_manifest(output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, big_blind = game_config_for_stack(args.stack_bb)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "preflop_parallel_v3",
        "checkpoint_policy": "merged worker snapshots only",
        "checkpoint_every": args.checkpoint_every,
        "workers": args.workers,
        "iterations_per_worker": args.iterations_per_worker,
        "merge_every": args.merge_every,
        "merged_iterations_per_checkpoint": args.workers * args.merge_every,
        "prune_after": args.prune_after,
        "prune_probability": args.prune_probability,
        "prune_threshold": args.prune_threshold,
        "max_actions": args.max_actions,
        "cutoff_street": args.cutoff_street,
        "seed": args.seed,
        "stack_bb": args.stack_bb,
        "big_blind": big_blind,
        "game_config": config,
        "preflop_infoset": (
            "[PF:<hand>][pos:P0/P1][tc:<to-call-bucket>]"
            "[eff:<effective-stack-bucket>][seq:<bucketed-sequence>]"
        ),
        "preflop_sequence_buckets": [
            "c",
            "f",
            "r:2bb",
            "r:2_5bb",
            "r:3bb",
            "r:4bb",
            "r:3x",
            "r:4x",
            "r:25eff",
            "r:50eff",
            "r:jam",
        ],
        "preflop_action_keys": [
            "fold",
            "call",
            "check",
            "min_raise",
            "open_2_5bb",
            "open_3bb",
            "raise_2_5bb",
            "raise_3bb",
            "raise_3x",
            "raise_4x",
            "raise_25eff",
            "raise_50eff",
            "jam",
        ],
    }
    with (output_dir / "run_manifest.json").open("w") as out_file:
        json.dump(manifest, out_file, indent=2, sort_keys=True)


def train_worker(worker_id, args, result_queue):
    try:
        import pyspiel

        config, big_blind = game_config_for_stack(args.stack_bb)
        rng = np.random.default_rng(int(args.seed) + worker_id * 1_000_003)
        game = pyspiel.load_game("universal_poker", config)
        abstractor = PreflopActionAbstractor(
            big_blind=big_blind,
            rng=rng,
            max_actions=args.max_actions,
        )
        trainer = LinearMCCFRTrainer(
            game,
            table=StrategyTable(),
            action_abstractor=abstractor,
            prune_after=args.prune_after,
            prune_probability=args.prune_probability,
            prune_threshold=args.prune_threshold,
            cutoff_street=args.cutoff_street,
            rng=rng,
        )

        remaining = int(args.iterations_per_worker)
        stage = 0
        while remaining > 0:
            stage += 1
            chunk = min(int(args.merge_every), remaining)
            stage_start = time.perf_counter()
            before_iteration = trainer.iteration
            trainer.train(chunk)
            train_seconds = time.perf_counter() - stage_start
            remaining -= chunk
            local_iterations_delta = trainer.iteration - before_iteration
            infosets = len(trainer.table.data)
            result_queue.put(
                {
                    "type": "snapshot",
                    "worker_id": worker_id,
                    "stage": stage,
                    "local_iterations": trainer.iteration,
                    "local_iterations_delta": local_iterations_delta,
                    "worker_train_seconds": train_seconds,
                    "worker_iterations_per_second": (
                        local_iterations_delta / train_seconds
                        if train_seconds > 0
                        else 0.0
                    ),
                    "worker_infosets": infosets,
                    "worker_infosets_per_second": (
                        infosets / train_seconds if train_seconds > 0 else 0.0
                    ),
                    "table": trainer.table.data,
                }
            )

        result_queue.put({"type": "done", "worker_id": worker_id})
    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "error": repr(exc),
            }
        )


def append_metrics(metrics_path, metrics):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "iteration"] + [
        key for key in metrics.keys() if key not in ("stage", "iteration")
    ]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def write_stage(output_dir, stage, snapshots, save_checkpoint=True):
    stage_start = time.perf_counter()
    merge_start = time.perf_counter()
    merged = merge_strategy_tables(StrategyTable(snapshot["table"]) for snapshot in snapshots)
    merge_seconds = time.perf_counter() - merge_start
    merged_iteration = sum(int(snapshot["local_iterations"]) for snapshot in snapshots)
    checkpoint_path = None
    checkpoint_save_seconds = 0.0
    if save_checkpoint:
        checkpoint_path = output_dir / f"mccfr_table_iter_{merged_iteration:08d}.json.gz"
        checkpoint_start = time.perf_counter()
        save_table(merged, checkpoint_path)
        checkpoint_save_seconds = time.perf_counter() - checkpoint_start

    metrics_start = time.perf_counter()
    metrics = table_metrics(merged)
    metrics.update(preflop_coverage_metrics(merged))
    metrics_seconds = time.perf_counter() - metrics_start

    worker_train_seconds = [
        float(snapshot.get("worker_train_seconds", 0.0)) for snapshot in snapshots
    ]
    worker_iterations_per_second = [
        float(snapshot.get("worker_iterations_per_second", 0.0))
        for snapshot in snapshots
    ]
    worker_infosets_per_second = [
        float(snapshot.get("worker_infosets_per_second", 0.0))
        for snapshot in snapshots
    ]
    metrics["stage"] = stage
    metrics["iteration"] = merged_iteration
    metrics["workers"] = len(snapshots)
    metrics["worker_train_seconds_sum"] = sum(worker_train_seconds)
    metrics["worker_train_seconds_max"] = (
        max(worker_train_seconds) if worker_train_seconds else 0.0
    )
    metrics["worker_iterations_per_second_sum"] = sum(worker_iterations_per_second)
    metrics["worker_iterations_per_second_mean"] = (
        sum(worker_iterations_per_second) / len(worker_iterations_per_second)
        if worker_iterations_per_second
        else 0.0
    )
    metrics["worker_infosets_per_second_sum"] = sum(worker_infosets_per_second)
    metrics["worker_infosets_per_second_mean"] = (
        sum(worker_infosets_per_second) / len(worker_infosets_per_second)
        if worker_infosets_per_second
        else 0.0
    )
    metrics["parent_merge_seconds"] = merge_seconds
    metrics["metrics_seconds"] = metrics_seconds
    metrics["checkpoint_save_seconds"] = checkpoint_save_seconds
    metrics["stage_wall_seconds"] = time.perf_counter() - stage_start
    append_metrics(output_dir / "metrics.csv", metrics)
    return checkpoint_path, metrics


def run(args):
    run_start = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, args)

    stages = int(math.ceil(args.iterations_per_worker / args.merge_every))
    context = multiprocessing_context()
    result_queue = context.Queue()
    workers = [
        context.Process(target=train_worker, args=(worker_id, args, result_queue))
        for worker_id in range(args.workers)
    ]
    for worker in workers:
        worker.start()

    snapshots_by_stage = {}
    done = 0
    while done < args.workers:
        try:
            message = result_queue.get(timeout=5)
        except queue.Empty:
            continue

        if message["type"] == "error":
            raise RuntimeError(
                f"Worker {message['worker_id']} failed: {message['error']}"
            )
        if message["type"] == "done":
            done += 1
            continue

        stage = int(message["stage"])
        snapshots_by_stage.setdefault(stage, []).append(message)
        if len(snapshots_by_stage[stage]) == args.workers:
            checkpoint_path, metrics = write_stage(
                output_dir,
                stage,
                snapshots_by_stage.pop(stage),
                save_checkpoint=(
                    stage % args.checkpoint_every == 0 or stage == stages
                ),
            )
            checkpoint_label = str(checkpoint_path) if checkpoint_path else "metrics-only"
            print(
                f"stage={stage}/{stages} iteration={metrics['iteration']} "
                f"infosets={metrics['infosets']} checkpoint={checkpoint_label}"
            )

    for worker in workers:
        worker.join()

    elapsed = time.perf_counter() - run_start
    total_iterations = int(args.workers) * int(args.iterations_per_worker)
    return {
        "elapsed_seconds": elapsed,
        "total_iterations": total_iterations,
        "iterations_per_second": (
            total_iterations / elapsed if elapsed > 0 else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--iterations-per-worker", type=int, default=1_000)
    parser.add_argument("--merge-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write a full checkpoint every N merge stages. Metrics still write every stage.",
    )
    parser.add_argument("--output-dir", default="checkpoints/preflop_100bb")
    parser.add_argument(
        "--smoke-benchmark",
        action="store_true",
        help=(
            "Run a fixed 2-worker x 8-iteration smoke benchmark and print "
            "aggregate iterations/sec."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--stack-bb", type=int, default=100)
    parser.add_argument("--prune-after", type=int, default=200)
    parser.add_argument("--prune-probability", type=float, default=0.95)
    parser.add_argument("--prune-threshold", type=float, default=-300_000_000)
    parser.add_argument(
        "--max-actions",
        type=int,
        help=(
            "Optional cap on abstract actions per decision. Useful for faster "
            "preflop blueprint runs; omit for the full configured menu."
        ),
    )
    parser.add_argument(
        "--cutoff-street",
        choices=("none", "preflop"),
        default="none",
        help=(
            "When set to preflop, stop traversal after preflop and use a cached "
            "hand-strength value estimate instead of traversing postflop."
        ),
    )
    args = parser.parse_args()
    if args.smoke_benchmark:
        default_output_dir = parser.get_default("output_dir")
        args.workers = 2
        args.iterations_per_worker = 8
        args.merge_every = 8
        args.checkpoint_every = 1
        if args.output_dir == default_output_dir:
            args.output_dir = "checkpoints/preflop_smoke_benchmark"
    if args.cutoff_street == "none":
        args.cutoff_street = None

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.iterations_per_worker < 1:
        raise ValueError("--iterations-per-worker must be at least 1")
    if args.merge_every < 1:
        raise ValueError("--merge-every must be at least 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.prune_after < 0:
        raise ValueError("--prune-after must be non-negative")
    if not 0.0 <= args.prune_probability <= 1.0:
        raise ValueError("--prune-probability must be between 0 and 1")
    if args.max_actions is not None and args.max_actions < 2:
        raise ValueError("--max-actions must be at least 2 when provided")

    summary = run(args)
    if args.smoke_benchmark:
        print(
            "smoke_benchmark "
            f"iterations={summary['total_iterations']} "
            f"seconds={summary['elapsed_seconds']:.6f} "
            f"iterations_per_second={summary['iterations_per_second']:.3f}"
        )


if __name__ == "__main__":
    main()
