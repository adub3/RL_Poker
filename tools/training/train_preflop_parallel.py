import argparse
import csv
import ctypes as _ctypes
import gc
import gzip
import json
import math
import os
import pickle as _pickle
import queue
import re
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
    BET_SIZINGS,
    bet_sizing_for_checkpoint,
    game_config_for_checkpoint,
    linear_weight_total,
    run_manifest,
    LinearMCCFRTrainer,
    PreflopActionAbstractor,
    StrategyTable,
    load_table,
    save_table,
    table_metrics,
)
from const import game_config  # noqa: E402
from preflop import preflop_coverage_metrics  # noqa: E402


def _save_snapshot(table, path):
    with gzip.open(path, "wb", compresslevel=1) as f:
        _pickle.dump(table.data, f, protocol=4)


def _load_snapshot(path):
    with gzip.open(path, "rb") as f:
        return StrategyTable(_pickle.load(f))


def _malloc_trim():
    try:
        _ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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


def merge_table_into(target, source):
    source_data = source.data if isinstance(source, StrategyTable) else source
    for infoset, node in source_data.items():
        target_node = target.data.setdefault(
            infoset,
            {"regret": {}, "strategy": {}, "strategy_sum": {}, "visits": 0},
        )
        target_node["visits"] += int(node.get("visits", 0))
        for field in ("regret", "strategy_sum"):
            target_field = target_node[field]
            for action, value in node.get(field, {}).items():
                key = str(action)
                target_field[key] = float(target_field.get(key, 0.0)) + float(value)
                target_node["strategy"].setdefault(key, 0.0)
        for action, value in node.get("strategy", {}).items():
            target_node["strategy"].setdefault(str(action), float(value))


def finalize_table_strategy(table):
    for node in table.data.values():
        actions = sorted(node["strategy"].keys())
        total_strategy = sum(
            float(node["strategy_sum"].get(action, 0.0)) for action in actions
        )
        if total_strategy > 0:
            for action in actions:
                node["strategy"][action] = (
                    float(node["strategy_sum"].get(action, 0.0)) / total_strategy
                )
            continue

        regrets = [max(0.0, float(node["regret"].get(action, 0.0))) for action in actions]
        normalizer = sum(regrets)
        if not actions:
            continue
        for action, regret in zip(actions, regrets):
            node["strategy"][action] = (
                regret / normalizer if normalizer > 0 else 1.0 / len(actions)
            )


def write_run_manifest(output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, big_blind = args.game_config, args.big_blind
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
        "bet_sizing": args.bet_sizing,
        "seed": args.seed,
        "start_iteration": args.start_iteration,
        "worker_start_iteration": args.worker_start_iteration,
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

        config, big_blind = args.game_config, args.big_blind
        rng = np.random.default_rng(int(args.seed) + worker_id * 1_000_003)
        game = pyspiel.load_game("universal_poker", config)
        starting_stack = int(config["stack"].split()[0])
        abstractor = PreflopActionAbstractor(
            big_blind=big_blind,
            starting_stack=starting_stack,
            rng=rng,
            max_actions=args.max_actions,
            bet_sizing=args.bet_sizing,
        )
        trainer = LinearMCCFRTrainer(
            game,
            table=StrategyTable(),
            action_abstractor=abstractor,
            prune_after=args.prune_after,
            prune_probability=args.prune_probability,
            prune_threshold=args.prune_threshold,
            regret_floor=args.prune_threshold - 10_000_000,
            cutoff_street=args.cutoff_street,
            start_iteration=args.worker_start_iteration,
            rng=rng,
        )

        remaining = int(args.iterations_per_worker)
        stage = 0
        snapshot_dir = Path(args.output_dir) / "_worker_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
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
            snapshot_path = (
                snapshot_dir
                / f"worker_{worker_id:03d}_stage_{stage:06d}_iter_"
                  f"{trainer.iteration:08d}.pkl.gz"
            )
            _save_snapshot(trainer.table, snapshot_path)
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
                    "snapshot_path": str(snapshot_path),
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


def write_stage(
    output_dir,
    stage,
    snapshots_meta,
    merged_workers_table,
    base_table=None,
    save_checkpoint=True,
    start_iteration=0,
    base_weight_total=0,
    worker_start_iteration=0,
):
    # merged_workers_table is already the incremental sum of all worker snapshots.
    # base_table (if present) is merged in-place to avoid allocating a third full copy.
    stage_start = time.perf_counter()
    merge_start = time.perf_counter()
    if base_table is not None:
        merge_table_into(merged_workers_table, base_table)
    finalize_table_strategy(merged_workers_table)
    merged = merged_workers_table
    merge_seconds = time.perf_counter() - merge_start
    merged_iteration = sum(int(s["local_iterations"]) for s in snapshots_meta)
    effective_iteration = int(start_iteration) + merged_iteration
    checkpoint_path = None
    checkpoint_save_seconds = 0.0
    if save_checkpoint:
        checkpoint_path = output_dir / f"mccfr_table_iter_{effective_iteration:08d}.json.gz"
        checkpoint_start = time.perf_counter()
        save_table(merged, checkpoint_path)
        checkpoint_save_seconds = time.perf_counter() - checkpoint_start

    metrics_start = time.perf_counter()
    # Each worker weights its iterations from its own count (plus the resume
    # offset); merged regrets are the sum over workers and the base table.
    weight_total = base_weight_total + sum(
        linear_weight_total(worker_start_iteration + int(s["local_iterations"]))
        - linear_weight_total(worker_start_iteration)
        for s in snapshots_meta
    )
    metrics = table_metrics(merged, weight_total=weight_total)
    metrics.update(preflop_coverage_metrics(merged))
    metrics_seconds = time.perf_counter() - metrics_start

    worker_train_seconds = [
        float(s.get("worker_train_seconds", 0.0)) for s in snapshots_meta
    ]
    worker_iterations_per_second = [
        float(s.get("worker_iterations_per_second", 0.0))
        for s in snapshots_meta
    ]
    worker_infosets_per_second = [
        float(s.get("worker_infosets_per_second", 0.0))
        for s in snapshots_meta
    ]
    stage_iterations_delta = sum(
        int(s.get("local_iterations_delta", 0)) for s in snapshots_meta
    )
    worker_train_seconds_max = (
        max(worker_train_seconds) if worker_train_seconds else 0.0
    )
    metrics["stage"] = stage
    metrics["iteration"] = merged_iteration
    metrics["effective_iteration"] = effective_iteration
    metrics["workers"] = len(snapshots_meta)
    metrics["stage_iterations_delta"] = stage_iterations_delta
    metrics["worker_train_seconds_sum"] = sum(worker_train_seconds)
    metrics["worker_train_seconds_max"] = worker_train_seconds_max
    metrics["effective_worker_iterations_per_second"] = (
        stage_iterations_delta / worker_train_seconds_max
        if worker_train_seconds_max > 0
        else 0.0
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


def run(args, resume_from_path=None):
    run_start = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, args)
    snapshot_dir = output_dir / "_worker_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for stale_snapshot in snapshot_dir.glob("worker_*_stage_*.pkl.gz"):
        stale_snapshot.unlink()

    stages = int(math.ceil(args.iterations_per_worker / args.merge_every))
    context = multiprocessing_context()
    result_queue = context.Queue()
    workers = [
        context.Process(target=train_worker, args=(worker_id, args, result_queue))
        for worker_id in range(args.workers)
    ]
    for worker in workers:
        worker.start()

    # Load base_table AFTER forking so workers don't inherit its COW pages.
    base_table = None
    if resume_from_path:
        base_table = load_table(resume_from_path)
        print(f"base_table loaded post-fork: {len(base_table.data)} infosets")

    # One StrategyTable accumulator per in-flight stage. Workers write snapshots
    # to disk and queue only the path so multiprocessing never pickles a full
    # table in a feeder thread while training continues.
    stage_tables = {}  # stage -> StrategyTable accumulator
    stage_meta = {}    # stage -> list of snapshot metadata
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
        snapshot_path = Path(message.pop("snapshot_path"))
        snapshot_table = _load_snapshot(snapshot_path)
        snapshot_path.unlink(missing_ok=True)
        if stage not in stage_tables:
            stage_tables[stage] = StrategyTable()
            stage_meta[stage] = []
        merge_table_into(stage_tables[stage], snapshot_table)
        del snapshot_table
        stage_meta[stage].append(message)

        if len(stage_meta[stage]) == args.workers:
            merged_workers_table = stage_tables.pop(stage)
            snapshots_meta = stage_meta.pop(stage)
            checkpoint_path, metrics = write_stage(
                output_dir,
                stage,
                snapshots_meta,
                merged_workers_table,
                base_table=base_table,
                save_checkpoint=(
                    stage % args.checkpoint_every == 0 or stage == stages
                ),
                start_iteration=args.start_iteration,
                base_weight_total=args.base_weight_total if base_table is not None else 0,
                worker_start_iteration=args.worker_start_iteration,
            )
            del merged_workers_table
            gc.collect()
            _malloc_trim()
            checkpoint_label = str(checkpoint_path) if checkpoint_path else "metrics-only"
            print(
                f"stage={stage}/{stages} iteration={metrics['effective_iteration']} "
                f"infosets={metrics['infosets']} "
                f"iter_per_sec={metrics['effective_worker_iterations_per_second']:.2f} "
                f"checkpoint={checkpoint_label}"
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
    parser.add_argument("--output-dir", default="auto")
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
    parser.add_argument(
        "--bet-sizing",
        choices=BET_SIZINGS,
        default=None,
        help="Postflop bet sizing. Defaults to v2 for new runs and to the "
             "checkpoint's own sizing with --resume-from.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a prior .json.gz checkpoint. Workers start fresh but the "
             "base table is merged into every subsequent checkpoint.",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        metavar="N",
        help="Effective iteration offset for linear weighting on resume. "
             "Auto-detected from the checkpoint filename when --resume-from is used.",
    )
    args = parser.parse_args()
    if args.smoke_benchmark:
        args.workers = 2
        args.iterations_per_worker = 8
        args.merge_every = 8
        args.checkpoint_every = 1
        if args.output_dir == "auto":
            args.output_dir = "checkpoints/smoke_benchmark"
    if args.cutoff_street == "none":
        args.cutoff_street = None

    # Auto output directory derived from run type and stack size.
    if args.output_dir == "auto":
        label = "preflop" if args.cutoff_street == "preflop" else "fullgame"
        args.output_dir = f"checkpoints/{label}_{args.stack_bb}bb"

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

    # Resume: detect start_iteration from filename now (before workers fork),
    # but defer the actual table load until after forking inside run().
    if args.resume_from:
        # A table only lines up with the bet sizing it was trained with.
        resume_sizing = bet_sizing_for_checkpoint(args.resume_from)
        if args.bet_sizing is None:
            args.bet_sizing = resume_sizing
        elif args.bet_sizing != resume_sizing:
            raise ValueError(
                f"--bet-sizing {args.bet_sizing} does not match {args.resume_from}, "
                f"which was trained with {resume_sizing} sizing. Start a new run "
                "to switch sizing."
            )
        if args.start_iteration is None:
            m = re.search(r"iter_(\d+)", str(args.resume_from))
            args.start_iteration = int(m.group(1)) if m else 0
        print(f"Resuming from {args.resume_from}, start_iteration={args.start_iteration}")
    else:
        if args.start_iteration is None:
            args.start_iteration = 0
    if args.bet_sizing is None:
        args.bet_sizing = "v2"
    print(f"Bet sizing: {args.bet_sizing}")

    # A resumed table keeps the rules it was trained under; new runs use const.py.
    if args.resume_from:
        args.game_config = game_config_for_checkpoint(args.resume_from)
        previous_workers = int(run_manifest(args.resume_from).get("workers") or args.workers)
    else:
        args.game_config, _ = game_config_for_stack(args.stack_bb)
        previous_workers = args.workers
    args.big_blind = max(int(value) for value in str(args.game_config["blind"]).split())
    args.stack_bb = int(args.game_config["stack"].split()[0]) // args.big_blind
    print(f"Seat order (firstPlayer): {args.game_config['firstPlayer']}")
    # start_iteration counts every worker's iterations together, but each
    # worker's linear weights follow its own count, so continue from that.
    args.worker_start_iteration = int(args.start_iteration) // previous_workers
    args.base_weight_total = previous_workers * linear_weight_total(args.worker_start_iteration)

    summary = run(args, resume_from_path=args.resume_from)
    if args.smoke_benchmark:
        print(
            "smoke_benchmark "
            f"iterations={summary['total_iterations']} "
            f"seconds={summary['elapsed_seconds']:.6f} "
            f"iterations_per_second={summary['iterations_per_second']:.3f}"
        )


if __name__ == "__main__":
    main()
