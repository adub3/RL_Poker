"""
TensorBoard logging for training and exploitability metrics.

Training writes to <run dir>/tensorboard. View it with:

    tensorboard --logdir checkpoints/fullgame_100bb/tensorboard

The step on every chart is the effective training iteration.
"""

import os

from ai import ACTION_BUCKETS

# chart tag -> metrics key (metrics.csv column)
TRAINING_SCALARS = {
    "convergence/avg_regret_bound": "avg_regret_bound",
    "convergence/avg_regret_per_infoset": "avg_regret_per_infoset",
    "table/infosets": "infosets",
    "table/preflop_infosets": "preflop_infosets",
    "table/smallest_worker_table": "worker_table_infosets_min",
    "strategy/preflop_entropy_by_hand": "avg_strategy_entropy_by_hand",
    "speed/iterations_per_second": "effective_worker_iterations_per_second",
    "speed/checkpoint_save_seconds": "checkpoint_save_seconds",
    "system/memory_used_gb": "system_memory_used_gb",
    "system/memory_available_gb": "system_memory_available_gb",
}


def open_writer(logdir):
    from tensorboardX import SummaryWriter

    return SummaryWriter(logdir=str(logdir), flush_secs=30)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def log_training_metrics(writer, metrics):
    """Log one metrics row (a dict, as written to metrics.csv)."""
    step = int(float(metrics["effective_iteration"]))
    for tag, key in TRAINING_SCALARS.items():
        value = _number(metrics.get(key))
        if value is not None:
            writer.add_scalar(tag, value, step)

    # Share of average-strategy mass per action type, so shifts in play style show up.
    total = _number(metrics.get("total_strategy_mass"))
    if total:
        for bucket in ACTION_BUCKETS:
            mass = _number(metrics.get(f"strategy_mass_{bucket}"))
            if mass:
                tag = bucket.replace("/", "_").replace("<", "under_").replace("+", "_plus")
                writer.add_scalar(f"strategy/action_share/{tag}", mass / total, step)


def log_exploitability(writer, iteration, result):
    """Log one mc_exploitability.py result (bb/100) at a checkpoint's iteration."""
    step = int(iteration)
    writer.add_scalar("exploitability/bb100", result["exploitability_bb100"], step)
    low, high = result["ci95_bb100"]
    writer.add_scalar("exploitability/ci95_low", low, step)
    writer.add_scalar("exploitability/ci95_high", high, step)
    for seat in result["seats"]:
        prefix = f"exploitability/seat_P{seat['seat']}"
        writer.add_scalar(f"{prefix}_best_response_bb100", seat["value_bb100"], step)
        writer.add_scalar(f"{prefix}_blueprint_missing_rate", seat["blueprint_missing_rate"], step)


def system_memory_gb():
    """(used, available) in GB from /proc/meminfo, or (None, None) off Linux."""
    try:
        with open("/proc/meminfo") as meminfo:
            fields = {line.split(":")[0]: int(line.split()[1]) for line in meminfo}
    except (OSError, ValueError, IndexError):
        return None, None
    total, available = fields.get("MemTotal"), fields.get("MemAvailable")
    if total is None or available is None:
        return None, None
    return (total - available) / 1024**2, available / 1024**2


def default_logdir(run_dir):
    return os.path.join(str(run_dir), "tensorboard")
