import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
NEW_CODE = ROOT / "new_code"
sys.path.insert(0, str(NEW_CODE))

from ai import load_table, table_metrics  # noqa: E402
from preflop import preflop_coverage_metrics  # noqa: E402


CHECKPOINT_RE = re.compile(r"mccfr_table_iter_(\d+)\.json\.gz$")


def checkpoint_iteration(path):
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def checkpoint_paths(checkpoint_dir):
    paths = []
    for path in Path(checkpoint_dir).glob("mccfr_table_iter_*.json.gz"):
        iteration = checkpoint_iteration(path)
        if iteration is not None:
            paths.append((iteration, path))
    return [path for _, path in sorted(paths)]


def strategy_l1_delta(previous, current):
    if previous is None:
        return 0.0, 0

    common_infosets = set(previous) & set(current)
    if not common_infosets:
        return 0.0, 0

    total = 0.0
    for infoset in common_infosets:
        previous_actions = previous[infoset]
        current_actions = current[infoset]
        actions = set(previous_actions) | set(current_actions)
        total += sum(
            abs(
                float(current_actions.get(action, 0.0))
                - float(previous_actions.get(action, 0.0))
            )
            for action in actions
        )

    return total / len(common_infosets), len(common_infosets)


def collect_rows(paths):
    rows = []
    previous_strategy = None
    previous_infosets = set()

    for path in paths:
        table = load_table(path)
        metrics = table_metrics(table)
        coverage = preflop_coverage_metrics(table)
        strategy = table.average_strategy()
        delta, common_infosets = strategy_l1_delta(previous_strategy, strategy)
        infosets = set(table.data)
        new_infosets = len(infosets - previous_infosets) if previous_infosets else len(infosets)
        visits = max(1, int(metrics["total_visits"]))
        infoset_count = max(1, int(metrics["infosets"]))

        row = {
            "iteration": checkpoint_iteration(path),
            "checkpoint": str(path),
            "infosets": metrics["infosets"],
            "preflop_infosets": coverage["preflop_infosets"],
            "starting_hand_classes": coverage["starting_hand_classes"],
            "avg_strategy_entropy_by_hand": coverage["avg_strategy_entropy_by_hand"],
            "positive_regret": metrics["positive_regret"],
            "negative_regret": metrics["negative_regret"],
            "positive_regret_per_infoset": metrics["positive_regret"] / infoset_count,
            "negative_regret_abs_per_infoset": abs(metrics["negative_regret"]) / infoset_count,
            "positive_regret_per_visit": metrics["positive_regret"] / visits,
            "negative_regret_abs_per_visit": abs(metrics["negative_regret"]) / visits,
            "strategy_l1_delta": delta,
            "common_infosets_for_delta": common_infosets,
            "new_infosets_since_previous": new_infosets,
        }
        rows.append(row)
        previous_strategy = strategy
        previous_infosets = infosets

    return rows


def write_csv(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(rows, keys, title, ylabel, output_path):
    iterations = [int(row["iteration"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in keys:
        ax.plot(iterations, [float(row[key]) for row in rows], marker="o", label=key)
    ax.set_title(title)
    ax.set_xlabel("Merged iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints/preflop_100bb")
    parser.add_argument(
        "--output-dir",
        default="assets/diagrams/preflop_convergence_100bb",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only process the most recent N checkpoints. Use 0 for all.",
    )
    args = parser.parse_args()

    paths = checkpoint_paths(args.checkpoint_dir)
    if args.last:
        paths = paths[-args.last :]
    if not paths:
        raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(paths)
    write_csv(output_dir / "convergence_metrics.csv", rows)
    plot_lines(
        rows,
        ["strategy_l1_delta"],
        "Average Strategy Drift Between Checkpoints",
        "Mean L1 probability change",
        output_dir / "strategy_l1_delta.png",
    )
    plot_lines(
        rows,
        ["positive_regret_per_infoset", "negative_regret_abs_per_infoset"],
        "Regret Magnitude per Infoset",
        "Regret / infoset",
        output_dir / "regret_per_infoset.png",
    )
    plot_lines(
        rows,
        ["positive_regret_per_visit", "negative_regret_abs_per_visit"],
        "Regret Magnitude per Visit",
        "Regret / visit",
        output_dir / "regret_per_visit.png",
    )
    plot_lines(
        rows,
        ["avg_strategy_entropy_by_hand"],
        "Average Preflop Strategy Entropy",
        "Bits",
        output_dir / "entropy_by_hand.png",
    )
    plot_lines(
        rows,
        ["infosets", "preflop_infosets", "new_infosets_since_previous"],
        "Coverage Growth",
        "Count",
        output_dir / "coverage_growth.png",
    )

    print(f"Wrote convergence metrics to {output_dir}")


if __name__ == "__main__":
    main()
