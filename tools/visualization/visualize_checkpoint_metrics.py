import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r") as csv_file:
        return list(csv.DictReader(csv_file))


def numeric_series(rows, key):
    return [float(row[key]) for row in rows]


def plot_lines(rows, keys, title, ylabel, output_path):
    iterations = numeric_series(rows, "iteration")

    fig, ax = plt.subplots(figsize=(11, 6))
    for key in keys:
        ax.plot(iterations, numeric_series(rows, key), marker="o", label=key)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="checkpoints/smoke/metrics.csv",
        help="Checkpoint metrics CSV produced during training.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/diagrams/checkpoint_metrics",
        help="Directory where metric charts should be written.",
    )
    args = parser.parse_args()

    rows = load_rows(args.metrics)
    if not rows:
        raise ValueError(f"No metrics rows found in {args.metrics}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_lines(
        rows,
        ["infosets", "total_visits"],
        "Table Growth Over Training",
        "Count",
        output_dir / "table_growth.png",
    )
    plot_lines(
        rows,
        ["positive_regret", "negative_regret"],
        "Regret Totals Over Training",
        "Regret",
        output_dir / "regret_totals.png",
    )
    plot_lines(
        rows,
        [
            "strategy_mass_fold",
            "strategy_mass_call/check",
            "strategy_mass_<500",
            "strategy_mass_1k-4.9k",
            "strategy_mass_10k-19.9k",
            "strategy_mass_20k+",
        ],
        "Strategy Mass by Action Bucket",
        "Strategy mass",
        output_dir / "strategy_mass_by_bucket_over_time.png",
    )

    print(f"Wrote checkpoint metric charts to {output_dir}")


if __name__ == "__main__":
    main()
