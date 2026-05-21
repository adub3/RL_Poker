import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def action_bucket(action):
    action = int(action)
    if action == 0:
        return "fold"
    if action == 1:
        return "call/check"
    if action < 500:
        return "<500"
    if action < 1_000:
        return "500-999"
    if action < 5_000:
        return "1k-4.9k"
    if action < 10_000:
        return "5k-9.9k"
    if action < 20_000:
        return "10k-19.9k"
    return "20k+"


def load_table(path):
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as in_file:
        return json.load(in_file)


def summarize_table(table):
    bucket_counts = Counter()
    bucket_strategy = defaultdict(float)
    bucket_regret = defaultdict(float)
    bucket_positive_regret = defaultdict(float)
    visit_histogram = Counter()

    for node in table.values():
        visits = int(node.get("visits", 0))
        visit_histogram[visits] += 1

        strategy_sum = node.get("strategy_sum", {})
        regret = node.get("regret", {})
        for action, mass in strategy_sum.items():
            bucket = action_bucket(action)
            bucket_counts[bucket] += 1
            bucket_strategy[bucket] += float(mass)
            action_regret = float(regret.get(action, 0.0))
            bucket_regret[bucket] += action_regret
            bucket_positive_regret[bucket] += max(0.0, action_regret)

    return {
        "bucket_counts": bucket_counts,
        "bucket_strategy": bucket_strategy,
        "bucket_regret": bucket_regret,
        "bucket_positive_regret": bucket_positive_regret,
        "visit_histogram": visit_histogram,
    }


def plot_bar(summary, key, title, ylabel, output_path):
    buckets = [
        "fold",
        "call/check",
        "<500",
        "500-999",
        "1k-4.9k",
        "5k-9.9k",
        "10k-19.9k",
        "20k+",
    ]
    values = [summary[key].get(bucket, 0.0) for bucket in buckets]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(buckets, values, color="#3A6EA5")
    ax.set_title(title)
    ax.set_xlabel("Action bucket")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_visits(summary, output_path):
    visits = sorted(summary["visit_histogram"])
    counts = [summary["visit_histogram"][visit] for visit in visits]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([str(visit) for visit in visits], counts, color="#6B8E23")
    ax.set_title("Infosets by Visit Count")
    ax.set_xlabel("Visits")
    ax.set_ylabel("Infosets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(summary, output_path, table_size):
    lines = [
        "# Strategy Table Summary",
        "",
        f"Infosets: {table_size}",
        "",
        "## Action Buckets",
        "",
        "| Bucket | Action entries | Strategy mass | Regret sum | Positive regret |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    buckets = [
        "fold",
        "call/check",
        "<500",
        "500-999",
        "1k-4.9k",
        "5k-9.9k",
        "10k-19.9k",
        "20k+",
    ]
    for bucket in buckets:
        lines.append(
            "| {bucket} | {count} | {strategy:.4f} | {regret:.4f} | {pos:.4f} |".format(
                bucket=bucket,
                count=summary["bucket_counts"].get(bucket, 0),
                strategy=summary["bucket_strategy"].get(bucket, 0.0),
                regret=summary["bucket_regret"].get(bucket, 0.0),
                pos=summary["bucket_positive_regret"].get(bucket, 0.0),
            )
        )

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default="new_code/mccfr_table_smoke.json",
        help="Path to a saved MCCFR table JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/diagrams/strategy_table",
        help="Directory where charts and summary should be written.",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(table_path)
    summary = summarize_table(table)

    plot_bar(
        summary,
        "bucket_counts",
        "Action Entries by Bucket",
        "Action entries",
        output_dir / "action_bucket_counts.png",
    )
    plot_bar(
        summary,
        "bucket_strategy",
        "Accumulated Strategy Mass by Bucket",
        "Strategy mass",
        output_dir / "strategy_mass_by_bucket.png",
    )
    plot_bar(
        summary,
        "bucket_positive_regret",
        "Positive Regret by Bucket",
        "Positive regret",
        output_dir / "positive_regret_by_bucket.png",
    )
    plot_visits(summary, output_dir / "infoset_visit_counts.png")
    write_summary(summary, output_dir / "summary.md", len(table))

    print(f"Wrote strategy-table charts to {output_dir}")


if __name__ == "__main__":
    main()
