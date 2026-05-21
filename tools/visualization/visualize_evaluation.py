import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", default="eval/smoke/hands.csv")
    parser.add_argument("--output-dir", default="assets/diagrams/evaluation_smoke")
    args = parser.parse_args()

    rows = load_rows(args.hands)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hands = [int(row["hand"]) for row in rows]
    bankroll = [float(row["bankroll"]) for row in rows]
    returns = [float(row["hero_return"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hands, bankroll, color="#3A6EA5")
    ax.set_title("Evaluation Bankroll Curve")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bankroll_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=30, color="#6B8E23")
    ax.set_title("Per-Hand Return Distribution")
    ax.set_xlabel("Hero return")
    ax.set_ylabel("Hands")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "return_distribution.png", dpi=160)
    plt.close(fig)

    print(f"Wrote evaluation charts to {output_dir}")


if __name__ == "__main__":
    main()
