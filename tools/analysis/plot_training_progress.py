"""
Plot full-game MCCFR training progress from the metrics CSV.

Handles multiple run segments in the CSV, assigns effective iteration offsets,
and produces one multi-panel figure plus an optional summary CSV.

Usage:
    .venv/bin/python tools/analysis/plot_training_progress.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Run-segment definitions
# Each entry: (csv_row_start, csv_row_end_exclusive, effective_iter_offset, label)
# Determined from seam analysis of checkpoints/fullgame_100bb/metrics.csv
# ---------------------------------------------------------------------------
SEGMENTS = [
    (0,   100, 0,      "Run 1 — 12 workers"),
    (157, 211, 240000, "Run 12 — 10 workers (resume 240k)"),
]


def load_segments(csv_path: Path, segments) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    parts = []
    for row_start, row_end, offset, label in segments:
        seg = raw.iloc[row_start:row_end].copy()
        seg["eff_iter"] = seg["iteration"] + offset
        seg["run_label"] = label
        parts.append(seg)
    df = pd.concat(parts, ignore_index=True)
    df["pos_regret_per_visit"] = df["positive_regret"] / df["total_visits"].clip(lower=1)
    df["neg_regret_per_visit"] = df["negative_regret"].abs() / df["total_visits"].clip(lower=1)
    return df


def fmt_millions(x, _pos=None):
    if abs(x) >= 1e6:
        return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return str(int(x))


def plot_panel(ax, df, x_col, y_col, ylabel, title, *, logy=False):
    for label, group in df.groupby("run_label", sort=False):
        ax.plot(group[x_col] / 1e3, group[y_col], marker=".", markersize=3,
                linewidth=1.4, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Effective iterations (thousands)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    # Mark the clean 240k boundary
    ax.axvline(240, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(240, ax.get_ylim()[1], "  240k\n  (clean base)", fontsize=6,
            va="top", color="grey")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="checkpoints/fullgame_100bb/metrics.csv")
    parser.add_argument("--output-dir", default="assets/diagrams/training_progress_fullgame")
    args = parser.parse_args()

    csv_path = Path(args.metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_segments(csv_path, SEGMENTS)

    # -----------------------------------------------------------------------
    # Figure 1: 6-panel overview
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Full-game MCCFR training progress — 100bb", fontsize=12, y=1.01)

    panels = [
        ("infosets",              "Infosets",                     "Infosets in table",           False),
        ("avg_strategy_entropy_by_hand", "Strategy entropy (mean/hand)", "Shannon entropy",        False),
        ("total_visits",          "Total visits",                  "Cumulative visits",           False),
        ("positive_regret",       "Positive regret (total)",       "Σ positive regret",           True),
        ("pos_regret_per_visit",  "Pos regret / visit",            "Positive regret per visit",   True),
        ("neg_regret_per_visit",  "Neg regret / visit",            "Neg regret per visit (abs)",  True),
    ]

    for ax, (col, title, ylabel, logy) in zip(axes.flat, panels):
        plot_panel(ax, df, "eff_iter", col, ylabel, title, logy=logy)

    # Fix the 240k marker after ylim is set
    for ax, (col, *_rest) in zip(axes.flat, panels):
        ymin, ymax = ax.get_ylim()
        for t in list(ax.texts):
            t.remove()
        ax.axvline(240, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(242, ymax, "240k\n(base)", fontsize=6, va="top", color="grey")

    fig.tight_layout()
    out_path = out_dir / "training_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Infosets + entropy side-by-side, clean axes
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Infosets and entropy convergence", fontsize=11)

    for label, group in df.groupby("run_label", sort=False):
        ax1.plot(group["eff_iter"] / 1e3, group["infosets"] / 1e3,
                 marker=".", markersize=3, linewidth=1.4, label=label)
        ax2.plot(group["eff_iter"] / 1e3, group["avg_strategy_entropy_by_hand"],
                 marker=".", markersize=3, linewidth=1.4, label=label)

    for ax in (ax1, ax2):
        ax.axvline(240, color="grey", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_xlabel("Effective iterations (thousands)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    ax1.set_title("Infosets discovered")
    ax1.set_ylabel("Infosets (thousands)")
    ax2.set_title("Mean strategy entropy per hand class")
    ax2.set_ylabel("Shannon entropy (nats)")

    fig.tight_layout()
    out_path = out_dir / "infosets_entropy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # -----------------------------------------------------------------------
    # Figure 3: Strategy mass by action bucket over time (Run 1 only,
    #           which has the longest clean history)
    # -----------------------------------------------------------------------
    run1 = df[df["run_label"].str.startswith("Run 1")].copy()

    action_cols = [c for c in df.columns if c.startswith("strategy_mass_") and not c.endswith("_per_visit")]
    # Normalise to fractions
    total_mass = run1[action_cols].sum(axis=1)
    run1_frac = run1[action_cols].div(total_mass, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        run1["eff_iter"] / 1e3,
        [run1_frac[c] for c in action_cols],
        labels=[c.replace("strategy_mass_", "") for c in action_cols],
        alpha=0.75,
    )
    ax.set_title("Strategy mass distribution over training (Run 1 — 12 workers)", fontsize=10)
    ax.set_xlabel("Effective iterations (thousands)")
    ax.set_ylabel("Fraction of strategy mass")
    ax.set_xlim(0, run1["eff_iter"].max() / 1e3)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out_path = out_dir / "strategy_mass_stackplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # -----------------------------------------------------------------------
    # Figure 4: Regret balance — positive vs negative over training
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Regret balance over training", fontsize=11)

    for label, group in df.groupby("run_label", sort=False):
        x = group["eff_iter"] / 1e3
        axes[0].plot(x, group["positive_regret"], marker=".", markersize=3, linewidth=1.4, label=label)
        axes[0].plot(x, group["negative_regret"].abs(), marker=".", markersize=3,
                     linewidth=1.4, linestyle="--", label=f"{label} (neg)")
        ratio = group["positive_regret"] / group["negative_regret"].abs().clip(lower=1)
        axes[1].plot(x, ratio, marker=".", markersize=3, linewidth=1.4, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("Absolute regret totals (log scale)")
    axes[0].set_ylabel("Regret")
    axes[0].set_xlabel("Effective iterations (thousands)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)

    axes[1].set_title("Positive / negative regret ratio")
    axes[1].set_ylabel("Ratio (>1 = explorative)")
    axes[1].set_xlabel("Effective iterations (thousands)")
    axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.axvline(240, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    out_path = out_dir / "regret_balance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    cols = ["eff_iter", "infosets", "avg_strategy_entropy_by_hand",
            "positive_regret", "negative_regret", "total_visits",
            "pos_regret_per_visit", "neg_regret_per_visit", "run_label"]
    # Pick ~6 evenly-spaced rows across the full combined dataset
    idx = np.linspace(0, len(df) - 1, 6, dtype=int).tolist()
    summary = df.iloc[idx][cols]
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4g")
    print(f"saved: {summary_path}")

    print("\nDone. All graphs in:", out_dir)


if __name__ == "__main__":
    main()
