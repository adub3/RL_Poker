"""
Training metrics dashboard from metrics.csv.

Produces a single multi-panel figure covering:
  1. Infoset growth
  2. Strategy entropy
  3. Action distribution (stacked area)
  4. Regret per iteration (convergence signal)
  5. Training throughput

Usage:
    python tools/viz/training_dashboard.py [metrics_csv] [--out path]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BG       = "#16213e"
PANEL_BG = "#0d1b2a"
GRID     = "#1e3a5f"

BUCKET_MAP = {
    "fold":        "fold",
    "call/check":  "call",
    "min_raise":   "small",
    "open_2_5bb":  "small",
    "open_3bb":    "small",
    "raise_2_5bb": "small",
    "raise_3bb":   "small",
    "bet_33":      "small",
    "bet_50":      "small",
    "<500":        "small",
    "raise_3x":    "medium",
    "raise_4x":    "medium",
    "raise_25eff": "medium",
    "500-999":     "medium",
    "1k-4.9k":     "medium",
    "raise_50eff": "large",
    "bet_pot":     "large",
    "bet_2x":      "large",
    "5k-9.9k":     "large",
    "10k-19.9k":   "large",
    "jam":         "jam",
    "20k+":        "jam",
}

BUCKET_COLORS = {
    "fold":   "#c0392b",
    "call":   "#7f8c8d",
    "small":  "#27ae60",
    "medium": "#f39c12",
    "large":  "#8e44ad",
    "jam":    "#2c3e50",
}
BUCKET_ORDER = ["fold", "call", "small", "medium", "large", "jam"]
BUCKET_LABELS = {
    "fold": "Fold", "call": "Call/Check",
    "small": "Raise small", "medium": "Raise med",
    "large": "Raise large", "jam": "Jam",
}


def _safe(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("iteration")]

    # The iteration counter resets every time training is resumed from a checkpoint.
    # Compute cumulative iterations so the x-axis is monotonically increasing.
    cumulative = 0
    prev_iter  = 0
    for r in rows:
        cur = int(r["iteration"])
        if cur > prev_iter:
            cumulative += cur - prev_iter
        else:
            cumulative += cur
        r["_cumulative_iters"] = cumulative
        prev_iter = cur

    return rows


def _iter_fmt(x, _):
    x = int(x)
    if x >= 1_000_000:
        s = f"{x/1_000_000:.1f}M"
        return s.replace(".0M", "M")
    if x >= 1_000:
        s = f"{x/1_000:.0f}k"
        return s
    return str(x)


def _style_ax(ax, title, xlabel="Cumulative iterations", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_xlabel(xlabel, color="#aabbcc", fontsize=8)
    ax.set_ylabel(ylabel, color="#aabbcc", fontsize=8)
    ax.tick_params(colors="#aabbcc", labelsize=7)
    ax.spines[:].set_color(GRID)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_iter_fmt))
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)


def plot_infoset_growth(ax, rows):
    iters    = [int(r["_cumulative_iters"]) for r in rows]
    total    = [_safe(r["infosets"]) for r in rows]
    preflop  = [_safe(r.get("preflop_infosets")) for r in rows]

    ax.plot(iters, total, color="#3498db", linewidth=1.5, label="Total")
    if any(v for v in preflop):
        ax.plot(iters, preflop, color="#1abc9c", linewidth=1.5,
                linestyle="--", label="Preflop")
        ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white",
                  edgecolor=GRID, framealpha=0.9)

    ax.fill_between(iters, total, alpha=0.10, color="#3498db")
    _style_ax(ax, "Infoset growth", ylabel="Infosets")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x < 1e6 else f"{x/1e6:.1f}M"))


def plot_entropy(ax, rows):
    iters = [int(r["_cumulative_iters"]) for r in rows]
    ents  = [_safe(r.get("avg_strategy_entropy_by_hand")) for r in rows]
    valid = [(i, e) for i, e in zip(iters, ents) if e is not None]
    if not valid:
        return
    xi, ye = zip(*valid)
    ax.plot(xi, ye, color="#e74c3c", linewidth=1.5)
    ax.fill_between(xi, ye, alpha=0.12, color="#e74c3c")
    _style_ax(ax, "Strategy entropy (avg per hand class)",
              ylabel="Shannon entropy (nats)")


def plot_action_distribution(ax, rows):
    iters = [int(r["_cumulative_iters"]) for r in rows]

    # Accumulate mass per bucket per row
    bucket_series = {b: [] for b in BUCKET_ORDER}
    for r in rows:
        totals = {b: 0.0 for b in BUCKET_ORDER}
        grand  = 0.0
        for col, bucket in BUCKET_MAP.items():
            v = _safe(r.get(f"strategy_mass_{col}"))
            if v and v > 0:
                totals[bucket] += v
                grand += v
        if grand > 0:
            for b in BUCKET_ORDER:
                bucket_series[b].append(totals[b] / grand)
        else:
            for b in BUCKET_ORDER:
                bucket_series[b].append(0.0)

    ys = np.array([bucket_series[b] for b in BUCKET_ORDER])
    ax.stackplot(iters, ys,
                 labels=[BUCKET_LABELS[b] for b in BUCKET_ORDER],
                 colors=[BUCKET_COLORS[b] for b in BUCKET_ORDER],
                 alpha=0.85)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=GRID, framealpha=0.9,
              ncol=3)
    _style_ax(ax, "Action distribution over training",
              ylabel="Share of strategy mass")


def plot_regret(ax, rows):
    iters = [int(r["_cumulative_iters"]) for r in rows]
    pos   = []
    neg   = []
    for r in rows:
        it  = int(r["_cumulative_iters"])
        inf = _safe(r.get("infosets")) or 1
        pr  = _safe(r.get("positive_regret"))
        nr  = _safe(r.get("negative_regret"))
        pos.append(pr / (inf * it) if pr is not None else None)
        neg.append(abs(nr) / (inf * it) if nr is not None else None)

    vp = [(i, v) for i, v in zip(iters, pos) if v is not None and v > 0]
    vn = [(i, v) for i, v in zip(iters, neg) if v is not None and v > 0]
    if vp:
        xi, yp = zip(*vp)
        ax.plot(xi, yp, color="#27ae60", linewidth=1.5, label="Positive")
    if vn:
        xi, yn = zip(*vn)
        ax.plot(xi, yn, color="#e74c3c", linewidth=1.5,
                linestyle="--", label="|Negative|")

    ax.set_yscale("log")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white",
              edgecolor=GRID, framealpha=0.9)
    _style_ax(ax, "Regret per infoset per iteration (log)",
              ylabel="Regret / (infosets × iters)")


def plot_throughput(ax, rows):
    iters = [int(r["_cumulative_iters"]) for r in rows]
    tp    = [_safe(r.get("worker_iterations_per_second_mean")) for r in rows]
    valid = [(i, v) for i, v in zip(iters, tp) if v is not None and v > 0]
    if not valid:
        return
    xi, yv = zip(*valid)
    ax.plot(xi, yv, color="#f39c12", linewidth=1.5)
    ax.fill_between(xi, yv, alpha=0.12, color="#f39c12")
    _style_ax(ax, "Training throughput (mean per worker)",
              ylabel="Iterations / second")


def plot_exploitability_bound(ax, rows, max_payoff=20_000):
    """
    Remaining exploitability as % of the worst-case maximum.

    Formula: positive_regret / (total_strategy_mass × max_payoff) × 100
      - 0 %  = Nash equilibrium (no remaining regret)
      - 100 % = fully random strategy (maximum possible regret)

    Both positive_regret and total_strategy_mass accumulate as O(T²) in
    LinearMCCFR, so the ratio cancels that growth and decreases toward 0.
    max_payoff = full stack (20 000 chips = 200 bb) caps the scale.
    """
    iters  = [int(r["_cumulative_iters"]) for r in rows]
    pcts   = []
    for r in rows:
        pr  = _safe(r.get("positive_regret"))
        tsm = _safe(r.get("total_strategy_mass"))
        if pr is not None and pr > 0 and tsm and tsm > 0:
            pcts.append(pr / (tsm * max_payoff) * 100)
        else:
            pcts.append(None)

    valid = [(i, v) for i, v in zip(iters, pcts) if v is not None]
    if not valid:
        return
    xi, yv = zip(*valid)

    # Reference bands: actual exploitability ≈ regret_pct × max_payoff_bb100 / bound_factor
    # bound_factor ≈ 10–20 (MCCFR bounds are empirically this much looser than reality)
    # Player skill estimates (bb/100 exploitable vs Nash opponent):
    #   Pluribus/Libratus  ~1–3 bb/100   →  ~0.05–0.30 %
    #   Professional       ~10–30 bb/100  →  ~0.50–3.0  %
    #   Strong amateur     ~50–100 bb/100 →  ~2.5–10    %
    #   Recreational       ~200–500 bb/100 → ~10–50     %
    REFS = [
        (0.05,  0.30,  "#3498db", "Solver / Pluribus level  (~1–3 bb/100)"),
        (0.50,  3.0,   "#2ecc71", "Professional player       (~10–30 bb/100)"),
        (2.5,  10.0,   "#f39c12", "Strong amateur            (~50–100 bb/100)"),
        (10.0, 50.0,   "#e74c3c", "Recreational player       (~200–500 bb/100)"),
    ]
    for lo, hi, color, label in REFS:
        ax.axhspan(lo, hi, alpha=0.10, color=color, linewidth=0)
        ax.axhline(lo, color=color, linewidth=0.5, linestyle=":", alpha=0.6)

    ax.plot(xi, yv, color="#e67e22", linewidth=2, zorder=5)
    ax.fill_between(xi, yv, alpha=0.15, color="#e67e22", zorder=4)
    ax.axhline(0, color="#27ae60", linewidth=0.8, linestyle="--")

    # Right-side labels for the bands
    x_label = xi[-1] * 1.01
    for lo, hi, color, label in REFS:
        mid = (lo + hi) / 2
        ax.text(x_label, mid, label, color=color, fontsize=6,
                va="center", ha="left", clip_on=False)

    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    _style_ax(ax,
              "Remaining exploitability  (0 % = Nash,  100 % = random — log scale)",
              ylabel="% of max possible regret remaining")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", nargs="?",
                        default="checkpoints/fullgame_100bb/metrics.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_csv(args.metrics_csv)
    print(f"Loaded {len(rows)} rows from {args.metrics_csv}")

    fig = plt.figure(figsize=(14, 13))
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(4, 2, hspace=0.48, wspace=0.32,
                          left=0.07, right=0.97, top=0.94, bottom=0.04)

    plot_infoset_growth(fig.add_subplot(gs[0, 0]), rows)
    plot_entropy(fig.add_subplot(gs[0, 1]), rows)
    plot_action_distribution(fig.add_subplot(gs[1, :]), rows)
    plot_exploitability_bound(fig.add_subplot(gs[2, :]), rows)
    plot_regret(fig.add_subplot(gs[3, 0]), rows)
    plot_throughput(fig.add_subplot(gs[3, 1]), rows)

    fig.suptitle("MCCFR Training Dashboard", color="white",
                 fontsize=13, fontweight="bold", y=0.97)

    out = args.out
    if out is None:
        ckpt_dir = Path(args.metrics_csv).parent
        plots_dir = Path("plots") / ckpt_dir.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = str(plots_dir / "training_dashboard.png")

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
