"""
Strategy convergence analysis across MCCFR checkpoints.

Produces two outputs:
  1. convergence_<dir>.png  — L1 distance between consecutive checkpoints (per position)
  2. evolution_<pos>_<dir>.png — grid of preflop charts at sampled milestones

Usage:
    python tools/viz/convergence.py [checkpoint_dir] [--pos P0|P1|all]
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_code"))

from ai import game_config_for_checkpoint  # noqa: E402
from preflop import first_preflop_spots  # noqa: E402

RANKS = list("AKQJT98765432")
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

ACTION_BUCKET = {
    "fold":        "fold",
    "call":        "call",
    "min_raise":   "raise_small",
    "open_2_5bb":  "raise_small",
    "open_3bb":    "raise_small",
    "raise_2_5bb": "raise_small",
    "raise_3bb":   "raise_small",
    "raise_3x":    "raise_med",
    "raise_4x":    "raise_med",
    "raise_25eff": "raise_med",
    "raise_50eff": "raise_large",
    "jam":         "jam",
}

BUCKET_COLOR = {
    "fold":        "#c0392b",
    "call":        "#7f8c8d",
    "raise_small": "#27ae60",
    "raise_med":   "#f39c12",
    "raise_large": "#8e44ad",
    "jam":         "#2c3e50",
}
BUCKET_ORDER = ["fold", "call", "raise_small", "raise_med", "raise_large", "jam"]

# {seat: (seq_filter, tc_filter, label)}; set in main() from the run's seat order.
POS_CONFIG = {}

BG = "#16213e"


def hand_class(infoset):
    s = infoset.find("[PF:") + 4
    e = infoset.find("]", s)
    return infoset[s:e] if s >= 4 and e >= 0 else None


def cell_key(hc):
    if len(hc) == 2:
        r = RANK_IDX[hc[0]]
        return r, r
    r1, r2, suit = hc[0], hc[1], hc[2]
    i1, i2 = RANK_IDX[r1], RANK_IDX[r2]
    if suit == "s":
        return max(i1, i2), min(i1, i2)
    return min(i1, i2), max(i1, i2)


def load_strategies(path, pos, seq_f, tc_f):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    accum = defaultdict(lambda: defaultdict(float))
    for key, node in data.items():
        if "[PF:" not in key:
            continue
        hc = hand_class(key)
        if not hc:
            continue
        if f"[pos:{pos}]" not in key:
            continue
        if seq_f and f"[seq:{seq_f}]" not in key:
            continue
        if tc_f and f"[tc:{tc_f}]" not in key:
            continue
        ss = node.get("strategy_sum", {})
        total = sum(float(v) for v in ss.values())
        if total <= 0:
            continue
        for action, mass in ss.items():
            accum[hc][ACTION_BUCKET.get(action, "call")] += float(mass)
    result = {}
    for hc, bm in accum.items():
        total = sum(bm.values())
        if total > 0:
            result[hc] = {b: v / total for b, v in bm.items()}
    return result


def strategy_l1(s1, s2):
    """Average per-hand L1 distance between two strategy dicts."""
    hands = set(s1) & set(s2)
    if not hands:
        return 0.0
    total = 0.0
    for hc in hands:
        d1 = s1[hc]
        d2 = s2[hc]
        buckets = set(d1) | set(d2)
        total += sum(abs(d1.get(b, 0) - d2.get(b, 0)) for b in buckets)
    return total / len(hands)


def draw_mini_chart(ax, strategies, title, show_labels=True):
    N = 13
    cell = 1.0
    pad = 0.03

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    for hc, dist in strategies.items():
        try:
            row, col = cell_key(hc)
        except (KeyError, IndexError):
            continue
        x = col * cell
        y = (N - 1 - row) * cell

        ax.add_patch(plt.Rectangle(
            (x + pad, y + pad), cell - 2*pad, cell - 2*pad,
            facecolor="#0f3460", edgecolor="none", linewidth=0, zorder=1,
        ))

        offset = 0.0
        strip_w = cell - 2 * pad
        for bucket in BUCKET_ORDER:
            prob = dist.get(bucket, 0.0)
            if prob < 1e-4:
                continue
            w = prob * strip_w
            ax.add_patch(plt.Rectangle(
                (x + pad + offset, y + pad), w, cell - 2*pad,
                facecolor=BUCKET_COLOR[bucket], edgecolor="none", zorder=2,
            ))
            offset += w

    if show_labels:
        for i, r in enumerate(RANKS):
            ax.text(i + 0.5, N + 0.05, r, ha="center", va="bottom",
                    fontsize=5, color="white", fontweight="bold")
            ax.text(-0.1, (N - 1 - i) + 0.5, r, ha="right", va="center",
                    fontsize=5, color="white", fontweight="bold")

    ax.set_title(title, color="white", fontsize=7, pad=3)


def plot_convergence(checkpoints, pos_list, out_path):
    fig, axes = plt.subplots(len(pos_list), 1,
                             figsize=(10, 3.5 * len(pos_list)),
                             squeeze=False)
    fig.patch.set_facecolor(BG)

    for ax_row, pos in zip(axes, pos_list):
        ax = ax_row[0]
        seq_f, tc_f, label = POS_CONFIG[pos]

        iters = []
        l1s = []
        prev = None
        for path in checkpoints:
            strats = load_strategies(path, pos, seq_f, tc_f)
            m = int(Path(path).stem.split("iter_")[1].split(".")[0])
            if prev is not None:
                l1 = strategy_l1(prev, strats)
                iters.append(m)
                l1s.append(l1)
            prev = strats

        ax.set_facecolor("#0d1b2a")
        ax.plot(iters, l1s, color="#27ae60", linewidth=2, marker="o",
                markersize=4, markerfacecolor="#2ecc71")
        ax.fill_between(iters, l1s, alpha=0.15, color="#27ae60")
        ax.set_xlabel("Iterations (log scale)", color="white", fontsize=10)
        ax.set_ylabel("Avg L1 distance (consecutive ckpts)", color="white", fontsize=10)
        ax.set_title(f"Strategy convergence — {label}", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334455")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{int(x)//1_000_000}M" if x >= 1e6 else f"{int(x)//1000}k"))
        ax.grid(color="#334455", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_evolution(checkpoints, pos, out_path, n_milestones=8):
    seq_f, tc_f, pos_label = POS_CONFIG[pos]

    # Pick evenly-spaced milestones
    indices = np.linspace(0, len(checkpoints) - 1, min(n_milestones, len(checkpoints)), dtype=int)
    selected = [checkpoints[i] for i in indices]

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.8, rows * 3.8))
    fig.patch.set_facecolor(BG)

    for idx, path in enumerate(selected):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.set_facecolor(BG)
        strats = load_strategies(path, pos, seq_f, tc_f)
        iters = int(Path(path).stem.split("iter_")[1].split(".")[0])
        draw_mini_chart(ax, strats, f"{iters//1000}k iters")

    # Legend
    patches = [mpatches.Patch(facecolor=BUCKET_COLOR[b],
                               label=b.replace("_", " ").title(),
                               edgecolor="#ffffff33")
               for b in BUCKET_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=len(BUCKET_ORDER),
               fontsize=8, facecolor="#0f3460", labelcolor="white",
               edgecolor="#1a4a8a", framealpha=1.0,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Strategy evolution — {pos_label}", color="white",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", nargs="?",
                        default="checkpoints/fullgame_100bb")
    parser.add_argument("--pos", default="all",
                        choices=("P0", "P1", "all"))
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("*.json.gz"))
    if not checkpoints:
        sys.exit(f"No checkpoints found in {ckpt_dir}")
    print(f"Found {len(checkpoints)} checkpoints: "
          f"{Path(checkpoints[0]).stem} → {Path(checkpoints[-1]).stem}")

    POS_CONFIG.update(first_preflop_spots(game_config_for_checkpoint(ckpt_dir)))
    pos_list = ["P0", "P1"] if args.pos == "all" else [args.pos]
    dir_tag = ckpt_dir.name

    plots_dir = Path("plots") / dir_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Convergence plot (both positions on same figure)
    conv_path = plots_dir / f"convergence_{dir_tag}.png"
    plot_convergence(checkpoints, pos_list, conv_path)

    # Evolution grid per position
    for pos in pos_list:
        evo_path = plots_dir / f"evolution_{pos}_{dir_tag}.png"
        plot_evolution(checkpoints, pos, evo_path)


if __name__ == "__main__":
    main()
