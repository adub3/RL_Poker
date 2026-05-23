"""
Preflop strategy chart from a MCCFR checkpoint.

Each cell is filled with horizontal colour strips proportional to the
action probability distribution. Hand name is shown centred in the cell.

Usage:
    python tools/viz/preflop_chart.py [checkpoint.json.gz] [--pos P0|P1|both] [--seq open]
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

RANKS = list("AKQJT98765432")
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

# Bucket actions into 5 clean visual categories
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
    "fold":        "#c0392b",   # red
    "call":        "#7f8c8d",   # grey
    "raise_small": "#27ae60",   # green
    "raise_med":   "#f39c12",   # orange
    "raise_large": "#8e44ad",   # purple
    "jam":         "#2c3e50",   # near-black
}

BUCKET_ORDER = ["fold", "call", "raise_small", "raise_med", "raise_large", "jam"]
BUCKET_LABEL = {
    "fold": "Fold",
    "call": "Call / Limp",
    "raise_small": "Raise small (2-3bb)",
    "raise_med": "Raise medium (3-4x / 25%)",
    "raise_large": "Raise large (50%)",
    "jam": "Jam",
}


def hand_class(infoset):
    start = infoset.find("[PF:") + 4
    end = infoset.find("]", start)
    if start < 4 or end < 0:
        return None
    return infoset[start:end]


def cell_key(hc):
    if len(hc) == 2:
        r = RANK_IDX[hc[0]]
        return r, r
    r1, r2, suit = hc[0], hc[1], hc[2]
    i1, i2 = RANK_IDX[r1], RANK_IDX[r2]
    if suit == "s":
        row, col = max(i1, i2), min(i1, i2)
    else:
        row, col = min(i1, i2), max(i1, i2)
    return row, col


def load_strategies(path, pos_filter, seq_filter, tc_filter=None):
    with gzip.open(path, "rt") as f:
        data = json.load(f)

    accum = defaultdict(lambda: defaultdict(float))
    for key, node in data.items():
        if "[PF:" not in key:
            continue
        hc = hand_class(key)
        if not hc:
            continue
        if pos_filter != "both" and f"[pos:{pos_filter}]" not in key:
            continue
        if seq_filter and f"[seq:{seq_filter}]" not in key:
            continue
        if tc_filter and f"[tc:{tc_filter}]" not in key:
            continue
        ss = node.get("strategy_sum", {})
        total = sum(float(v) for v in ss.values())
        if total <= 0:
            continue
        for action, mass in ss.items():
            bucket = ACTION_BUCKET.get(action, "call")
            accum[hc][bucket] += float(mass)

    result = {}
    for hc, bm in accum.items():
        total = sum(bm.values())
        if total > 0:
            result[hc] = {b: v / total for b, v in bm.items()}
    return result


def draw_chart(strategies, title, out_path):
    N = 13
    cell = 1.0
    pad = 0.04
    fig_w = N * cell + 1.8   # extra for legend
    fig_h = N * cell + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")
    ax.axis("off")

    BG = "#16213e"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for hc, dist in strategies.items():
        try:
            row, col = cell_key(hc)
        except (KeyError, IndexError):
            continue

        x = col * cell
        y = (N - 1 - row) * cell

        # Cell background
        ax.add_patch(plt.Rectangle(
            (x + pad, y + pad), cell - 2*pad, cell - 2*pad,
            facecolor="#0f3460", edgecolor="#1a4a8a", linewidth=0.8, zorder=1,
        ))

        # Horizontal colour strips filling the cell
        strip_y = y + pad
        strip_h = cell - 2 * pad
        strip_x = x + pad
        strip_w = cell - 2 * pad
        offset = 0.0
        for bucket in BUCKET_ORDER:
            prob = dist.get(bucket, 0.0)
            if prob < 1e-4:
                continue
            w = prob * strip_w
            ax.add_patch(plt.Rectangle(
                (strip_x + offset, strip_y), w, strip_h,
                facecolor=BUCKET_COLOR[bucket], edgecolor="none", zorder=2,
            ))
            offset += w

        # Hand label centred in cell
        # Pairs → white, suited → light cyan, offsuit → light yellow
        if len(hc) == 2:
            txt_color = "white"
        elif hc.endswith("s"):
            txt_color = "#a8e6cf"
        else:
            txt_color = "#ffd3a5"

        ax.text(
            x + cell / 2, y + cell / 2, hc,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=txt_color,
            fontfamily="monospace", zorder=3,
        )

    # Rank labels — top row and left column
    for i, r in enumerate(RANKS):
        ax.text(i * cell + cell / 2, N + 0.12, r,
                ha="center", va="bottom", fontsize=11,
                color="white", fontweight="bold")
        ax.text(-0.18, (N - 1 - i) * cell + cell / 2, r,
                ha="right", va="center", fontsize=11,
                color="white", fontweight="bold")

    # "Suited" and "Offsuit" corner annotations
    ax.text(1.5, N - 1.2, "suited ▸", fontsize=8, color="#556677",
            ha="left", va="top", style="italic")
    ax.text(N - 1.5, 1.2, "◂ offsuit", fontsize=8, color="#556677",
            ha="right", va="bottom", style="italic")

    # Legend
    patches = [
        mpatches.Patch(facecolor=BUCKET_COLOR[b], label=BUCKET_LABEL[b],
                       edgecolor="#ffffff44")
        for b in BUCKET_ORDER
    ]
    leg = ax.legend(
        handles=patches, loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9, facecolor="#0f3460",
        labelcolor="white", edgecolor="#1a4a8a",
        framealpha=1.0, title="Action",
        title_fontsize=9,
    )
    leg.get_title().set_color("white")

    ax.set_title(title, color="white", fontsize=11, pad=8)
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# Per-position defaults: (seq_filter, tc_filter, display label)
POS_CONFIG = {
    "P0": ("open",  None,     "P0 · BB · first postflop · opens preflop last"),
    "P1": ("c",    "0_5bb",  "P1 · SB/BTN · first preflop action"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--pos", default="all",
                        choices=("P0", "P1", "both", "all"),
                        help="'all' generates one chart per position")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.checkpoint is None:
        candidates = sorted(Path("checkpoints/fullgame_100bb").glob("*.json.gz"))
        if not candidates:
            candidates = sorted(Path("checkpoints").rglob("*.json.gz"))
        if not candidates:
            sys.exit("No checkpoint found.")
        args.checkpoint = str(candidates[-1])
        print(f"Using: {args.checkpoint}")

    stem = Path(args.checkpoint).stem.replace("mccfr_table_", "")
    ckpt_dir = Path(args.checkpoint).parent
    plots_dir = Path("plots") / ckpt_dir.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    positions = ["P0", "P1"] if args.pos == "all" else [args.pos]

    for pos in positions:
        seq_f, tc_f, label = POS_CONFIG.get(pos, ("open", None, pos))
        out_path = args.out or str(
            plots_dir / f"{Path(args.checkpoint).stem}_chart_{pos}.png"
        )
        title = f"Preflop opening  ·  {label}\n{stem}"
        strategies = load_strategies(args.checkpoint, pos, seq_f, tc_filter=tc_f)
        print(f"[{pos}] Loaded {len(strategies)} hand classes")
        draw_chart(strategies, title, out_path)


if __name__ == "__main__":
    main()
