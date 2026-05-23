"""
Strategy visualizations from MCCFR checkpoint.

  1. opening_range_heatmap.png  — 13×13 range grid coloured by action mix
  2. strategy_tree_AK.png       — AKo/AKs decision path for key betting lines
  3. hand_strategy_bars.png     — action breakdown for selected hands

Usage:
  .venv/bin/python tools/analysis/plot_strategy_charts.py \
      --checkpoint checkpoints/fullgame_100bb/mccfr_table_iter_00815000.json.gz
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_code"))
from ai import load_table  # noqa: E402

# ── Colour scheme ────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"

C = {
    "fold":        "#da3633",
    "call":        "#388bfd",
    "small_raise": "#3fb950",
    "mid_raise":   "#a5d6a7",
    "big_raise":   "#f78166",
    "jam":         "#bc8cff",
}
LABELS = {
    "fold": "fold", "call": "call/limp",
    "small_raise": "2–3bb", "mid_raise": "3–4x",
    "big_raise": "big raise", "jam": "jam",
}
SMALL_RAISES = {"min_raise", "open_2_5bb", "open_3bb", "raise_2_5bb", "raise_3bb"}
MID_RAISES   = {"raise_3x", "raise_4x", "raise_4bb"}
BIG_RAISES   = {"raise_25eff", "raise_50eff"}

RANKS = list("AKQJT98765432")


def group(strategy: dict) -> dict:
    g = {k: 0.0 for k in C}
    for a, p in strategy.items():
        if a in ("fold", "check"):  g["fold"] += p
        elif a == "call":           g["call"] += p
        elif a in SMALL_RAISES:     g["small_raise"] += p
        elif a in MID_RAISES:       g["mid_raise"] += p
        elif a in BIG_RAISES:       g["big_raise"] += p
        elif a == "jam":            g["jam"] += p
    return g


def hand_cell(hand: str):
    if len(hand) == 2 and hand[0] == hand[1]:
        i = RANKS.index(hand[0]); return i, i
    r1, r2 = RANKS.index(hand[0]), RANKS.index(hand[1])
    if hand.endswith("s"): return min(r1,r2), max(r1,r2)
    return max(r1,r2), min(r1,r2)


def cell_label(r, c):
    if r == c: return RANKS[r]*2
    if c > r:  return f"{RANKS[r]}{RANKS[c]}s"
    return f"{RANKS[c]}{RANKS[r]}o"


# ── 1. Opening range heatmap ─────────────────────────────────────────────────

def plot_heatmap(data: dict, out: Path):
    opening = {k: v for k, v in data.items()
               if "[pos:P0]" in k and k.endswith("[seq:open]")}

    grid = [[None]*13 for _ in range(13)]
    for key, node in opening.items():
        hand = key.split("[PF:")[1].split("]")[0]
        r, c = hand_cell(hand)
        grid[r][c] = group(node["strategy"])

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ORDER = ["fold","call","small_raise","mid_raise","big_raise","jam"]

    for r in range(13):
        for c in range(13):
            g = grid[r][c]
            # Cell background
            bg = mpatches.Rectangle((c, r), 1, 1, facecolor=BG2, edgecolor=BORDER,
                                     linewidth=0.6, zorder=1)
            ax.add_patch(bg)
            if g is None:
                continue

            # Horizontal stacked bar (bottom 35% of cell)
            bar_bottom = r + 0.07
            bar_h      = 0.30
            bar_x      = c + 0.07
            bar_w      = 0.86
            left = bar_x
            for act in ORDER:
                w = g[act] * bar_w
                if w < 0.003: continue
                ax.add_patch(mpatches.Rectangle(
                    (left, bar_bottom), w, bar_h,
                    facecolor=C[act], zorder=3, linewidth=0))
                left += w

            # Hand label (top of cell)
            ax.text(c + 0.5, r + 0.82, cell_label(r, c),
                    ha="center", va="center", fontsize=7.2,
                    color=TEXT, fontweight="bold", zorder=4)

            # Dominant action % (below bar)
            dom = max(g, key=g.get)
            ax.text(c + 0.5, r + 0.49, f"{g[dom]*100:.0f}%",
                    ha="center", va="center", fontsize=6.2,
                    color=C[dom], zorder=4)

    ax.set_xlim(0, 13); ax.set_ylim(0, 13)
    ax.set_xticks(np.arange(13)+0.5); ax.set_yticks(np.arange(13)+0.5)
    ax.set_xticklabels(RANKS, color=TEXT, fontsize=10, fontweight="bold")
    ax.set_yticklabels(RANKS, color=TEXT, fontsize=10, fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

    # "suited" / "offsuit" annotations
    ax.text(9.5, 13.6, "suited →", color=SUBTEXT, fontsize=9, ha="center")
    ax.text(-0.7, 3.5, "← offsuit", color=SUBTEXT, fontsize=9,
            ha="center", rotation=90)

    # Legend
    patches = [mpatches.Patch(color=C[k], label=LABELS[k]) for k in ORDER]
    ax.legend(handles=patches, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), fontsize=9,
              facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)

    ax.set_title("P0 Opening Strategy — 100bb  (MCCFR 815k iterations)\n"
                 "Bar = action mix · % = dominant action frequency",
                 color=TEXT, fontsize=12, pad=14, linespacing=1.5)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {out}")


# ── 2. AK strategy tree ──────────────────────────────────────────────────────

NODE_W, NODE_H = 2.6, 1.1

def node(ax, cx, cy, title, strategy, accent=TEXT):
    g = group(strategy)
    ORDER = ["fold","call","small_raise","mid_raise","big_raise","jam"]

    # Card / box
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - NODE_W/2, cy - NODE_H/2), NODE_W, NODE_H,
        boxstyle="round,pad=0.08", facecolor=BG2, edgecolor="#30363d",
        linewidth=1.2, zorder=3))

    # Title
    ax.text(cx, cy + NODE_H/2 - 0.17, title,
            ha="center", va="top", fontsize=8, color=accent,
            fontweight="bold", zorder=5)

    # Stacked bar (horizontal, middle of box)
    bx, by, bw, bh = cx - NODE_W/2 + 0.15, cy - 0.05, NODE_W - 0.3, 0.24
    left = bx
    for act in ORDER:
        w = g[act] * bw
        if w < 0.003: continue
        ax.add_patch(mpatches.Rectangle((left, by), w, bh,
                     facecolor=C[act], zorder=5, linewidth=0))
        left += w

    # Key action labels below bar
    parts = [f"{LABELS[a]} {g[a]*100:.0f}%"
             for a in ORDER if g[a] >= 0.09]
    ax.text(cx, by - 0.06, "  ".join(parts[:3]),
            ha="center", va="top", fontsize=6.3, color=SUBTEXT, zorder=5)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2 + NODE_H/2 + 0.04),
                xytext=(x1, y1 - NODE_H/2 - 0.04),
                arrowprops=dict(arrowstyle="-|>", color="#4a6fa5",
                                lw=1.4, mutation_scale=10),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label, fontsize=7.5,
                color="#6e9fcc", va="center", style="italic")


def plot_ak_tree(data: dict, out: Path):
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(BG)

    for col, (hand, label) in enumerate([("AKo", "AK offsuit"), ("AKs", "AK suited")]):
        ax = fig.add_subplot(1, 2, col+1)
        ax.set_facecolor(BG)
        ax.set_xlim(0, 10); ax.set_ylim(0, 13)
        ax.axis("off")
        ax.set_title(f"{label}", color=TEXT, fontsize=13, fontweight="bold", pad=10)

        def get(pos, tc, seq):
            key = f"[PF:{hand}][pos:{pos}][tc:{tc}][eff:100bb][seq:{seq}]"
            nd = data.get(key)
            return nd["strategy"] if nd else None

        Y = [11.5, 9.0, 6.5, 4.0, 1.5]  # row y-centres

        # Row 0 — P0 opening
        s = get("P0", "0bb", "open")
        if s: node(ax, 5, Y[0], "P0 — first to act  [seq:open]", s, accent="#f0e040")

        # Row 1 — P1 faces open  |  P1 faces limp
        s1 = get("P1", "2bb", "r:2_5bb")
        if s1:
            node(ax, 2.5, Y[1], "P1 faces 2.5bb open", s1, accent="#58a6ff")
            if s: arrow(ax, 5, Y[0], 2.5, Y[1], "P0 raises 2.5bb")

        s2 = get("P1", "0_5bb", "c")
        if s2:
            node(ax, 7.5, Y[1], "P1 (BB) faces limp", s2, accent="#58a6ff")
            if s: arrow(ax, 5, Y[0], 7.5, Y[1], "P0 limps")

        # Row 2 — P0 faces 3-bet
        s3 = get("P0", "4bb", "r:2_5bb|r:3x")
        if s3:
            node(ax, 1.8, Y[2], "P0 faces 3-bet  (3x)", s3, accent="#f0e040")
            if s1: arrow(ax, 2.5, Y[1], 1.8, Y[2], "P1 3-bets 3x")

        s4 = get("P0", "100bb", "r:2_5bb|r:jam")
        if s4:
            node(ax, 4.6, Y[2], "P0 faces 3-bet jam", s4, accent="#f0e040")
            if s1: arrow(ax, 2.5, Y[1], 4.6, Y[2], "P1 jams")

        # Row 3 — P1 faces 4-bet
        s5 = get("P1", "100bb", "r:2_5bb|r:3x|r:jam")
        if s5:
            node(ax, 1.8, Y[3], "P1 faces 4-bet jam", s5, accent="#58a6ff")
            if s3: arrow(ax, 1.8, Y[2], 1.8, Y[3], "P0 jams")

        s6 = get("P1", "25bb", "r:2_5bb|r:3x|r:25eff")
        if s6:
            node(ax, 4.6, Y[3], "P1 faces 4-bet 25bb", s6, accent="#58a6ff")
            if s3: arrow(ax, 1.8, Y[2], 4.6, Y[3], "P0 raises 25eff")

        # Legend (bottom right)
        patches = [mpatches.Patch(color=C[k], label=LABELS[k])
                   for k in ["fold","call","small_raise","mid_raise","big_raise","jam"]]
        ax.legend(handles=patches, loc="lower right", fontsize=8,
                  facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)

    fig.suptitle("AK Decision Tree — 100bb Preflop  (MCCFR 815k iters)",
                 color=TEXT, fontsize=15, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {out}")


# ── 3. Hand strategy bars ────────────────────────────────────────────────────

def plot_bars(data: dict, out: Path):
    hands = ["AA","KK","QQ","JJ","TT","99","88",
             "AKs","AKo","AQs","AQo","AJs","KQs",
             "22","72o","32o"]
    ORDER = ["fold","call","small_raise","mid_raise","big_raise","jam"]

    rows = {}
    for h in hands:
        key = f"[PF:{h}][pos:P0][tc:0bb][eff:100bb][seq:open]"
        nd = data.get(key)
        if nd: rows[h] = group(nd["strategy"])

    labels = list(rows.keys())
    n = len(labels)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bottoms = np.zeros(n)
    for act in ORDER:
        vals = np.array([rows[h][act] for h in labels])
        ax.bar(x, vals, 0.74, bottom=bottoms, color=C[act],
               label=LABELS[act], zorder=3)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.08:
                ax.text(x[i], b + v/2, f"{v*100:.0f}",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=4)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10.5, fontweight="bold")
    ax.set_yticks([0,.25,.5,.75,1])
    ax.set_yticklabels(["0%","25%","50%","75%","100%"], color=SUBTEXT, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, zorder=0)

    ax.set_title("P0 Opening Action Mix by Hand — 100bb  (MCCFR 815k iters)",
                 color=TEXT, fontsize=12, pad=12)
    ax.set_ylabel("Strategy probability", color=SUBTEXT, fontsize=10)
    ax.legend(loc="upper right", fontsize=9,
              facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)

    # Divider between premiums and bluffs
    ax.axvline(12.5, color=BORDER, linewidth=1, linestyle="--", zorder=1)
    ax.text(13.0, 1.02, "bluffs →", color=SUBTEXT, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/fullgame_100bb/mccfr_table_iter_00815000.json.gz")
    ap.add_argument("--output-dir", default="assets/diagrams/strategy_815k")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint} …")
    t = load_table(args.checkpoint)
    print(f"  {len(t.data):,} infosets")

    plot_heatmap(t.data, out / "opening_range_heatmap.png")
    plot_ak_tree(t.data, out / "strategy_tree_AK.png")
    plot_bars(t.data, out / "hand_strategy_bars.png")
    print(f"\nDone → {out}/")


if __name__ == "__main__":
    main()
