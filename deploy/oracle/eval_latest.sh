#!/usr/bin/env bash
# Measure exploitability of each run's newest checkpoint with local best
# response, at low CPU priority. Run by the rl-poker-eval systemd timer.
# Results go to <run dir>/exploitability_lbr.json and <run dir>/tensorboard.
set -euo pipefail

REPO="${REPO:-$HOME/RL_Poker}"
# Run folders under checkpoints/, space-separated.
EVAL_RUNS="${EVAL_RUNS:-fullgame_100bb}"
LBR_HANDS="${LBR_HANDS:-2000}"
LBR_WORKERS="${LBR_WORKERS:-2}"

cd "$REPO"
for run in $EVAL_RUNS; do
  run_dir="$REPO/checkpoints/$run"
  # Newest checkpoint whose strategy DB training already wrote (exporting a
  # large checkpoint here would reload the whole table); else the newest.
  latest=""
  for ckpt in $(ls "$run_dir"/mccfr_table_iter_*.json.gz 2>/dev/null | sort -r); do
    iters="$(basename "$ckpt" | sed -E 's/mccfr_table_iter_([0-9]+).*/\1/')"
    if [[ -f "$run_dir/strategies/strategy_${iters}.db" ]]; then
      latest="$ckpt"
      break
    fi
  done
  if [[ -z "$latest" ]]; then
    latest="$(ls "$run_dir"/mccfr_table_iter_*.json.gz 2>/dev/null | sort | tail -n 1 || true)"
  fi
  if [[ -z "$latest" ]]; then
    echo "No checkpoints yet in $run_dir"
    continue
  fi
  # The script skips checkpoints it has already evaluated with these settings.
  nice -n 19 .venv/bin/python -u tools/evaluation/lbr_exploitability.py \
    --checkpoint "$latest" \
    --hands "$LBR_HANDS" \
    --workers "$LBR_WORKERS" \
    --out "$run_dir/exploitability_lbr.json" || echo "LBR failed for $run"
done
