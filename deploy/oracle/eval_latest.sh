#!/usr/bin/env bash
# Measure exploitability of the newest checkpoint with local best response,
# at low CPU priority. Run by the rl-poker-eval systemd timer. Results go to
# RUN_DIR/exploitability_lbr.json and RUN_DIR/tensorboard.
set -euo pipefail

REPO="${REPO:-$HOME/RL_Poker}"
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/fullgame_100bb}"
LBR_HANDS="${LBR_HANDS:-2000}"
LBR_WORKERS="${LBR_WORKERS:-2}"

cd "$REPO"
latest="$(ls "$RUN_DIR"/mccfr_table_iter_*.json.gz 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$latest" ]]; then
  echo "No checkpoints yet in $RUN_DIR"
  exit 0
fi

# The script skips checkpoints it has already evaluated with these settings.
exec nice -n 19 .venv/bin/python -u tools/evaluation/lbr_exploitability.py \
  --checkpoint "$latest" \
  --hands "$LBR_HANDS" \
  --workers "$LBR_WORKERS" \
  --out "$RUN_DIR/exploitability_lbr.json"
