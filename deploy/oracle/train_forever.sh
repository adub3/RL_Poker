#!/usr/bin/env bash
# Train continuously, resuming from the newest checkpoint in RUN_DIR each time.
# Run by the rl-poker-train systemd service; safe to restart at any point
# (at most the iterations since the last checkpoint are lost).
set -euo pipefail

REPO="${REPO:-$HOME/RL_Poker}"
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/fullgame_100bb}"
# Leave one core for evaluation, TensorBoard and the system. Workers sync in
# lockstep, so one slowed-down worker would hold up all of them.
WORKERS="${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
ITERATIONS_PER_WORKER="${ITERATIONS_PER_WORKER:-100000}"
MERGE_EVERY="${MERGE_EVERY:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-400}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-6}"

cd "$REPO"
mkdir -p "$RUN_DIR"

while true; do
  latest="$(ls "$RUN_DIR"/mccfr_table_iter_*.json.gz 2>/dev/null | sort | tail -n 1 || true)"
  resume=()
  if [[ -n "$latest" ]]; then
    resume=(--resume-from "$latest")
    echo "$(date '+%Y-%m-%d %H:%M:%S') resuming from $latest"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') starting a new run in $RUN_DIR"
  fi

  .venv/bin/python -u tools/training/train_preflop_parallel.py \
    --workers "$WORKERS" \
    --iterations-per-worker "$ITERATIONS_PER_WORKER" \
    --merge-every "$MERGE_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --output-dir "$RUN_DIR" \
    ${resume[@]+"${resume[@]}"}

  # Keep the newest checkpoints (and their exported strategy DBs). Results for
  # older ones are already in exploitability_greedy_br.json and TensorBoard.
  checkpoints=()
  while IFS= read -r path; do checkpoints+=("$path"); done \
    < <(ls "$RUN_DIR"/mccfr_table_iter_*.json.gz | sort)
  for (( i = 0; i < ${#checkpoints[@]} - KEEP_CHECKPOINTS; i++ )); do
    old="${checkpoints[$i]}"
    iters="$(basename "$old" | sed -E 's/mccfr_table_iter_([0-9]+).*/\1/')"
    rm -f "$old" "$RUN_DIR/strategies/strategy_${iters}.db"
  done
done
