#!/usr/bin/env bash
# Install or update the systemd services. Safe to rerun; setup.sh calls it.
#
#   TRAIN_RUNS="v6c v6e" deploy/oracle/install_services.sh
#
# Each abstraction in TRAIN_RUNS trains as rl-poker-train@<name> in
# checkpoints/run_<name>, with the cores split evenly between runs and a
# memory cap per run (if a run outgrows it, only that run restarts from its
# last checkpoint). The eval timer measures every run's newest checkpoint.
set -euo pipefail

REPO="${REPO:-$HOME/RL_Poker}"
TRAIN_RUNS="${TRAIN_RUNS:-v6c v6e}"
read -r -a runs <<< "$TRAIN_RUNS"
count=${#runs[@]}
workers="${TRAIN_WORKERS:-$(( $(nproc) / count > 0 ? $(nproc) / count : 1 ))}"
total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
memory_max="${TRAIN_MEMORY_MAX:-$(( total_kb * 85 / 100 / count ))K}"
eval_runs=""
for run in "${runs[@]}"; do eval_runs+="run_$run "; done

sudo tee /etc/systemd/system/rl-poker-train@.service >/dev/null <<UNIT
[Unit]
Description=RL_Poker MCCFR training (%i)
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$REPO
Environment=RUN_DIR=$REPO/checkpoints/run_%i BET_SIZING=%i WORKERS=$workers
ExecStart=$REPO/deploy/oracle/train_forever.sh
MemoryMax=$memory_max
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/rl-poker-eval.service >/dev/null <<UNIT
[Unit]
Description=RL_Poker exploitability of each run's newest checkpoint

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$REPO
Environment="EVAL_RUNS=${eval_runs% }"
ExecStart=$REPO/deploy/oracle/eval_latest.sh
UNIT

sudo tee /etc/systemd/system/rl-poker-eval.timer >/dev/null <<UNIT
[Unit]
Description=Measure RL_Poker exploitability every 6 hours

[Timer]
OnBootSec=1h
OnUnitActiveSec=6h

[Install]
WantedBy=timers.target
UNIT

sudo tee /etc/systemd/system/rl-poker-tensorboard.service >/dev/null <<UNIT
[Unit]
Description=TensorBoard for RL_Poker

[Service]
User=$USER
WorkingDirectory=$REPO
# All runs under checkpoints/ (current and archived) show side by side.
ExecStart=$REPO/.venv/bin/tensorboard --logdir $REPO/checkpoints --host 127.0.0.1 --port 6006
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

# The single-run service from earlier setups is replaced by the template.
if [[ -f /etc/systemd/system/rl-poker-train.service ]]; then
  sudo systemctl disable --now rl-poker-train.service || true
  sudo rm /etc/systemd/system/rl-poker-train.service
fi

sudo systemctl daemon-reload
for run in "${runs[@]}"; do
  sudo systemctl enable --now "rl-poker-train@$run.service"
done
sudo systemctl enable --now rl-poker-eval.timer rl-poker-tensorboard.service
sudo systemctl restart rl-poker-tensorboard.service
echo "Training: ${runs[*]} ($workers workers each, memory cap $memory_max each)"
