#!/usr/bin/env bash
# One-time setup on a fresh Oracle Cloud Ubuntu VM (Ampere A1 / ARM or x86).
#
#   curl -fsSL https://raw.githubusercontent.com/adub3/RL_Poker/main/deploy/oracle/setup.sh | bash
#
# Installs Python 3.11 and the requirements, then starts the services
# (deploy/oracle/install_services.sh):
#   rl-poker-train@<name> one training run per abstraction in TRAIN_RUNS,
#                         resuming after any restart
#   rl-poker-eval.timer   measures exploitability (LBR) of the newest checkpoint every 6 hours
#   rl-poker-tensorboard  TensorBoard on 127.0.0.1:6006 (reach it with an SSH tunnel)
set -euo pipefail

REPO="$HOME/RL_Poker"

sudo apt-get update -y
sudo apt-get install -y git curl build-essential

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" pull --ff-only
else
  git clone https://github.com/adub3/RL_Poker.git "$REPO"
fi
cd "$REPO"

uv venv --python 3.11 .venv
# eval7 has no ARM wheel, and its build needs Cython without declaring it.
uv pip install --python .venv/bin/python cython setuptools wheel
uv pip install --python .venv/bin/python --no-build-isolation eval7
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import pyspiel; assert 'universal_poker' in pyspiel.registered_names(); print('universal_poker OK')"
.venv/bin/python -B new_code/test_ai_core.py
.venv/bin/python -B tools/evaluation/test_lbr.py

chmod +x deploy/oracle/*.sh

deploy/oracle/install_services.sh

cat <<DONE

Training is running. On your own computer:

  ssh -L 6006:localhost:6006 ubuntu@<this VM's public IP>

then open http://localhost:6006 for TensorBoard.

Logs:    journalctl -u 'rl-poker-train@*' -f
Status:  systemctl status 'rl-poker-train@*' rl-poker-eval.timer rl-poker-tensorboard
DONE
