# Poker MCCFR Bot

This repository contains a poker AI prototype focused on abstraction and Monte
Carlo Counterfactual Regret Minimization.

## Layout

- `new_code/` - active implementation.
  - `ai.py` - Linear MCCFR trainer and strategy table.
  - `abstraction.py` - card and betting abstraction helpers.
  - `skeleton.py` - pokerbot integration that loads an exported strategy.
  - `simulation.py` - interactive simulation helper.
  - `test_ai_core.py` - lightweight tests for MCCFR table mechanics.
- `docs/` - papers and reference material.
- `assets/` - diagrams and images.
- `data/` - generated data files.
- `tools/visualization/` - scripts that generate diagrams.
- `archive/` - older prototypes and legacy experiments kept for reference.

## Training on Oracle Cloud

1. In Oracle Cloud, create a VM: Ubuntu 22.04 or 24.04, shape
   `VM.Standard.A1.Flex` (Always Free covers up to 4 OCPUs and 24 GB RAM),
   and add your SSH public key.
2. SSH in and run:

   ```sh
   curl -fsSL https://raw.githubusercontent.com/adub3/RL_Poker/main/deploy/oracle/setup.sh | bash
   ```

   This installs everything and starts continuous training, an exploitability
   check of the newest checkpoint every 6 hours, and TensorBoard.
3. To watch it, from your own computer:

   ```sh
   ssh -L 6006:localhost:6006 ubuntu@<VM public IP>
   ```

   and open http://localhost:6006. TensorBoard listens only on the VM itself;
   the SSH tunnel is what lets you in.

Training resumes from the newest checkpoint after any restart and keeps the 6
newest checkpoints. Memory grows with the table (each worker holds a full
copy), so watch `system/memory_available_gb`; if it gets low, lower `WORKERS`
in `deploy/oracle/train_forever.sh`. To copy checkpoints home:

```sh
rsync -av ubuntu@<VM public IP>:RL_Poker/checkpoints/fullgame_100bb/ checkpoints/fullgame_100bb/
```

## Setup

The project is currently pinned to Python 3.11.9 with `pyenv-win`.

```powershell
pip install -r requirements.txt
```

On macOS, `universal_poker` works from the pip wheel:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Checks

```powershell
python -B new_code\test_ai_core.py
python -B tools\evaluation\test_best_response.py
```

## Measuring exploitability

```sh
.venv/bin/python tools/evaluation/mc_exploitability.py --br-iters 10000 --eval-games 5000
```

For each seat, a best responder learns action values against the frozen
blueprint, then plays its best action on fresh hands. The result is a lower
bound on exploitability within the bot's own abstraction, reported in bb/100
with a 95% interval. It also reports how often the blueprint had no entry for a
decision (those are played uniformly at random, like the live bot). More
`--br-iters` tightens the bound; more `--eval-games` narrows the interval.
`tools/evaluation/test_best_response.py` checks the estimator against exact
exploitability on Kuhn and Leduc poker.

## Rules versions

Each run's `run_manifest.json` records the rules it was trained under, and
evaluation, play, charts and `--resume-from` all load them from there, since a
table only lines up with the rules it was trained with. Existing runs keep
working unchanged; start a new run to get the current rules.

- Seat order (`game_config.firstPlayer`): P0 posts the big blind and P1 the
  small blind. Current runs use `"2 1 1 1"`: the small blind acts first
  preflop and the big blind first on later streets. Older runs used `"1"`,
  where the big blind acted first on every street; runs with no manifest are
  assumed to be those.
- Bet sizing (`bet_sizing`): `v2`, the current default, sizes pot-fraction
  bets on top of the chips already committed, and `min_raise` is the real
  minimum on every street. `legacy` is the sizing used before that fix; runs
  without a `bet_sizing` entry used it.

## Training metrics

`metrics.csv` includes `avg_regret_bound`: the sum over infosets of each
infoset's largest positive regret, divided by the total iteration weight. In
exact CFR on a perfect-recall game this bounds twice the exploitability. Here
regrets are sampled and the card abstraction forgets earlier streets, so treat
it as a convergence signal (it should trend down), not a guarantee.
