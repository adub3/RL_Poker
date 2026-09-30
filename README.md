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

Training runs on an Oracle Cloud Infrastructure (OCI) "Always Free" server:
an Ampere ARM machine with up to 4 cores and 24 GB of RAM at no cost. Check
the current terms at oracle.com/cloud/free. A few catches:

- Signup needs a credit card to verify your identity. You aren't charged as
  long as you stay on the free tier and don't upgrade to Pay As You Go.
- Free ARM servers are sometimes sold out ("Out of host capacity"). Try
  another availability domain, or try again later.
- Oracle can reclaim free servers that sit idle. Continuous training keeps it
  busy.

### 1. Create an account

Sign up at oracle.com/cloud/free. Choose your home region carefully; it can't
be changed later.

### 2. Get your SSH key

On your own computer (create a key only if you don't have one):

```sh
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the line it prints.

### 3. Create the server

In the Oracle console, go to **Compute → Instances → Create instance**:

- **Image:** Canonical Ubuntu 22.04 (24.04 works too).
- **Shape:** Change shape → **Ampere** → `VM.Standard.A1.Flex`, with
  **4 OCPUs** and **24 GB** memory.
- **Networking:** keep the defaults; make sure it assigns a public IPv4 address.
- **SSH keys:** "Paste public keys", then paste the line from step 2.

Click **Create**. When it shows "Running", copy its **Public IP address**.

### 4. Log in and run the setup

```sh
ssh ubuntu@<PUBLIC_IP>
curl -fsSL https://raw.githubusercontent.com/adub3/RL_Poker/main/deploy/oracle/setup.sh | bash
```

This takes a few minutes. It installs Python and the requirements, runs the
tests, and starts three services:

- `rl-poker-train`: trains continuously and resumes from the newest
  checkpoint after any restart. It keeps the 6 newest checkpoints.
- `rl-poker-eval.timer`: measures exploitability of the newest checkpoint
  with local best response every 6 hours, at low priority.
- `rl-poker-tensorboard`: TensorBoard on port 6006 of the server.

### 5. Open TensorBoard

From a new terminal on your own computer:

```sh
ssh -L 6006:localhost:6006 ubuntu@<PUBLIC_IP>
```

Leave it open and go to http://localhost:6006. TensorBoard has no login, so it
only listens on the server itself; the SSH tunnel is what lets you in.

Charts to watch:

- `convergence/avg_regret_bound`: should trend down.
- `exploitability/lbr_lower_bound95`: the bot is at least this exploitable,
  with 95% confidence. First point an hour or two after boot, then every 6
  hours. Should trend down; see "Measuring exploitability".
- `system/memory_available_gb`: memory grows with the table, since each
  worker holds a full copy. If it nears 0, set a lower `WORKERS` in
  `deploy/oracle/train_forever.sh` and restart training.

### Useful commands (on the server)

```sh
journalctl -u rl-poker-train -f        # live training log
systemctl status rl-poker-train        # is it running?
sudo systemctl restart rl-poker-train  # restart (resumes from the newest checkpoint)
```

To copy checkpoints to your own computer (run from the repo there):

```sh
rsync -av ubuntu@<PUBLIC_IP>:RL_Poker/checkpoints/fullgame_100bb/ checkpoints/fullgame_100bb/
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
.venv/bin/python tools/evaluation/lbr_exploitability.py --hands 2000 --workers 4
```

Local best response (LBR; Lisý & Bowling, 2017) plays the real game against
the frozen blueprint, with no abstraction and no training of its own. At each
decision it tracks the bot's range (a weight for each of the 1,326 hole-card
combos, updated from the blueprint's action probabilities), computes its
showdown equity against that range, and picks the action with the best value
assuming the hand is checked or called down afterwards; for raises this uses
the blueprint's fold probability. It only uses information its own seat could
know, so its winnings are a lower bound on exploitability in the full game,
reported in bb/100 with a 95% interval.

Several variants play the same duplicate deals (each deal from both seats with
the same cards) and differ in the raise sizes they consider:

- `fcpa`: pot and all-in.
- `fc_half_pot_2pot`: half pot, pot, 2x pot and all-in.
- `bucket_edges`: the smallest and largest raise the bot reads as each of its
  bet-size buckets, plus all-in. This finds leaks in how the bot maps sizes it
  didn't train on.

Two headline numbers: `exploitability_bb100` is the best variant's mean, and
`lower_bound95_bb100` is the largest variant mean minus its standard error
times a Bonferroni-corrected z, so with 95% confidence the bot is at least that
exploitable (picking the best of several noisy means alone biases it up).
Hands where both players are all-in are scored by their expected result over
the remaining board cards rather than the one board dealt, which is unbiased
and roughly halves the interval; `--no-allin-ev` turns this off.

The bot reads raises it never
trained on through action translation (`new_code/translation.py`): each
off-menu raise counts as a mix of the two nearest menu sizes, weighted by the
pseudo-harmonic mapping, instead of a betting line with no table entry.
`--no-translation` turns this off. `blueprint missing` is how often the bot
still hit a spot with no table entry and played uniformly at random.
`tools/evaluation/test_lbr.py` checks that the range uses the bot's exact
infoset keys, that decisions never depend on the bot's cards, the range
update, the equity calculation, and that LBR crushes a uniformly random bot.

The older greedy best response (`mc_exploitability.py`) learns a responder
inside the bot's own abstraction, so it shares the bot's blind spots and can
come out negative with too few `--br-iters`.

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
- Bet sizing (`bet_sizing`):
  - `v3`: v2's sizes with raise caps. After 4
    preflop raises or 3 raises on a postflop street, a player can only fold,
    call or jam, and facing a postflop bet the raise options are pot-sized or
    jam. Without caps, min-raise wars made up most of the v2 tree; v3 trains
    about 5x faster and its table grows about 4x more slowly.
  - `v2`: pot-fraction bets are sized on top of the chips already committed,
    and `min_raise` is the real minimum on every street. No raise cap.
  - `legacy`: the sizing used before the v2 fix; runs without a `bet_sizing`
    entry used it.
  - `v5` (default for new runs): v4's betting menu and keys with new
    postflop card buckets (`new_code/card_buckets.py`). A hand is bucketed by
    how it plays on this board, not by its category: on the flop and turn by
    its equity against a random hand (10 levels) and its potential, equity
    minus current strength (drawing / neutral / vulnerable); on the river by
    current strength (15 levels). Mid pair on K-9-2 rainbow and on J-T-9
    two-tone now land in different buckets. On held-out deals the buckets
    explain 98% of the variation in equity on the flop and turn and 100% of
    river strength, against 64-66% for the old hand categories, with 27
    buckets per street instead of 173-288. Boundaries come from
    `tools/analysis/fit_card_buckets.py`; changing them changes every v5 key.
  - `v4`: v3's sizes and caps with fixed postflop
    betting keys. Earlier keys treated the whole-hand totals in OpenSpiel's
    `Sequences` and `Pot` as amounts for the street, so every first bet on a
    street read as pot-sized, from a min-bet to an all-in. v4 labels each
    raise by its size over the call as a fraction of the pot after calling
    (the measure the menu uses), marks real all-ins, and adds the
    stack-to-pot ratio at the start of the street.

## Training metrics

`metrics.csv` includes `avg_regret_bound`: the sum over infosets of each
infoset's largest positive regret, divided by the total iteration weight. In
exact CFR on a perfect-recall game this bounds twice the exploitability. Here
regrets are sampled and the card abstraction forgets earlier streets, so treat
it as a convergence signal (it should trend down), not a guarantee.
