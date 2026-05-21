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

## Setup

The project is currently pinned to Python 3.11.9 with `pyenv-win`.

```powershell
pip install -r requirements.txt
```

## Checks

```powershell
python -B new_code\test_ai_core.py
```

OpenSpiel is installed and works for `kuhn_poker` and `leduc_poker`. The
current Windows wheel does not register `universal_poker`, so Texas hold'em
training needs either a source build with that binding enabled or a different
game backend.
