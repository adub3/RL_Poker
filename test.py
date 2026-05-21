import subprocess
import sys

# Install open_spiel from PyPI (the wheel includes universal_poker / ACPC)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "open_spiel"])

import pyspiel

# Confirm universal_poker is in the registered games list
assert "universal_poker" in pyspiel.registered_names(), "universal_poker not built into this wheel"
print("universal_poker is registered ✓")

# Load a standard heads-up no-limit hold'em variant and print basic info
game = pyspiel.load_game(
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numRounds=4,"
    "blind=100 50,"
    "firstPlayer=2 1 1 1,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numBoardCards=0 3 1 1,"
    "stack=20000 20000,"
    "bettingAbstraction=fullgame"
    ")"
)

print(f"Game: {game}")
print(f"Num players: {game.num_players()}")
print(f"Max chance outcomes: {game.max_chance_outcomes()}")

# Play out a random trajectory just to confirm it works end to end
import random
state = game.new_initial_state()
while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action = random.choices([a for a, _ in outcomes], [p for _, p in outcomes])[0]
    else:
        action = random.choice(state.legal_actions())
    state.apply_action(action)

print(f"Terminal returns: {state.returns()}")