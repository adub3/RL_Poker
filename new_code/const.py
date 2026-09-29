game_config = {
    "betting": "nolimit", # Betting style: "limit" or "nolimit"
    "numPlayers": 2,      # Number of players
    "numRounds": 4,       # Number of betting rounds (preflop, flop, turn, river)
    "blind": "100 50",    # Blind posted by P0 (big blind) and P1 (small blind)
    "bettingAbstraction": "fullgame",
    # First to act on each street, numbered from 1: the small blind (P1)
    # preflop, the big blind (P0) on the flop, turn and river.
    "firstPlayer": "2 1 1 1",
    "numSuits": 4,        # Number of suits in the deck
    "numRanks": 13,       # Number of ranks in the deck
    "numHoleCards": 2,    # Number of hole cards per player
    "numBoardCards": "0 3 1 1",  # Number of board cards per round
    "stack": "10000 10000",  # Starting stack sizes for each player (100bb @ BB=100)
}
# Runs from before the seat-order fix used firstPlayer "1": the big blind acted
# first on every street, including preflop. Their manifests record it.
LEGACY_FIRST_PLAYER = "1"
