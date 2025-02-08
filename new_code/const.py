import math

game_config = {
    "betting": "nolimit", # Betting style: "limit" or "nolimit"
    "numPlayers": 2,      # Number of players
    "numRounds": 4,       # Number of betting rounds (preflop, flop, turn, river)
    "blind": "100 50",    # Blinds (small blind and big blind),
    "bettingAbstraction": "fullgame",
    "firstPlayer": "1",   # First player to act
    "numSuits": 4,        # Number of suits in the deck
    "numRanks": 13,       # Number of ranks in the deck
    "numHoleCards": 2,    # Number of hole cards per player
    "numBoardCards": "0 3 1 1",  # Number of board cards per round
    "stack": "20000 20000",  # Starting stack sizes for each player
}