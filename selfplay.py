# for i in range(num_games):
#     while game not terminal:
#         y, move = ai(game)
#         x = game.make_move(move)
#       
#         data.append(x, y)
#
#     data.save
#
# train ai on data

# Just use a open ai gymnasium







### Less important notes:

# Generate training data with ai making the move for every player
# - Then train the ai on that data

# The action space: The set of all actions:
# - Fold
# - Call
# - Raising _n chips (Includes checking, aka raising 0)
# 
# Size of the action space = 1 + 1 + However many raising amount we want


# The AI prediction function:
# - Inputs: the game variables we want it to know
# - Returns: How good a state is/policy


# The decision tree:
# 