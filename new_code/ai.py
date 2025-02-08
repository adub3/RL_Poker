import numpy as np
import pyspiel
import treelib
from abstraction import abstractbetting, abstractioncards, parse_poker_string, generate_empty_strategy_and_regret
import ujson
import os

game_config = {
    "betting": "nolimit", # Betting style: "limit" or "nolimit"
    "numPlayers": 2,      # Number of players
    "numRounds": 4,       # Number of betting rounds (preflop, flop, turn, river)
    "blind": "100 50",    # Blinds (small blind and big blind)
    "firstPlayer": "1",   # First player to act
    "numSuits": 4,        # Number of suits in the deck
    "numRanks": 13,       # Number of ranks in the deck
    "numHoleCards": 2,    # Number of hole cards per player
    "numBoardCards": "0 3 1 1",  # Number of board cards per round
    "stack": "20000 20000",  # Starting stack sizes for each player
}

class NodeData:
    def __init__(self, state, value=None, policy=[]) -> None:
        # Store variables like the game state, value, policy, etc.
        self.state = state

        # These variables aren't used, they're just examples
        self.value = value
        self.policy = policy
        self.visited = False
        self.action_taken = None

def MCCFR(state, player: int, strategy, regrets):
    """
    Performs Counterfactual Regret.
    
    Here I'm not using treelib.Node or NodeData, because it's easier.

    Args:
        state: The current state of the game.
        player (int): The identifier of the player for whom CFR is computed.
    """

    if state.is_terminal():
        return state.rewards()[player] # Get the reward for the player
    
    # elif player not in range(state.num_players()):
        # This case should never occur

    elif state.is_chance_node():
        new_state = state.clone() # Make a copy

        outcomes_with_probs = new_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)

        action = np.random.choice(action_list, p=prob_list) # Choose a random "action"
        new_state.apply_action(action)
    
        return MCCFR(new_state, player, strategy, regrets)
    
    elif state.current_player() == player:
        strategy = None

        temp = parse_poker_string(state.information_state_string())
        cards = abstractioncards(temp)
        context = abstractbetting(temp)
        res = cards + ' ' + context

        value = 0
        action_space = state.legal_actions()
        policy_list = calculate_strategy(state, strategy, regrets)[res]
        # MATCH POLICY AND ACTIONS TOGETHER, AND SOFTMAX

        # Actions are numbered 0 to n - 1 in the action_space, 
        # thus the corresponding policy is policy_list[action]
        # Truncate; this new list is normalized
        policy_list = [policy_list[i]/sum(policy_list) for i in action_space]

        action_value_list = []
        for action, policy in zip(action_space, policy_list):
            new_state = state.clone()
            new_state.apply_action(action)

            action_value = MCCFR(new_state, player, strategy, regrets)
            action_value_list.append(action_value)

            value = value + action_value * policy
        
        # Update regrets
        for i, action_index in enumerate(action_space): # This is confusing but it works
            regrets[res][action_index] = regrets[res][action_index] + action_value_list[i] - value
        
        return value
    else: # I believe this case occurs when it's the other player's turn
        temp = parse_poker_string(state.information_state_string())
        cards = abstractioncards(temp)
        context = abstractbetting(temp)
        res = cards + ' ' + context
        new_state = state.clone()

        action_space = state.legal_actions()
        policy = calculate_strategy(state, strategy, regrets)[res]
        policy_list = [policy_list[i]/sum(policy_list) for i in action_space]

        action = np.random.choice(action_space, p=policy)
        new_state.apply_action(action)
        return MCCFR(new_state, player, strategy, regrets)

def calculate_strategy(state, strategy, regrets):
    """
    Uses regrets to update the strategy.
    
    Why is it called calculate
    """
    sum = 0

    infostate = None

    temp = parse_poker_string(state.information_state_string())
    cards = abstractioncards(temp)
    context = abstractbetting(temp)
    infostate = cards + ' ' + context
    

    policy = strategy[infostate]
    node_regrets = regrets[infostate]

    action_size = len(state.legal_actions())

    for regret in node_regrets:
        sum += max(0, regret)
    
    # MATCH REGRETS TO POLICY
    for i, regret in enumerate(node_regrets):
        if sum > 0:
            policy[i] = round(max(0, regret) / sum, 2)
        else:
            policy[i] = round(1 / action_size, 2)
    
    return strategy

def save_strategy(strategy):
    json_data = ujson.dumps(strategy)

    path = os.path.dirname(os.path.realpath(__file__))

    with open(f"{path}/blackjack.txt", 'w') as out_file:
        out_file.write(json_data)

def load_strategy():
    path = os.path.dirname(os.path.realpath(__file__))
    set = ujson.load(open(f"{path}/blackjack.txt", 'r'))
    return set

def selfplay():
    # strategy, regrets = generate_empty_strategy_and_regret()
    # save_strategy(strategy)

    _, regrets = generate_empty_strategy_and_regret()
    strategy = load_strategy()

    game = pyspiel.load_game("universal_poker", game_config)

    for i in range(5):
        state = game.new_initial_state()
        MCCFR(state, 0, strategy, regrets)

        state = game.new_initial_state()
        MCCFR(state, 1, strategy, regrets)

        if i % 3 == 0:
            save_strategy(strategy)
    return strategy

selfplay()