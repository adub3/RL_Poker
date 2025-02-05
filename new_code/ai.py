import numpy as np
import pyspiel
import treelib

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

def get_actions_without_allin(state):
    current_player = state.current_player()

    legal_actions = state.legal_actions()
    legal_action_strings = [state.action_to_string(current_player, action) for action in legal_actions]
    
    actions_excluding_all_in = []

    for action_num, action_str in zip(legal_actions, legal_action_strings):
        _, move = action_str.split()
        _, move = move.split("=")
        if move != "AllIn":
            actions_excluding_all_in.append(action_num)
    
    return actions_excluding_all_in

class NodeData:
    def __init__(self, state, value=None, policy=[]) -> None:
        # Store variables like the game state, value, policy, etc.
        self.state = state

        # These variables aren't used, they're just examples
        self.value = value
        self.policy = policy
        self.visited = False
        self.action_taken = None

def playout_game_with_tree() -> treelib.Tree:
    # Play through a game and save it to a treelib tree
    
    # Variable hierarchy:
    # Tree (treelib)
    #   Node (treelib)
    #     NodeData
    #       State (openspiel)
    #       Value
    #       Policy

    tree = treelib.Tree()

    # Initialize a game
    game = pyspiel.load_game("universal_poker", game_config)
    state = game.new_initial_state()

    # Create root node
    root_state = NodeData(state=state)
    tree.create_node(identifier='root', data=root_state)

    # Traverse the tree
    node = tree.get_node("root")
    state = node.data.state

    while not state.is_terminal():
        # Create a copy of the state
        # This will become a child of the current node
        new_state = state.clone()

        if new_state.is_chance_node(): # Chance node
            outcomes_with_probs = new_state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)

            action = np.random.choice(action_list, p=prob_list)
            new_state.apply_action(action)
        else: # Decision node

            # Example code for using a strategy:
            if False:
                strategy = {}
                infostate = ""
                value, policy = strategy[infostate] # Not sure how to use value
                policy_mask = policy * new_state.legal_actions() # Set invalid actions to a policy of 0

                action = np.random.choice(legal_actions, p=policy_mask)

            legal_actions = new_state.legal_actions()
            
            action = np.random.choice(legal_actions)
            new_state.apply_action(action)

            # Update node info
            node.data.action_taken = action
        
        # Save the copy of the state to the tree
        new_nodedata = NodeData(new_state)
        
        new_node = tree.create_node(data=new_nodedata, parent=node.identifier)

        # Update values
        node = new_node
        state = new_state
    
    print(state)
    print(state.rewards())
    return tree

def adjust_policy_tree_traversal_example(tree: treelib.Tree, reward: list[int, int], strategy=None) -> None:
    # Tree traversal example
    node = tree.get_node("root")
    
    # Each node should only have one child
    while not node.is_leaf():
        state = node.data.state

        if not state.is_chance_node():
            current_player = state.current_player()
            action_space = state.legal_actions()
            action_taken = node.data.action_taken
            
            current_player_reward = reward[current_player]

            # Example code (This is not CFR, this is some bullshit)
            if False:
                new_strategy = strategy.copy()

                for i, action in enumerate(action_space):
                    reward_derivative = learning_rate * current_player_reward # For the action taken, adjust it relative to the reward
                    if action != action_taken:
                        reward_derivative = -reward_derivative 

                    new_strategy["infostate-xyz"][i] += reward_derivative

        child_id = node.successors(tree.identifier)[0]
        child = tree.get_node(child_id)
        node = child

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

        value = 0
        action_space = state.legal_actions()
        policy_list = calculate_strategy("infostate-xyz", strategy, regrets)["infostate-xyz"]
        # MATCH POLICY AND ACTIONS TOGETHER, AND SOFTMAX

        action_value_list = []
        for action, policy in zip(action_space, policy_list):
            new_state = state.clone()
            new_state.apply_action(action)

            action_value = MCCFR(new_state, player, strategy, regrets)
            action_value_list.append(action_value)

            value = value + action_value * policy
        
        # Update regrets
        for action in action_space:
            regrets["infostate-xyz"][action] = regrets["infostate-xyz"][action] + action_value_list[action] - value
        
        return value
    else: # I believe this case occurs when it's the other player's turn
        new_state = state.clone()

        policy = calculate_strategy(None, strategy, regrets)["infostate-xyz"]
        action_space = state.legal_actions()

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

    policy = strategy[infostate]
    node_regrets = regrets[infostate]

    action_size = len(state.legal_actions())

    for regret in node_regrets:
        sum += max(0, regret)
    
    # MATCH REGRETS TO POLICY
    for i, regret in enumerate(node_regrets):
        if sum > 0:
            policy[i] = max(0, regret) / sum
        else:
            policy[i] = 1 / action_size
    
    return strategy

if __name__ == "__main__":
    ex_tree = playout_game_with_tree()
    adjust_policy_tree_traversal_example(ex_tree, [-200, 200])