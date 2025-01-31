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

def generate_self_play_tree():
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
        if state.is_chance_node(): # Chance node
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)

            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else: # Decision node

            # Example code for using a strategy:
            if False:
                strategy = {}
                infostate = ""
                value, policy = strategy[infostate] # Not sure how to use value
                policy_mask = policy * state.legal_actions() # Set invalid actions to a policy of 0

                action = np.random.choice(legal_actions, p=policy_mask)

            legal_actions = state.legal_actions()
            
            action = np.random.choice(legal_actions)
            state.apply_action(action)
        
        # Save a copy of the state to the tree
        new_state = state.clone()
        new_nodedata = NodeData(new_state)
        
        new_node = tree.create_node(data=new_nodedata, parent=node.identifier)

        # Update values
        node = new_node
        state = new_state
    
    # Tree traversal example
    node = tree.get_node("root")
    
    # Each node should only have one child
    while not node.is_leaf():
        child_id = node.successors(tree.identifier)[0]
        child = tree.get_node(child_id)
        print(child.data.state)

        node = child

if __name__ == "__main__":
    generate_self_play_tree()