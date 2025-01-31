import numpy as np
import time

from ai import *

def test_simulate_round():
    # Play through a round of two player poker
    # Load the Universal Poker game with Texas Hold'em configuration
    game = pyspiel.load_game("universal_poker", game_config)

    state = game.new_initial_state()

    print("Initial state:")
    print(state)

    # Simulate the game until it's over
    while not state.is_terminal():
        # Get the current player
        current_player = state.current_player()

        # If it's a chance node, sample a random outcome (e.g., dealing cards)
        if state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)

            action = np.random.choice(action_list, p=prob_list) # Choose a random "action"
            state.apply_action(action)

        else:
            # Get the legal actions for the current player without all ins cause they're boring
            actions_excluding_all_in = get_actions_without_allin(state)
            
            # Choose a random action
            action = np.random.choice(actions_excluding_all_in)
            print(f"Player {current_player} takes action: {action}")
            state.apply_action(action)

        # Print the current state
        print(state)
        time.sleep(0.1)

    # Print the final state and returns (utilities)
    print("-----Final state-----:")
    print(state)
    print(f"Returns: {state.returns()}")

if __name__ == "__main__":
    test_simulate_round()