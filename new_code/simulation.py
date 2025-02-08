import numpy as np
import time

from ai import *

def test_simulate_round():
    # Play through a round of two player poker
    # Load the Universal Poker game with Texas Hold'em configuration
    game = pyspiel.load_game("universal_poker", game_config)

    strategy = load_strategy()

    active_player = 1

    while True:
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
                action_space = state.legal_actions()
                print(action_space)
                print([state.action_to_string(current_player, action) for action in action_space])

                if current_player == active_player:
                    print(parse_poker_string(state.information_state_string()))
                    action = int(input("Enter an action: "))
                
                else:
                    # Choose a random action
                    res = get_infostate(state)
                    policy = strategy[res]
                    policy_list = [policy[i] for i in action_space]
                    print(res, policy)
                    if(sum(policy_list) == 0):
                        action = 1
                    else:
                        policy_list = [p / sum(policy_list) for p in policy_list]
                        action = np.random.choice(action_space, p=policy_list)

                # action = np.random.choice(action_space)
                print(f"Player {current_player} takes action: {action}")
                state.apply_action(action)

            # Print the current state
            # print(state)
            time.sleep(0.1)

        # Print the final state and returns (utilities)
        print("-----Final state-----:")
        print(state)
        print(f"Returns: {state.returns()}")
        active_player = 1 - active_player

if __name__ == "__main__":
    test_simulate_round()