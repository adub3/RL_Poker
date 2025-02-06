import math
import numpy as np
import pyspiel
from ai import game_config

# Define action space
action_space = np.array([0, 1, 2, 3, 4], dtype=np.uint8)

# Save to binary file
np.save("action_space.npy", action_space)

# Load from binary file
loaded_action_space = np.load("action_space.npy")
print(loaded_action_space)

def test_mccfr():
    def MCCFR(state, player: int, strategy, regrets):
        # TESTING FUNCTINO NOT USED
        # THIS IS DIFFERENT
        
        if state.is_terminal():
            return state.rewards()[player]

        elif state.is_chance_node():
            new_state = state.clone()

            outcomes_with_probs = new_state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)

            action = np.random.choice(action_list, p=prob_list)
            new_state.apply_action(action)
        
            return MCCFR(new_state, player, strategy, regrets)
        
        elif state.current_player() == player:
            value = 0
            action_space = state.legal_actions()
            policy_list = calculate_strategy(state, strategy, regrets)
            policy_list = [policy_list[i] for i in action_space] # Truncate

            # Normalize probabilities
            action_value_list = []
            for action, policy in zip(action_space, policy_list):
                new_state = state.clone()
                new_state.apply_action(action)

                action_value = MCCFR(new_state, player, strategy, regrets)
                action_value_list.append(action_value)

                value = value + action_value * policy
            
            for i, action_index in enumerate(action_space):
                regrets[action_index] = regrets[action_index] + action_value_list[i] - value
            
            return value
        else:
            new_state = state.clone()

            action_space = state.legal_actions()
            policy_list = calculate_strategy(state, strategy, regrets)
            policy_list = [policy_list[i] for i in action_space] # Truncate

            # Normalize probabilities
            action = np.random.choice(action_space, p=policy_list)
            new_state.apply_action(action)
            return MCCFR(new_state, player, strategy, regrets)

    def calculate_strategy(state, strategy, regrets):
        """
        Uses regrets to update the strategy.
        
        Why is it called calculate
        """
        sum = 0

        infostate = None

        policy = strategy
        node_regrets = regrets

        actions = state.legal_actions()

        for action in actions:
            sum += max(0, regrets[action]) # 1 index
        
        # MATCH REGRETS TO POLICY
        for action in actions:
            if sum > 0:
                policy[action] = max(0, regrets[action]) / sum
            else:
                policy[action] = 1 / len(actions)
        
        return strategy

    strategy = [1/4, 1/4, 1/4, 1/4]
    regrets = [0, 0, 0, 0]
    
    game = pyspiel.load_game("universal_poker", game_config)

    for _ in range(10):
        state = game.new_initial_state()
        MCCFR(state, 0, strategy, regrets)

        state = game.new_initial_state()
        MCCFR(state, 1, strategy, regrets)
    print(strategy, regrets)
    return strategy

def test_play_against_strategy(strategy):
    game = pyspiel.load_game("universal_poker", game_config)
    state = game.new_initial_state()

    while not state.is_terminal():
        current_player = state.current_player()

        if state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)

            action = np.random.choice(action_list, p=prob_list) # Choose a random "action"
            state.apply_action(action)

        else:
            print(state)
            action_space = state.legal_actions()
            
            if current_player == 0:
                print(action_space, [state.action_to_string(current_player, action) for action in action_space])
                action = int(input("Enter an action"))
            else:
                policy = [strategy[action - 1] for action in action_space]
                policy = [p / sum(policy) for p in policy]
                action = np.random.choice(action_space, p=policy)

            print(f"Player {current_player} takes action: {action}")
            state.apply_action(action)

if __name__ == "__main__":
    strategy = test_mccfr()
    test_play_against_strategy(strategy)