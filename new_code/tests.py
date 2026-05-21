import math
import numpy as np
from abstraction import abstractioncards, abstractbettinge

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

def test_abstraction_functions():
    print("\n--- Testing Abstraction Functions ---")

    # Mock data for abstractioncards
    # Example 1: Pre-flop, two hole cards
    data_dict_1 = {"Private": ["Ac", "Kd"], "Public": []}
    card_abstraction_1 = abstractioncards(data_dict_1)
    print(f"Card Abstraction 1 (Ac Kd, pre-flop): {card_abstraction_1}") # Expected: High card type for Ace, King with no draws

    # Example 2: Flop, pair on board
    data_dict_2 = {"Private": ["As", "Qs"], "Public": ["Kc", "Qd", "Jh"]}
    card_abstraction_2 = abstractioncards(data_dict_2)
    print(f"Card Abstraction 2 (As Qs, Kc Qd Jh): {card_abstraction_2}") # Expected: Pair type (Queens), possible straight/flush draws

    # Example 3: Flush draw on turn
    data_dict_3 = {"Private": ["7h", "8h"], "Public": ["2h", "Kh", "Th", "5d"]}
    card_abstraction_3 = abstractioncards(data_dict_3)
    print(f"Card Abstraction 3 (7h 8h, 2h Kh Th 5d): {card_abstraction_3}") # Expected: Flush type, flush draw info

    # Mock data for abstractbettinge
    # Need to mock a RoundState object minimally for abstractbettinge
    class MockRoundState:
        def __init__(self, pips_active, pips_opponent, street, recursion_depth=0):
            self.pips = [pips_active, pips_opponent]
            self.street = street
            # minimal previous_state needed for skeleton.py
            if recursion_depth < 1:
                self.previous_state = MockRoundState(pips_active, pips_opponent, street, recursion_depth + 1)
            else:
                self.previous_state = None

        
        # Override previous_state.pips for testing purposes.
        # This is a hack, but sufficient for just abstractbettinge
        def set_previous_pips(self, pips_active, pips_opponent):
            self.previous_state.pips = [pips_active, pips_opponent]

    # Example 4: Simple betting log
    mock_round_state_4 = MockRoundState(10, 10, 0) # my_pip, opp_pip, street
    mock_round_state_4.set_previous_pips(0, 0) # previous round pips
    betting_log_4 = "cc" # two calls
    betting_abstraction_4 = abstractbettinge(betting_log_4, mock_round_state_4, 0)
    print(f"Betting Abstraction 4 (cc): {betting_abstraction_4}")

    # Example 5: Raise and call
    mock_round_state_5 = MockRoundState(20, 20, 0)
    mock_round_state_5.set_previous_pips(10, 10)
    betting_log_5 = "rc" # raise then call
    betting_abstraction_5 = abstractbettinge(betting_log_5, mock_round_state_5, 0)
    print(f"Betting Abstraction 5 (rc): {betting_abstraction_5}")

    # Example 6: Three actions
    mock_round_state_6 = MockRoundState(30, 30, 0)
    mock_round_state_6.set_previous_pips(20, 20)
    betting_log_6 = "rcr" # raise, call, raise
    betting_abstraction_6 = abstractbettinge(betting_log_6, mock_round_state_6, 0)
    print(f"Betting Abstraction 6 (rcr): {betting_abstraction_6}")

    # Example 7: Betting with amounts
    mock_round_state_7 = MockRoundState(100, 100, 0)
    mock_round_state_7.set_previous_pips(0,0)
    betting_log_7 = "r50c" # raise 50, call
    betting_abstraction_7 = abstractbettinge(betting_log_7, mock_round_state_7, 0)
    print(f"Betting Abstraction 7 (r50c): {betting_abstraction_7}")



if __name__ == "__main__":
    test_abstraction_functions()