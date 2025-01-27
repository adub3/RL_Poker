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