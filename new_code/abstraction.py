import re
import eval7

def parse_poker_string(poker_string):
    # Define a regex pattern to extract key-value pairs inside the brackets
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, poker_string)
    
    # Initialize an empty dictionary to store the parsed information
    parsed_data = {}
    
    # Iterate through the matches and parse them into the dictionary
    for match in matches:
        # Split the match into key and value using the first colon or space
        if ":" in match:
            key, value = match.split(":", 1)
        else:
            key, value = match.split(" ", 1)
        
        key = key.strip()
        value = value.strip()
        
        # Handle specific cases for Money, Private, and Public
        if key == "Money":
            parsed_data[key] = list(map(int, value.split()))
        elif key == "Private":
            parsed_data[key] = [value[i:i+2] for i in range(0, len(value), 2)]
        elif key == "Public":
            parsed_data[key] = [value[i:i+2] for i in range(0, len(value), 2)] if value else []
        else:
            # Convert numeric values to integers if possible
            if value.isdigit():
                parsed_data[key] = int(value)
            else:
                parsed_data[key] = value
    
    
    return parsed_data


def abstraction(data):
    """
    Categorize a poker hand based on its type and rank.

    Args:
        hand_cards (list of eval7.Card): Player's two hole cards.
        board_cards (list of eval7.Card): Community cards on the board.

    Returns:
        dict: Categorized hand information. (eventually transformed into a string)
    """
    data_dict = parse_poker_string(data)
    all_cards = data_dict['Private'] + data_dict['Public'] #data_dict[priv] are user cards, else is non user
    eval7allcards = [eval7.Card(s) for s in all_cards]
    hand_type = handstr = eval7.handtype(eval7.evaluate(eval7allcards)) #eval7 can identify what type of hand it is but idk if strength of hand like Ace pair etc. Maybe some func
    print(data_dict["Public"])
    categorized_hand = {
        "hand_type": hand_type,
        "details": {}
    }
    community_cards = [eval7.Card(c) for c in data_dict['Public']]
    rank_counts = {}
    for card in eval7allcards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
    rank_counts_board = {}
    rank_counts_board
    for card in [eval7.Card(s) for s in data_dict["Public"]]:
        rank_counts_board[card.rank] = rank_counts.get(card.rank, 0) + 1
    if hand_type == "High Card": #str starts w/ 1
        highest_card = max(all_cards, key=lambda card: card.rank)
        if highest_card == 14 or 13:
            return "11"
        else:
            return "12"
    elif hand_type == "Pair" or hand_type == "Two Pair": #str starts w/ 2
        # Extract community cards
        
        # Find the highest pair
        pairs = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
        if not pairs:
            return None  # No pair found

        highest_pair = pairs[0]  # The highest ranked pair

        # Determine objective value Eval7 treats 2 = 0, 3 = 1, ... A = 12
        if highest_pair <= 6:
            objective_value = "2-8"
        elif highest_pair <= 9:
            objective_value = "9-11" 
        elif highest_pair <= 11:
            objective_value = "12-13"
        else:
            objective_value = "Overpair"

        # Get board ranks
        board_ranks = sorted(set(card.rank for card in community_cards), reverse=True)

        # Determine relative value
        if highest_pair in board_ranks:
                # Pair is on the board
            if len(board_ranks) >= 3 and highest_pair == board_ranks[-1]:  # Lowest pair for 3 cards
                relative_value = "Low Pair"
            elif len(board_ranks) >= 4 and highest_pair in board_ranks[-2:]:  # Bottom two for 4+ cards
                relative_value = "Low Pair"
            elif len(board_ranks) == 3 and highest_pair == board_ranks[1]:  # Middle pair for 3 cards
                relative_value = "Middle Pair"
            elif len(board_ranks) == 5 and highest_pair == board_ranks[2]:  # Middle pair for 5 cards
                relative_value = "Middle Pair"
            else:
                relative_value = "Top Pair"
        else:
            relative_value = "Overpair"

        return {  #this function wants to be broken down into a return of two numbers where first is either 1 or 2 depending on hand str, then the second is the obj ranking i.e. 3i+j where i is relative rank and j is obj rank and 0 is over or stmn of the sort
            "Highest Pair": highest_pair + 2,
            "Objective Value": objective_value,
            "Relative Value": relative_value
            
            }
    elif hand_type == "Trips":
        board_ranks = sorted(set(card.rank for card in community_cards), reverse=True)
        
        # Determine if the hand is trips or a set
        pairs = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
        board_pairs = sorted([rank for rank, count in rank_counts_board.items() if count >= 2], reverse=True)
        highest_pair = pairs[0]
        if board_pairs:
            typetrips = "Set"
            print("set bullshit worked")
        else:
            typetrips = "Trips"
        if highest_pair in board_ranks:
            # Pair is on the board
            if len(board_ranks) >= 3 and highest_pair == board_ranks[-1]:  # Lowest pair for 3 cards
                relative_value = "Low Pair"
            elif len(board_ranks) >= 4 and highest_pair in board_ranks[-2:]:  # Bottom two for 4+ cards
                relative_value = "Low Pair"
            elif len(board_ranks) == 3 and highest_pair == board_ranks[1]:  # Middle pair for 3 cards
                relative_value = "Middle Pair"
            elif len(board_ranks) == 5 and highest_pair == board_ranks[2]:  # Middle pair for 5 cards
                relative_value = "Middle Pair"
            else:
                relative_value = "Top Pair"
        # Relative strength evaluation for trips
        return{
            "Trip_types": typetrips,
            "Relative Value": relative_value
            }

    elif hand_type == "Full House":
        return
    elif hand_type == "Straight":
        return
    elif hand_type == "Flush":
        return
    elif hand_type == "Quads":
        return

    return categorized_hand

# Example usage
poker_string = "[Round 0][Player: 0][Pot: 40000][Money: 19900 0][Private: 8c8h][Public: 3s4c5d8sJs][Sequences: cr20000]"

result = abstraction(poker_string)
print(result)
# Print the parsed dictionary
abstraction(poker_string)