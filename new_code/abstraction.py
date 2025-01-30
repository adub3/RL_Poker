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
def abstraction(data): #func takes player info and abstracts it into bins
#changing data into dict
    data_dict = parse_poker_string(data)
    cards: list = []
    print(data_dict['Private'] + data_dict['Public'], "bannan")
    for card in data_dict['Private'] and data_dict['Public']:
        cards.append(card) # list of all cards
    handstr = eval7.handtype(eval7.evaluate([eval7.Card(s) for s in cards])) #type of
    return "A"
    #parsed data has 'round, player, pot, money, priv, pub, seq (ignore money only based on priv pub seq and seq)
# Example usage
import eval7

def abstraction(data):
    """
    Categorize a poker hand based on its type and rank.

    Args:
        hand_cards (list of eval7.Card): Player's two hole cards.
        board_cards (list of eval7.Card): Community cards on the board.

    Returns:
        dict: Categorized hand information.
    """
    data_dict = parse_poker_string(data)
    all_cards = data_dict['Private'] + data_dict['Public']
    hand_cards = data_dict['Private']
    board_cards = data_dict['Public']
    hand_rank = eval7.evaluate(all_cards)
    hand_type = eval7.handtype(hand_rank)

    categorized_hand = {
        "hand_type": hand_type,
        "rank": hand_rank,
        "details": {}
    }

    if hand_type == "High Card":
        categorized_hand["details"] = {
            "high_card": max(hand_cards, key=lambda card: card.rank),
            "is_mid_low": any(card.rank < eval7.Card("9s").rank for card in hand_cards)
        }

    elif hand_type == "Pair":
        pair_rank = max(card.rank for card in hand_cards if hand_cards.count(card) > 1)
        categorized_hand["details"] = {
            "pair_rank": pair_rank,
            "category": "Low" if pair_rank <= eval7.Card("8s").rank else
                        "Mid" if pair_rank <= eval7.Card("Js").rank else "High",
            "relative_to_board": ""  # To be computed based on board context
        }

    elif hand_type == "Two Pair":
        # Identify ranks of the two pairs
        paired_ranks = [card.rank for card in all_cards if all_cards.count(card) == 2]
        categorized_hand["details"] = {
            "paired_ranks": sorted(set(paired_ranks), reverse=True)
        }

    elif hand_type == "Trips":
        trips_rank = max(card.rank for card in all_cards if all_cards.count(card) == 3)
        categorized_hand["details"] = {
            "trips_rank": trips_rank,
            "is_set": all_cards.count(hand_cards[0]) == 3 or all_cards.count(hand_cards[1]) == 3,
            "category": "Low" if trips_rank <= eval7.Card("8s").rank else
                        "Mid" if trips_rank <= eval7.Card("Js").rank else "High"
        }

    elif hand_type == "Full House":
        # Evaluate full house ranking
        trip_rank = max(card.rank for card in all_cards if all_cards.count(card) == 3)
        pair_rank = max(card.rank for card in all_cards if all_cards.count(card) == 2)
        categorized_hand["details"] = {
            "trip_rank": trip_rank,
            "pair_rank": pair_rank,
            "is_two_card_fh": trip_rank in [card.rank for card in hand_cards] and \
                              pair_rank in [card.rank for card in hand_cards]
        }

    elif hand_type == "Straight":
        highest_card = max(all_cards, key=lambda card: card.rank)
        categorized_hand["details"] = {
            "highest_card": highest_card,
            "is_high_straight": highest_card.rank >= eval7.Card("Qs").rank
        }

    elif hand_type == "Flush":
        flush_suit = max([card.suit for card in all_cards if \
                          sum(1 for c in all_cards if c.suit == card.suit) >= 5],
                         key=lambda suit: suit)
        categorized_hand["details"] = {
            "flush_suit": flush_suit,
            "is_high_flush": any(card.rank >= eval7.Card("Qs").rank for card in all_cards if card.suit == flush_suit)
        }
    elif hand_type == "Quads":
        quad_rank = max(card.rank for card in all_cards if all_cards.count(card) == 4)
        categorized_hand["details"] = {
            "quad_rank": quad_rank,
            "quads_on_board": len([card for card in board_cards if board_cards.count(card) == 4]) > 0
        }

    return categorized_hand

# Example usage
poker_string = "[Round 0][Player: 0][Pot: 40000][Money: 19900 0][Private: Tc3h][Public: AcAdAhTh][Sequences: cr20000]"

hole_cards = [eval7.Card("As"), eval7.Card("Kh")]
board = [eval7.Card("2d"), eval7.Card("7c"), eval7.Card("Jc"), eval7.Card("8d"), eval7.Card("5h")]
result = categorize_hand(hole_cards, board)
print(result)
# Print the parsed dictionary
abstraction(poker_string)