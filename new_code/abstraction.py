import re
import eval7
import math

#Notes:
# trips and top pair same some similar code that could be combined maybe better to write a func that just sees the highest card on the board and matches it to the card in hand or smtn
#not rlly sure whats best here

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

def abstractbetting(sequence, initial_pot):
    result = ""
    current_number = ""
    current_pot = int(initial_pot)
    future_bets = []
    
    # First pass: collect all bets
    temp_num = ""
    for char in sequence:
        if char.isdigit():
            temp_num += char
        elif temp_num:
            future_bets.append(int(temp_num))
            temp_num = ""
    if temp_num:
        future_bets.append(int(temp_num))
    
    # Second pass: process sequence
    bet_index = 0
    for char in sequence:
        if char == 'c':
            result += 'c'
        elif char == 'r':
            if current_number:
                number = int(current_number)
                # Calculate remaining future bets
                remaining_bets = sum(future_bets[bet_index + 1:])
                # Adjust pot for current calculation
                adjusted_pot = current_pot - remaining_bets
                processed_number = math.floor(math.log(number/adjusted_pot + 1) * 4)
                result += str(processed_number)
                current_pot += number  # Update pot for next calculations
                current_number = ""
                bet_index += 1
            result += 'r'
        elif char.isdigit():
            current_number += char
    
    # Process any remaining number at the end
    if current_number:
        number = int(current_number)
        processed_number = math.floor(math.log(number/current_pot + 1) * 4)
        result += str(processed_number)
    
    return result

# Example usage



def abstractioncards(data):
    """
    Categorize a poker hand based on its type and rank.

    Args:
        hand_cards (list of eval7.Card): Player's two hole cards.
        board_cards (list of eval7.Card): Community cards on the board.

    Returns:
        dict: Categorized hand information. (eventually transformed into a string)
    """
    data_dict = parse_poker_string(data) #Transforming Raw Game State into variables

    all_cards = data_dict['Private'] + data_dict['Public'] #data_dict[priv] are user cards, else is non user
    eval7allcards = [eval7.Card(s) for s in all_cards] #making eval7 cards
    hand_type = eval7.handtype(eval7.evaluate(eval7allcards)) #eval7 can identify what type of hand it is but idk if strength of hand like Ace pair etc. Maybe some func

    categorized_hand = {
        "hand_type": hand_type,
        "details": {}
    }
    print(data_dict)
    community_cards = [eval7.Card(c) for c in data_dict['Public']]

    rank_counts = {} #Rank counts for whole board

    for card in eval7allcards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

    rank_counts_board = {} #Rank counts  for just the board, also gives implied information on how paired the board is

    for card in [eval7.Card(s) for s in data_dict["Public"]]:
        rank_counts_board[card.rank] = rank_counts.get(card.rank, 0) + 1

    rank_counts_priv = {}

    for card in [eval7.Card(s) for s in data_dict["Private"]]:
        rank_counts_priv[card.rank] = rank_counts.get(card.rank, 0) + 1

    if hand_type == "High Card": #str starts w/ 1
        highest_card = max(all_cards, key=lambda card: card.rank)
        if highest_card == 14 or 13:
            return "11"
        else:
            return "12"
    elif hand_type == "Pair" or hand_type == "Two Pair" or hand_type == "Trips": #str starts w/ 2
        # Extract community cards
        
        # Find the highest pair
        pairs = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
        board_pairs = sorted([rank for rank, count in rank_counts_board.items() if count >= 2], reverse=True)

        highest_pair = pairs[0]  # The highest ranked pair
        if hand_type == "Pair" or hand_type == "Two Pair":
            # Determine objective value Eval7 treats 2 = 0, 3 = 1, ... A = 12
            if highest_pair <= 6:
                objective_value = "2-8"
            elif highest_pair <= 9:
                objective_value = "9-11" 
            elif highest_pair <= 11:
                objective_value = "12-13"
            else:
                objective_value = "Overpair"
        else:
            if board_pairs:
                typetrips = "Set"
            else:
                typetrips = "Trips"
        # Get board ranks
        board_ranks = sorted(set(card.rank for card in community_cards), reverse=True)
        # Relative Value func
        if highest_pair in board_ranks:
                # Pair is on the board
            if len(board_ranks) >= 3 and highest_pair == board_ranks[-1]:  # Lowest pair for 3 cards
                relative_value = "Low"
            elif len(board_ranks) >= 4 and highest_pair in board_ranks[-2:]:  # Bottom two for 4+ cards
                relative_value = "Low"
            elif len(board_ranks) == 3 and highest_pair == board_ranks[1]:  # Middle pair for 3 cards
                relative_value = "Middle"
            elif len(board_ranks) == 5 and highest_pair == board_ranks[2] or highest_pair == board_ranks[1]:  # Middle pair for 5 cards
                relative_value = "Middle"
            else:
                relative_value = "Top Pair"
        else:
            relative_value = "Overpair"
        if hand_type == "Pair" or hand_type == "Two Pair":
            return {  #this function wants to be broken down into a return of two numbers where first is either 1 or 2 depending on hand str, then the second is the obj ranking i.e. 3i+j where i is relative rank and j is obj rank and 0 is over or stmn of the sort
                "Highest Pair": highest_pair + 2,
                "Objective Value": objective_value,
                "Relative Value": relative_value,
                "Ranking": hand_type
                }
        else:
            return {
                "Relative Value": relative_value,
                "Ranking": hand_type,
                "Type": typetrips
            }


    elif hand_type == "Full House":
        total_freq = rank_counts
        pocket_freq = rank_counts_priv
        print("Total frequencies (all cards):", total_freq)
        print("Pocket frequencies:", pocket_freq)

        # Identify the best triple: the highest rank with at least 3 occurrences.
        triple_rank = None
        for r in sorted(total_freq.keys(), reverse=True):
            if total_freq[r] >= 3:
                triple_rank = r
                break

        if triple_rank is None:
            return {"Error": "No valid triple found for a full house."}

        # Identify the best pair from a different rank: the highest rank (≠ triple_rank) with at least 2 occurrences.
        pair_rank = None
        for r in sorted(total_freq.keys(), reverse=True):
            if r == triple_rank:
                continue
            if total_freq[r] >= 2:
                pair_rank = r
                break

        if pair_rank is None:
            return {"Error": "No valid pair found for a full house."}

        # Count how many pocket cards contributed to the triple and the pair.
        # The .get() method retrieves the count for a given rank (using our integer rank) or returns 0 if not found.
        pocket_used_triple = pocket_freq.get(triple_rank, 0)
        pocket_used_pair   = pocket_freq.get(pair_rank, 0)
        pocket_used_total  = pocket_used_triple + pocket_used_pair

        # Debug prints to show the counts:
        print(f"Triple rank {triple_rank} occurs {total_freq[triple_rank]} times overall; "
            f"in pocket: {pocket_used_triple}")
        print(f"Pair rank {pair_rank} occurs {total_freq[pair_rank]} times overall; "
            f"in pocket: {pocket_used_pair}")

        if pocket_used_total == 2:
            used_cards = "Two pocket cards used"
        elif pocket_used_total == 1:
            used_cards = "One pocket card used"
        else:
            used_cards = f"{pocket_used_total} pocket cards used"

        return {
            "hand_type": "Full House",
            "Triple Rank": triple_rank,
            "Pair Rank": pair_rank,
            "Used Pocket Cards": used_cards
        }
    elif hand_type == "Straight":
        return
    elif hand_type == "Flush":
        return
    elif hand_type == "Quads":
        return

    return categorized_hand

# Example usage
poker_string = "[Round 0][Player: 0][Pot: 40000][Money: 19900 0][Private: 8c8h][Public: 3sJc5d8sJs][Sequences: cr20000]"

result = abstractioncards(poker_string)
print(result)
# Print the parsed dictionary