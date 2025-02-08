import re
import eval7
import math
import itertools

#Notes:
# trips and top pair same some similar code that could be combined maybe better to write a func that just sees the highest card on the board and matches it to the card in hand or smtn
#not rlly sure whats best here

#(1: High Card, 2: Pair, 3: Trips, 4: Straight, 5: Flush, 6: Full House, 7: Quads, 8: Straight Flush and ...)

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

def missing_for_straight_with_debug(rank_counts):
    """
    Given a dictionary `rank_counts` where keys are card ranks (numeric)
    and values are the count of cards for that rank, this function returns
    a tuple (min_missing, sequence) where:
      - min_missing is the minimum number of additional cards required
        to complete any 5-card straight (5 consecutive ranks) in a standard deck.
      - sequence is the 5-card straight (list of consecutive ranks) for which
        that minimum was found.
    
    We assume:
      - Cards are valued from 2 up to Ace (14).
      - A straight must be exactly 5 consecutive values.
    """
    min_missing = float('inf')
    best_sequence = None
    
    # Check every possible straight from 2-6 up to 10-14.
    for start in range(1, 11):  # 10 is the last starting card: 10,11,12,13,14
        straight = list(range(start, start + 5))
        missing = sum(1 for card in straight if rank_counts.get(card, 0) == 0)
        
        # Debug: print the straight and how many cards are missing.
        #print(f"Checking straight {straight} => missing {missing}")
        
        if missing < min_missing:
            min_missing = missing
            best_sequence = straight
    
    return min_missing

def abstractbettinge(log, round_state, active): #abstracts raises into floor(math.log(number/adjusted_pot + 1) * 4) 
    sequence = log   #message log of game actions
    initial_pot = round_state.pips[active]
    result = ""
    current_number = ""
    current_pot = initial_pot
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
    
    return f"[{result}]"

# Example usage
def abstractbetting(datadict): #abstracts raises into floor(math.log(number/adjusted_pot + 1) * 4) 
    sequence = datadict["Sequences"]   #message log of game actions
    initial_pot = datadict["Pot"]
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
    
    return f"[{result}]"


def abstractioncards(data_dict):
    """
    Categorize a poker hand based on its type and rank.

    Args:
        hand_cards (list of eval7.Card): Player's two hole cards.
        board_cards (list of eval7.Card): Community cards on the board.

    Returns:
        dict: Categorized hand information. (eventually transformed into a string)
    """ 

    all_cards = data_dict['Private'] + data_dict['Public'] #data_dict[priv] are user cards, else is non user
    eval7allcards = [eval7.Card(s) for s in all_cards] #making eval7 cards
    hand_type = eval7.handtype(eval7.evaluate(eval7allcards)) #eval7 can identify what type of hand it is but idk if strength of hand like Ace pair etc. Maybe some func

    print(data_dict)
    community_cards = [eval7.Card(c) for c in data_dict['Public']]

    rank_counts = {} #Rank counts for whole board
    suits_count = {}
    for card in eval7allcards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
    rank_counts_board = {} #Rank counts  for just the board, also gives implied information on how paired the board is
    suit_counts_board = {}  # Suit counts for just the board

    for card in [eval7.Card(s) for s in data_dict["Public"]]:
        rank_counts_board[card.rank] = rank_counts.get(card.rank, 0) + 1
        #print(rank_counts_board)
        suit_counts_board[card.suit] = suit_counts_board.get(card.suit, 0) + 1

    if hand_type == "High Card": #str starts w/ 1
        highest_card = max(all_cards, key=lambda card: card.rank)
        if highest_card == 14:
            abtype = "11"
        if highest_card == 13:
            abtype = "12"
        else:
            abtype = "13"
    elif hand_type == "Pair" or hand_type == "Two Pair" or hand_type == "Trips": #str starts w/ 2
        # Extract community cards
        
        # Find the highest pair
        pairs = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
        board_pairs = sorted([rank for rank, count in rank_counts_board.items() if count >= 2], reverse=True)

        highest_pair = pairs[0]  # The highest ranked pair
        if hand_type == "Pair" or hand_type == "Two Pair":
            # Determine objective value Eval7 treats 2 = 0, 3 = 1, ... A = 12
            if highest_pair <= 6:
                objective_value = 0
            elif highest_pair <= 9:
                objective_value = 1
            elif highest_pair <= 11:
                objective_value = 2
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
        if hand_type == "Pair":
            abtype = "2"
        if hand_type == "Two Pair":
            abtype = "3"
        if hand_type == "Pair" or "Two Pair":
            if relative_value == "Low":
                abtype = abtype + str(1+objective_value*3)
            if relative_value == "Middle":
                abtype = abtype + str(2+objective_value*3)
            if relative_value == "Top Pair":
                abtype = abtype + str(3+objective_value*3)
            if relative_value == "Overpair" and objective_value == "Overpair":
                abtype = abtype + "0"
        if hand_type == 'Trips':
            abtype = "4"
            if relative_value == "Set":
                abtype = abtype + str(objective_value * 3 + 1)
            else:
                abtype = abtype + str(objective_value * 3 + 0)

    elif hand_type == "Full House":

        # Count the number of ranks that appear exactly twice
        num_pairs = sum(1 for count in rank_counts_board.values() if count == 2)

        # Check if there is at least one triplet
        has_trip = any(count == 3 for count in rank_counts_board.values())

        if num_pairs >= 2 or has_trip:
            type = 1
            
        else:
            type = 2
        
        abtype = "5" + str(type)

    elif hand_type == "Straight":
        missingcards = missing_for_straight_with_debug(rank_counts_board)
        if missingcards == 2:
            type = 1
        if missingcards == 1:
            type = 2
        if missingcards == 0:
            type = 3
        abtype = "4" + str(type)
    
    elif hand_type == "Flush":
        boardsuited = max(suit_counts_board.values(), default=0)
        if boardsuited == 3:
            cardsused = "2"
        if boardsuited == 4:
            cardused = "1"
        if boardsuited == 5:
            cardsused = "0"
        abtype =  "6" + cardsused
    
    elif hand_type == "Quads":
        abtype =  "7"

    elif hand_type == "Straight Flush" or "Royal Flush":
        abtype = "8"
    
    #flushdraw and straightdrawcode

    flushdraw = str(5 - max(suits_count.values(), default=0))
    straightdraw = str(5 - missing_for_straight_with_debug(rank_counts))

    num_pairs = sum(1 for count in rank_counts_board.values() if count == 2)
    boardsuited = max(suit_counts_board.values(), default=0)
    missingcard = missing_for_straight_with_debug(rank_counts_board)
    return f"[{abtype}{flushdraw}{straightdraw}]" + f"[{str(missingcard) +str(boardsuited)+str(num_pairs)}]"

def generate_empty_strategy_and_regret():
    strategy = {}
    regret = {}
    # Betting log:
    permutations = []
    # Generate all permutations for lengths from 1 to max_length
    for length in range(1, 7 + 1):
        # Use product to generate strings of 'c' and 'f' of the given length
        permutations.extend("".join(p) for p in itertools.product("cr", repeat=length))

    num1 = list(range(0, 60 + 1))
    num2 = list(range(0, 4 + 1))
    num3 = list(range(0, 4 + 1))

    num4 = list(range(0, 14 + 1))
    num5 = [0, 1]
    num6 = [0, 1, 2]

    for n1,  n2, n3, n4, n5, n6, log in itertools.product(num1, num2, num3, num4, num5, num6, permutations):
        string = f"[{n1}{n2}{n3}][{n4}{n5}{n6}][{log}]"

        strategy[string] = [1/4 for _ in range(4)]
        regret[string] = [0 for _ in range(4)]
    
    return strategy, regret

s, r = generate_empty_strategy_and_regret()




# Example usage

poker_string = "[Round 0][Player: 0][Pot: 40000][Money: 19900 0][Private: 8c8h][Public: 3sJc5d8sJs][Sequences: cr20000]"

datadict = parse_poker_string(poker_string) #Transforming Raw Game State into variables
result = abstractioncards(datadict)
result2 = abstractbetting(datadict)

print(result + result2)

# Example usage
"""
poker_string = "[Round 0][Player: 0][Pot: 40000][Money: 19900 0][Private: 8c8h][Public: 3sJc5d8sJs][Sequences: cr20000]"

datadict = parse_poker_string(poker_string) #Transforming Raw Game State into variables
result = abstractioncards(datadict)
result2 = abstractbetting(datadict)

print(result + result2)
"""
# Print the parsed dictionary
