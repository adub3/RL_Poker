import re
import eval7
import math
import itertools

#Notes:
# trips and top pair same some similar code that could be combined maybe better to write a func that just sees the highest card on the board and matches it to the card in hand or smtn
#not rlly sure whats best here

#(1: High Card, 2: Pair, 3: Trips, 4: Straight, 5: Flush, 6: Full House, 7: Quads, 8: Straight Flush and ...)

RANK_SYMBOLS_BY_EVAL7 = "23456789TJQKA"
RANK_VALUE_BY_SYMBOL = {rank: index for index, rank in enumerate(RANK_SYMBOLS_BY_EVAL7)}
BRACKET_RE = re.compile(r"\[(.*?)\]")
PREFLOP_SEQUENCE_TOKEN_RE = re.compile(r"r\d+|[cf]")

def parse_poker_string(poker_string):
    matches = BRACKET_RE.findall(poker_string)
    
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
      - eval7 ranks are used: 2 is 0, 3 is 1, ..., Ace is 12.
      - A straight must be exactly 5 consecutive values.
    """
    min_missing = float('inf')
    best_sequence = None
    
    straights = [[0, 1, 2, 3, 12]]  # A-2-3-4-5
    straights.extend([list(range(start, start + 5)) for start in range(0, 9)])

    for straight in straights:
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
                # processed_number = math.floor(math.log(number/adjusted_pot + 1) * 4)
                # result += str(processed_number)
                current_pot += number  # Update pot for next calculations
                current_number = ""
                bet_index += 1
            result += 'r'
        elif char.isdigit():
            current_number += char
    
    # # Process any remaining number at the end
    # if current_number:
    #     number = int(current_number)
    #     processed_number = math.floor(math.log(number/current_pot + 1) * 4)
    #     result += str(processed_number)
    
    return f"[{result}]"

# Example usage
def abstractbetting(datadict): #abstracts raises into floor(math.log(number/adjusted_pot + 1) * 4) 
    result = []
    for char in datadict["Sequences"]:
        if char == "c" or char == "r":
            result.append(char)
            if len(result) >= 3:
                break
    return f"[{''.join(result)}]"


def preflop_betting_context(data_dict, big_blind=100):
    player = int(data_dict.get("Player", 0) or 0)
    money = data_dict.get("Money") or []
    sequence = data_dict.get("Sequences", "") or ""
    to_call = _to_call_from_money(money, player)
    effective_stack = _effective_stack_before_current_action(money, to_call)
    sequence_bucket = _preflop_sequence_bucket(sequence, effective_stack, big_blind)

    return (
        f"[pos:P{player}]"
        f"[tc:{_bb_bucket(to_call, big_blind)}]"
        f"[eff:{_stack_bucket(effective_stack, big_blind)}]"
        f"[seq:{sequence_bucket}]"
    )


def _to_call_from_money(money, player):
    if len(money) < 2:
        return 0
    opponent = 1 - player
    return max(0, int(money[player]) - int(money[opponent]))


def _effective_stack_before_current_action(money, to_call):
    if len(money) < 2:
        return 0
    return max(0, min(int(money[0]), int(money[1])) + int(to_call))


def _preflop_sequence_bucket(sequence, effective_stack, big_blind):
    tokens = PREFLOP_SEQUENCE_TOKEN_RE.findall(sequence)
    if not tokens:
        return "open"

    result = []
    previous_raise = None
    for token in tokens:
        if token in ("c", "f"):
            result.append(token)
            continue

        amount = int(token[1:])
        result.append(
            "r:" + _raise_size_bucket(
                amount,
                previous_raise=previous_raise,
                effective_stack=effective_stack,
                big_blind=big_blind,
            )
        )
        previous_raise = amount

    return "|".join(result)


def _raise_size_bucket(amount, previous_raise, effective_stack, big_blind):
    if effective_stack and amount >= 0.95 * effective_stack:
        return "jam"

    candidates = {
        "2bb": 2.0 * big_blind,
        "2_5bb": 2.5 * big_blind,
        "3bb": 3.0 * big_blind,
        "4bb": 4.0 * big_blind,
    }

    if previous_raise:
        candidates["3x"] = 3.0 * previous_raise
        candidates["4x"] = 4.0 * previous_raise

    if effective_stack:
        candidates["25eff"] = 0.25 * effective_stack
        candidates["50eff"] = 0.50 * effective_stack

    return min(candidates, key=lambda label: abs(candidates[label] - amount))


def _bb_bucket(amount, big_blind):
    if amount <= 0:
        return "0bb"
    value = amount / big_blind
    buckets = (0.5, 1, 2, 2.5, 3, 4, 6, 8, 10, 15, 25, 50, 100)
    nearest = min(buckets, key=lambda bucket: abs(bucket - value))
    return _format_bucket(nearest, "bb")


def _stack_bucket(amount, big_blind):
    if amount <= 0:
        return "0bb"
    value = amount / big_blind
    buckets = (10, 20, 30, 40, 50, 75, 100, 150, 200)
    nearest = min(buckets, key=lambda bucket: abs(bucket - value))
    return _format_bucket(nearest, "bb")


def _format_bucket(value, suffix):
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{str(value).replace('.', '_')}{suffix}"


def preflop_lossless_cards(data_dict):
    private_cards = data_dict.get("Private", [])
    if len(private_cards) != 2:
        raise ValueError("preflop lossless abstraction requires exactly two private cards")

    first, second = private_cards
    first_rank, first_suit = first[0], first[1]
    second_rank, second_suit = second[0], second[1]

    ranked = sorted(
        [first_rank, second_rank],
        key=lambda rank: RANK_VALUE_BY_SYMBOL[rank],
        reverse=True,
    )
    suitedness = "s" if first_suit == second_suit else "o"
    if ranked[0] == ranked[1]:
        return f"[PF:{ranked[0]}{ranked[1]}]"
    return f"[PF:{ranked[0]}{ranked[1]}{suitedness}]"


def postflop_lossy_cards(data_dict):
    return abstractioncards(data_dict)


def abstractioncards_street_aware(data_dict):
    if not data_dict.get("Public"):
        return preflop_lossless_cards(data_dict)
    return postflop_lossy_cards(data_dict)


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

    # print(data_dict)
    community_cards = [eval7.Card(c) for c in data_dict['Public']]

    rank_counts = {} #Rank counts for whole board
    suits_count = {}
    for card in eval7allcards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
    rank_counts_board = {} #Rank counts  for just the board, also gives implied information on how paired the board is
    suit_counts_board = {}  # Suit counts for just the board

    for card in [eval7.Card(s) for s in data_dict["Public"]]:
        rank_counts_board[card.rank] = rank_counts_board.get(card.rank, 0) + 1
        #print(rank_counts_board)
        suit_counts_board[card.suit] = suit_counts_board.get(card.suit, 0) + 1

    if hand_type == "High Card": #str starts w/ 1
        highest_rank = max(card.rank for card in eval7allcards)
        if highest_rank == 12:
            abtype = "11"
        elif highest_rank == 11:
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
                objective_value = 'Overpair'
                objective_value = objective_value.strip()
        else:
            if board_pairs:
                typetrips = "Set"
            else:
                typetrips = "Trips"
        # Get board ranks
        board_ranks = sorted([card.rank for card in community_cards], reverse=True)
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
            relative_value = 'Overpair'
        if hand_type == "Pair":
            abtype = "2"
        if hand_type == "Two Pair":
            abtype = "3"
        if hand_type in ['Pair', 'Two Pair']:
            # if ((objective_value == 'Overpair') or (relative_value == 'Overpair')):
            if objective_value == 'Overpair':
                abtype = abtype + "0"
            if relative_value == "Low":
                abtype = abtype + '1'
            if relative_value == "Middle":
                abtype = abtype + '2'
            if relative_value == "Top Pair":
                abtype = abtype + '3'
        if hand_type == 'Trips':
            abtype = "4"
            if typetrips == "Set":
                abtype = abtype + "1"
            else:
                abtype = abtype + "0"

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
        elif missingcards == 1:
            type = 2
        elif missingcards == 0:
            type = 3
        else:
            type = 0
        abtype = "4" + str(type)
    
    elif hand_type == "Flush":
        boardsuited = max(suit_counts_board.values(), default=0)
        if boardsuited == 3:
            cardsused = '2'
        elif boardsuited == 4:
            cardsused = '1'
        elif boardsuited == 5:
            cardsused = "0"
        else:
            cardsused = "3"
        abtype =  "6" + cardsused
    
    elif hand_type == "Quads":
        abtype =  "7"

    elif hand_type in ("Straight Flush", "Royal Flush"):
        abtype = "8"
    
    #flushdraw and straightdrawcode

    flushdraw = str(max((5 - max(suits_count.values())), 0))
    straightdraw = str(5 - missing_for_straight_with_debug(rank_counts))

    num_pairs = sum(1 for count in rank_counts_board.values() if count == 2)
    boardsuited = max(suit_counts_board.values(), default=0)
    missingcard = missing_for_straight_with_debug(rank_counts_board)
    return f"[{abtype[0:2]}{flushdraw}{straightdraw}]" + f"[{str(missingcard) +str(boardsuited)}]"

def generate_empty_strategy_and_regret():
    strategy = {}
    regret = {}
    # Betting log:
    permutations = []
    # Generate all permutations for lengths from 1 to max_length
    for length in range(1, 3 + 1):
        # Use product to generate strings of 'c' and 'f' of the given length
        permutations.extend("".join(p) for p in itertools.product("cr", repeat=length))

    permutations += [""]

    num1 = list(range(0, 70 + 1))
    num2 = list(range(0, 4 + 1))
    num3 = list(range(0, 5 + 1))

    num4 = list(range(0, 14 + 1))
    num5 = [0, 1, 2, 3, 4, 5]
    # num6 = [0, 1, 2, 3, 4, 5]

    for n1,  n2, n3, n4, n5, log in itertools.product(num1, num2, num3, num4, num5, permutations):
        string = f"[{n1}{n2}{n3}][{n4}{n5}][{log}]"

        strategy[string] = [1/4 for _ in range(4)]
        regret[string] = [0 for _ in range(4)]
    
    return strategy, regret

if __name__ == "__main__":
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
