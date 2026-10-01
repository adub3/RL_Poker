import re
import math
import itertools
from bisect import bisect_left
from functools import lru_cache

# Hand type integers (0-8, higher = better)
# 0=High Card  1=Pair  2=Two Pair  3=Trips  4=Straight
# 5=Flush  6=Full House  7=Quads  8=Straight Flush

RANK_SYMBOLS_BY_EVAL7 = "23456789TJQKA"
RANK_VALUE_BY_SYMBOL = {rank: index for index, rank in enumerate(RANK_SYMBOLS_BY_EVAL7)}

_RANK_IDX = RANK_VALUE_BY_SYMBOL   # alias: '2'→0 … 'A'→12
_SUIT_IDX = {'c': 0, 'd': 1, 'h': 2, 's': 3}


# ── Pure-Python hand classifier ───────────────────────────────────────────────

def _straight_high(rank_frozenset):
    """Highest card of the best straight in rank_frozenset, or -1 if none."""
    rs = sorted(rank_frozenset)
    if len(rs) < 5:
        return -1
    best = -1
    # Ace-low (A-2-3-4-5): ace treated as rank -1
    if 12 in rank_frozenset and {0, 1, 2, 3}.issubset(rank_frozenset):
        best = 3  # five-high
    for i in range(len(rs) - 4):
        if rs[i + 4] - rs[i] == 4:   # 5 unique consecutive (frozenset has no dupes)
            best = max(best, rs[i + 4])
    return best


def _straight_draw(rank_frozenset):
    """
    0 = no draw (fewer than 3 cards toward any straight)
    1 = gutshot (4-to-a-straight with 1 hole in the middle)
    2 = OESD / double-gutter (4-to-a-straight open-ended, or already made)
    """
    rs = rank_frozenset
    # Ace can be high or low
    extended = rs | ({-1} if 12 in rs else frozenset())
    best = 0
    for start in range(-1, 9):   # windows A-5 through T-A
        window = list(range(start, start + 5))
        present = sum(1 for r in window if r in extended)
        if present >= 4:
            # check if it's open-ended (both ends open) or gutshot
            low_open = window[0] - 1 not in extended
            high_open = window[-1] + 1 not in extended
            if present == 5:
                return 2  # made straight
            # 4-to-straight: open-ended if gaps are at the ends
            gap_positions = [i for i, r in enumerate(window) if r not in extended]
            if len(gap_positions) == 1:
                gap = gap_positions[0]
                if gap == 0 or gap == 4:
                    best = max(best, 2)   # open-ended
                else:
                    best = max(best, 1)   # gutshot
    return best


def _board_straight_texture(board_rank_frozenset):
    """How many cards the best board partial-straight is missing (0-5)."""
    if not board_rank_frozenset:
        return 5
    best_present = 0
    rs = board_rank_frozenset
    extended = rs | ({-1} if 12 in rs else frozenset())
    for start in range(-1, 9):
        window = range(start, start + 5)
        present = sum(1 for r in window if r in extended)
        best_present = max(best_present, present)
    return max(0, 5 - best_present)


@lru_cache(maxsize=131072)
def _classify_cards(private_tuple, public_tuple):
    """
    Pure-Python postflop card abstraction.  Cached on immutable card tuples.

    Returns a 2-bracket string:
      [HTSD][BQ]
        H = hand type 0-8  (HC / Pair / 2P / Trips / Str / Flush / FH / Quads / SF)
        T = top-bucket  (0=strong/overpair, 1=mid/top-pair, 2=weak/underpair)
        S = straight draw  (0=none, 1=gutshot, 2=OESD/made)
        D = flush draw  (0=none, 1=backdoor 3-flush, 2=4-flush draw, 3=flush)
        B = board straight texture  (cards missing from best board straight, 0-5)
        Q = board suit texture  (max same-suit cards on board, 0-5)
    """
    all_cards = private_tuple + public_tuple
    board_cards = public_tuple

    ranks = [_RANK_IDX[c[0]] for c in all_cards]
    suits = [_SUIT_IDX[c[1]] for c in all_cards]
    board_ranks = [_RANK_IDX[c[0]] for c in board_cards]
    board_suits_raw = [_SUIT_IDX[c[1]] for c in board_cards]

    # Rank / suit frequency
    rc = {}
    for r in ranks:
        rc[r] = rc.get(r, 0) + 1
    sc = [0, 0, 0, 0]
    for s in suits:
        sc[s] += 1
    bsc = [0, 0, 0, 0]
    for s in board_suits_raw:
        bsc[s] += 1

    counts_desc = sorted(rc.values(), reverse=True)
    rank_set = frozenset(rc)
    board_rank_set = frozenset(_RANK_IDX[c[0]] for c in board_cards)

    # Flush / straight checks
    flush_suit = next((i for i, c in enumerate(sc) if c >= 5), -1)
    str_high = _straight_high(rank_set)

    sf_high = -1
    if flush_suit >= 0:
        fs_ranks = frozenset(ranks[i] for i, s in enumerate(suits) if s == flush_suit)
        sf_high = _straight_high(fs_ranks)

    # ── Hand type ──────────────────────────────────────────────────────────
    if sf_high >= 0:
        hand_type = 8
    elif counts_desc[0] == 4:
        hand_type = 7
    elif counts_desc[0] == 3 and len(counts_desc) > 1 and counts_desc[1] >= 2:
        hand_type = 6
    elif flush_suit >= 0:
        hand_type = 5
    elif str_high >= 0:
        hand_type = 4
    elif counts_desc[0] == 3:
        hand_type = 3
    elif counts_desc[0] == 2 and len(counts_desc) > 1 and counts_desc[1] == 2:
        hand_type = 2
    elif counts_desc[0] == 2:
        hand_type = 1
    else:
        hand_type = 0

    # ── Top bucket (relative strength within type) ─────────────────────────
    board_max = max(board_rank_set) if board_rank_set else -1
    if hand_type <= 2:   # HC, Pair, Two Pair
        top_rank = max(ranks)
        if top_rank > board_max:
            top = 0   # overcards / overpair / top-two
        elif top_rank == board_max:
            top = 1   # top pair / matched top card
        else:
            top = 2   # weak pair / underpair
    else:
        top = 0   # trips+ don't need sub-bucketing here

    # ── Draw quality ───────────────────────────────────────────────────────
    max_suited = max(sc)
    if max_suited >= 5:
        fd = 3   # made flush
    elif max_suited == 4:
        fd = 2   # flush draw
    elif max_suited == 3:
        fd = 1   # backdoor
    else:
        fd = 0

    sd = _straight_draw(rank_set)

    # ── Board texture ──────────────────────────────────────────────────────
    board_str_miss = _board_straight_texture(board_rank_set)
    board_suited = max(bsc) if bsc else 0

    return f"[{hand_type}{top}{fd}{sd}][{board_str_miss}{board_suited}]"
BRACKET_RE = re.compile(r"\[(.*?)\]")
PREFLOP_SEQUENCE_TOKEN_RE = re.compile(r"r\d+|[cf]")

def parse_poker_string(poker_string):
    """Parse "[Key: value][Key: value]..." info strings into a dict.

    Not cached: info strings include the cards, so they almost never repeat.
    """
    parsed_data = {}
    if not poker_string.startswith("[") or not poker_string.endswith("]"):
        return parsed_data

    for match in poker_string[1:-1].split("]["):
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
        elif value.isdigit():
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
# Not cached: pot-relative bet sizes make nearly every postflop betting state
# unique (a cache here hit 6.5% of the time).
def _postflop_context_cached(sequences, pot, money_tuple, player, street):
    """
    Postflop betting infoset key component.  All args are hashable for caching.

    Bet sizes are bucketed as fractions of the pot at the time of the bet:
      x  = check          c  = call
      rs = small  (<30%)  rh = half (30-60%)
      rp = pot   (60-110%) ro = overbet (>110%)  rj = jam (≥90% eff stack)
    """
    # Extract current street's action sequence (after last '|')
    idx = sequences.rfind("|")
    curr_seq = sequences[idx + 1:] if idx >= 0 else sequences
    tokens = PREFLOP_SEQUENCE_TOKEN_RE.findall(curr_seq)

    # Compute each player's total committed this street so we can recover
    # the initial pot (pot before any betting this street).
    commits = [0, 0]
    cur = 0
    for token in tokens:
        if token == "c":
            commits[cur] = commits[1 - cur]
            cur = 1 - cur
        elif token != "f":   # rN
            commits[cur] = int(token[1:])
            cur = 1 - cur
    initial_pot = max(0, pot - commits[0] - commits[1])
    eff_stack = max(0, min(money_tuple)) if money_tuple else 0

    # Re-walk tokens to generate bucketed labels while tracking running pot.
    result = []
    pot_now = initial_pot
    commits = [0, 0]
    cur = 0
    for token in tokens:
        if token == "f":
            result.append("f")
            break
        elif token == "c":
            to_call = commits[1 - cur] - commits[cur]
            result.append("x" if to_call == 0 else "c")
            pot_now += to_call
            commits[cur] = commits[1 - cur]
            cur = 1 - cur
        else:   # rN  — N = total committed by this player this street
            N = int(token[1:])
            # Raise amount above the opponent's current commitment
            raise_size = N - commits[1 - cur]
            if eff_stack > 0 and N >= int(0.90 * eff_stack):
                b = "j"
            elif pot_now > 0:
                frac = raise_size / pot_now
                if frac < 0.30:
                    b = "s"
                elif frac < 0.60:
                    b = "h"
                elif frac < 1.10:
                    b = "p"
                else:
                    b = "o"
            else:
                b = "p"   # first bet into empty pot: treat as pot-sized
            result.append(f"r{b}")
            pot_now += N - commits[cur]
            commits[cur] = N
            cur = 1 - cur

    seq_str = "".join(result) if result else "open"
    return f"[pos:P{player}][st:{street}][{seq_str}]"


# v4 bet-size buckets: raise over the call as a fraction of the pot after
# calling, the same measure the action menu uses. Boundaries are the geometric
# midpoints between the menu's sizes (1/3, 1/2, pot, 2x pot).
_V4_SIZE_BUCKETS = ((math.sqrt(1 / 6), "s"), (math.sqrt(1 / 2), "h"), (math.sqrt(2), "p"))
# Stack-to-pot ratio at the start of the street: < 1, < 3, < 8, and deeper.
_V4_SPR_BUCKETS = (1, 3, 8)


def _postflop_context_v4(sequences, player, street, big_blind, starting_stack):
    """
    Postflop betting infoset key component for v4 runs.

    Sequences hold whole-hand totals (rN = N chips committed in the hand), and
    both players start each street level with the last raise of earlier
    streets (the big blind if there was none). Labels:
      x = check  c = call  f = fold
      rs = up to ~40% pot  rh = up to ~70%  rp = up to ~1.4x  ro = bigger
      rj = all-in
    plus the stack-to-pot ratio bucket at the start of the street.
    """
    streets = sequences.split("|")
    earlier = [int(token[1:]) for token in PREFLOP_SEQUENCE_TOKEN_RE.findall("|".join(streets[:-1]))
               if token[0] == "r"]
    start = earlier[-1] if earlier else big_blind
    spr = (starting_stack - start) / (2 * start)
    spr_bucket = sum(spr >= bound for bound in _V4_SPR_BUCKETS)

    result = []
    commits = [start, start]
    cur = 0
    for token in PREFLOP_SEQUENCE_TOKEN_RE.findall(streets[-1]):
        if token == "f":
            result.append("f")
            break
        if token == "c":
            result.append("x" if commits[cur] == commits[1 - cur] else "c")
            commits[cur] = commits[1 - cur]
        else:
            amount = int(token[1:])
            if amount >= starting_stack:
                size = "j"
            else:
                fraction = (amount - commits[1 - cur]) / (2 * commits[1 - cur])
                size = next((label for bound, label in _V4_SIZE_BUCKETS if fraction < bound), "o")
            result.append(f"r{size}")
            commits[cur] = amount
        cur = 1 - cur

    seq_str = "".join(result) if result else "open"
    return f"[pos:P{player}][st:{street}][spr:{spr_bucket}][{seq_str}]"


# v6 runs differ only in card buckets: v6c uses the hand categories of
# _classify_cards (as v4), v6e the equity buckets of card_buckets.py (as v5).
V6_SIZINGS = ("v6c", "v6e")

# v6 seat order: P1 (small blind) acts first preflop, P0 first on later
# streets (game_config firstPlayer "2 1 1 1"). The info string does not name
# who made each action, so v6 reads it from the order of play.
_V6_FIRST_ACTOR = {0: 1}  # street -> first player; postflop streets: P0


def _street_summary(tokens, first_actor):
    """(raises on the street, who made the last raise or None)."""
    raises, last, actor = 0, None, first_actor
    for token in tokens:
        if token == "f":
            break
        if token[0] == "r":
            raises += 1
            last = actor
        actor = 1 - actor
    return raises, last


def _postflop_context_v6(sequences, player, street, big_blind, starting_stack):
    """
    Postflop betting key for v6 runs: v4's key for the current street plus
    the hand's history, which earlier keys forgot after each street.

      [pf:...]  preflop pot: L limped, S single raise, 3 3-bet, 4 4-bet or
                more, followed by the last raiser (0 or 1; none if limped)
      [h:...]   each earlier postflop street: x checked through, b bet and
                called, r raised (2+ raises), followed by who bet last
    """
    streets = sequences.split("|")
    raises, last = _street_summary(PREFLOP_SEQUENCE_TOKEN_RE.findall(streets[0]), _V6_FIRST_ACTOR[0])
    preflop = "L" if raises == 0 else "S3"[min(raises, 2) - 1] + str(last) if raises < 3 else f"4{last}"
    history = []
    for earlier in streets[1:-1]:
        raises, last = _street_summary(PREFLOP_SEQUENCE_TOKEN_RE.findall(earlier), 0)
        history.append("x" if raises == 0 else f"{'b' if raises == 1 else 'r'}{last}")
    current = _postflop_context_v4(sequences, player, street, big_blind, starting_stack)
    split = current.index("[spr:")
    return f"{current[:split]}[pf:{preflop}][h:{','.join(history) or '-'}]{current[split:]}"


def postflop_betting_context(data_dict, bet_sizing=None, big_blind=100, starting_stack=10_000):
    """Postflop betting key. Runs before v4 keep the original (size-blind) keys:
    they read every first bet on a street as pot-sized, since they treated the
    whole-hand totals in Sequences and Pot as amounts for the street. v6 adds
    the hand's history (see _postflop_context_v6)."""
    sequences = data_dict.get("Sequences", "") or ""
    player = int(data_dict.get("Player", 0) or 0)
    street = int(data_dict.get("Round", 1) or 1)
    if bet_sizing in V6_SIZINGS:
        return _postflop_context_v6(sequences, player, street, big_blind, starting_stack)
    if bet_sizing in ("v4", "v5"):
        return _postflop_context_v4(sequences, player, street, big_blind, starting_stack)
    pot = int(data_dict.get("Pot", 0) or 0)
    money = tuple(int(m) for m in (data_dict.get("Money") or [0, 0]))
    return _postflop_context_cached(sequences, pot, money, player, street)


def abstractbetting(datadict):
    """Legacy alias — delegates to postflop_betting_context."""
    return postflop_betting_context(datadict)


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

    best_label = "2bb"
    best_dist = abs(2.0 * big_blind - amount)

    for label, target in (
        ("2_5bb", 2.5 * big_blind),
        ("3bb", 3.0 * big_blind),
        ("4bb", 4.0 * big_blind),
    ):
        dist = abs(target - amount)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if previous_raise:
        for label, target in (("3x", 3.0 * previous_raise), ("4x", 4.0 * previous_raise)):
            dist = abs(target - amount)
            if dist < best_dist:
                best_dist = dist
                best_label = label

    if effective_stack:
        for label, target in (("25eff", 0.25 * effective_stack), ("50eff", 0.50 * effective_stack)):
            dist = abs(target - amount)
            if dist < best_dist:
                best_dist = dist
                best_label = label

    return best_label


_BB_BUCKETS = (0.5, 1, 2, 2.5, 3, 4, 6, 8, 10, 15, 25, 50, 100)
_STACK_BUCKETS = (10, 20, 30, 40, 50, 75, 100, 150, 200)


def _nearest_in_sorted(buckets, value):
    idx = bisect_left(buckets, value)
    if idx == 0:
        return buckets[0]
    if idx == len(buckets):
        return buckets[-1]
    lo, hi = buckets[idx - 1], buckets[idx]
    return lo if abs(lo - value) <= abs(hi - value) else hi


def _bb_bucket(amount, big_blind):
    if amount <= 0:
        return "0bb"
    return _format_bucket(_nearest_in_sorted(_BB_BUCKETS, amount / big_blind), "bb")


def _stack_bucket(amount, big_blind):
    if amount <= 0:
        return "0bb"
    return _format_bucket(_nearest_in_sorted(_STACK_BUCKETS, amount / big_blind), "bb")


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


def abstractioncards_street_aware(data_dict, bet_sizing=None):
    """Card part of the infoset key. v5 runs use board-relative strength and
    potential buckets postflop (card_buckets.py); earlier runs use the
    hand-category buckets of _classify_cards."""
    if not data_dict.get("Public"):
        return preflop_lossless_cards(data_dict)
    if bet_sizing in ("v5", "v6e"):
        from card_buckets import card_bucket

        return card_bucket(tuple(data_dict.get("Private") or ()), tuple(data_dict["Public"]))
    return postflop_lossy_cards(data_dict)


def abstractioncards(data_dict):
    """Postflop card abstraction.  Delegates to _classify_cards (cached)."""
    private = tuple(data_dict.get("Private") or [])
    public = tuple(data_dict.get("Public") or [])
    return _classify_cards(private, public)

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
