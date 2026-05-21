import math
import re
from collections import defaultdict


RANKS = "AKQJT98765432"
HAND_RE = re.compile(r"\[PF:([2-9TJQKA]{2}[so]?)\]")
CONTEXT_RE = re.compile(r"\[PF:[^\]]+\](.*)")


def extract_hand_class(infoset):
    match = HAND_RE.search(infoset)
    return match.group(1) if match else None


def extract_betting_context(infoset):
    match = CONTEXT_RE.search(infoset)
    return match.group(1) if match else ""


def hand_matrix_position(hand_class):
    if len(hand_class) == 2:
        rank = RANKS.index(hand_class[0])
        return rank, rank

    high = RANKS.index(hand_class[0])
    low = RANKS.index(hand_class[1])
    if hand_class.endswith("s"):
        return high, low
    return low, high


def empty_hand_matrix(default=""):
    return [[default for _ in RANKS] for _ in RANKS]


def hand_matrix_to_rows(matrix):
    rows = []
    for rank, values in zip(RANKS, matrix):
        rows.append([rank] + values)
    return rows


def aggregate_preflop_strategy(table):
    aggregates = defaultdict(lambda: defaultdict(float))
    contexts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for infoset, node in table.data.items():
        hand_class = extract_hand_class(infoset)
        if not hand_class:
            continue
        context = extract_betting_context(infoset)
        strategy_sum = node.get("strategy_sum", {})
        strategy = node.get("strategy", {})
        source = strategy_sum if sum(float(v) for v in strategy_sum.values()) > 0 else strategy
        for action, mass in source.items():
            value = float(mass)
            aggregates[hand_class][action] += value
            contexts[hand_class][context][action] += value

    return aggregates, contexts


def normalized_action_probs(action_mass):
    total = sum(float(value) for value in action_mass.values())
    if total <= 0:
        return {}
    return {action: float(value) / total for action, value in action_mass.items()}


def action_group(action):
    key = str(action)
    if key in ("0", "fold"):
        return "fold"
    if key in ("1", "call", "check", "call_check"):
        return "call_check"
    if key == "jam":
        return "all_in"
    try:
        return "all_in" if int(key) >= 10_000_000 else "raise"
    except ValueError:
        return "raise"


def grouped_action_probs(action_mass):
    grouped = defaultdict(float)
    for action, mass in action_mass.items():
        grouped[action_group(action)] += float(mass)
    return normalized_action_probs(grouped)


def entropy(action_mass):
    probs = normalized_action_probs(action_mass)
    return -sum(prob * math.log2(prob) for prob in probs.values() if prob > 0)


def preflop_coverage_metrics(table):
    hand_classes = set()
    preflop_infosets = 0
    entropy_total = 0.0

    aggregates, _ = aggregate_preflop_strategy(table)
    for infoset in table.data:
        hand_class = extract_hand_class(infoset)
        if hand_class:
            preflop_infosets += 1
            hand_classes.add(hand_class)

    for action_mass in aggregates.values():
        entropy_total += entropy(action_mass)

    return {
        "preflop_infosets": preflop_infosets,
        "starting_hand_classes": len(hand_classes),
        "avg_strategy_entropy_by_hand": (
            entropy_total / len(aggregates) if aggregates else 0.0
        ),
    }
