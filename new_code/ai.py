import csv
import gzip
import json
import os
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from abstraction import (
    abstractioncards_street_aware,
    parse_poker_string,
    postflop_betting_context,
    preflop_betting_context,
)
from const import LEGACY_FIRST_PLAYER, game_config


DEFAULT_PRUNE_THRESHOLD = -300_000_000
DEFAULT_POT_FRACTIONS = (1 / 3, 1 / 2, 3 / 4, 1.0, 1.5, 2.0)
DEFAULT_STACK_FRACTIONS = (1 / 4, 1 / 2, 3 / 4)
# "legacy" is the postflop sizing used before the fix; tables trained with it
# only line up with that sizing. Runs record theirs in run_manifest.json.
# v5 keeps v4's menu and betting keys and replaces the postflop card buckets
# with board-relative strength and potential (card_buckets.py).
# v6c / v6e add the hand's history to postflop betting keys; they differ only
# in card buckets (v4's hand categories vs v5's equity buckets).
BET_SIZINGS = ("legacy", "v2", "v3", "v4", "v5", "v6c", "v6e")
DEFAULT_BET_SIZING = "v6c"
# v3 caps raises per street; after the cap a player can only fold, call or
# jam. Without a cap, min-raise wars made most of the v2 tree (68% of
# postflop infosets had 3+ raises on the current street).
# Sizings that cap raises per street (v4 is v3's menu with fixed postflop keys).
RAISE_CAP_SIZINGS = ("v3", "v4", "v5", "v6c", "v6e")
PREFLOP_RAISE_CAP = 4   # open, 3-bet, 4-bet, 5-bet
POSTFLOP_RAISE_CAP = 3  # bet, raise, re-raise
# v3 raise sizes when facing a bet on a postflop street.
POSTFLOP_FACING_BET_FRACTIONS = (("bet_pot", 1.0),)
POSTFLOP_POT_FRACTIONS = (
    ("bet_33", 1 / 3),
    ("bet_50", 1 / 2),
    ("bet_pot", 1.0),
    ("bet_2x", 2.0),
)
RANK_VALUE_BY_CARD = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
_ACTION_KEY_CACHE: dict = {}

ACTION_BUCKETS = (
    "fold",
    "call/check",
    "min_raise",
    "open_2_5bb",
    "open_3bb",
    "raise_2_5bb",
    "raise_3bb",
    "raise_3x",
    "raise_4x",
    "raise_25eff",
    "raise_50eff",
    "jam",
    # postflop pot-fraction bets
    "bet_33",
    "bet_50",
    "bet_pot",
    "bet_2x",
    # numeric fallbacks
    "<500",
    "500-999",
    "1k-4.9k",
    "5k-9.9k",
    "10k-19.9k",
    "20k+",
    "other",
)


@dataclass(frozen=True, slots=True)
class DecisionAction:
    key: str
    action_id: int

    def __int__(self):
        return int(self.action_id)


def _action_key(action):
    if isinstance(action, DecisionAction):
        return action.key
    if isinstance(action, str):
        return action
    n = int(action)
    cached = _ACTION_KEY_CACHE.get(n)
    if cached is None:
        cached = sys.intern(str(n))
        _ACTION_KEY_CACHE[n] = cached
    return cached


def _action_keys(actions):
    return [_action_key(action) for action in actions]


def _infoset_from_parsed(parsed, bet_sizing=DEFAULT_BET_SIZING):
    """Abstract infoset key. `bet_sizing` must be the run's: v4 changed the
    postflop betting part of the key."""
    cards = abstractioncards_street_aware(parsed, bet_sizing)
    if not parsed.get("Public"):
        context = preflop_betting_context(parsed)
    else:
        context = postflop_betting_context(parsed, bet_sizing)
    return sys.intern(cards + context)


_LABEL_BUCKETS = {
    "fold": "fold",
    "call": "call/check",
    "check": "call/check",
    "call_check": "call/check",
    "min_raise": "min_raise",
    "open_2_5bb": "open_2_5bb",
    "open_3bb": "open_3bb",
    "raise_2_5bb": "raise_2_5bb",
    "raise_3bb": "raise_3bb",
    "raise_3x": "raise_3x",
    "raise_4x": "raise_4x",
    "raise_25eff": "raise_25eff",
    "raise_50eff": "raise_50eff",
    "jam": "jam",
    # postflop pot-fraction bets
    "bet_33": "bet_33",
    "bet_50": "bet_50",
    "bet_pot": "bet_pot",
    "bet_2x": "bet_2x",
}


def action_bucket(action):
    key = _action_key(action)
    if key in _LABEL_BUCKETS:
        return _LABEL_BUCKETS[key]
    try:
        action = int(action)
    except (ValueError, TypeError):
        return "other"
    if action == 0:
        return "fold"
    if action == 1:
        return "call/check"
    if action < 500:
        return "<500"
    if action < 1_000:
        return "500-999"
    if action < 5_000:
        return "1k-4.9k"
    if action < 10_000:
        return "5k-9.9k"
    if action < 20_000:
        return "10k-19.9k"
    return "20k+"


def _action_sort_key(action):
    key = _action_key(action)
    try:
        return (0, int(key))
    except ValueError:
        return (1, key)


def get_infostate(state, bet_sizing=DEFAULT_BET_SIZING):
    parsed = parse_poker_string(state.information_state_string())
    return _infoset_from_parsed(parsed, bet_sizing)


def preflop_hand_strength(private_cards):
    first, second = private_cards
    first_rank = RANK_VALUE_BY_CARD[first[0]]
    second_rank = RANK_VALUE_BY_CARD[second[0]]
    high = max(first_rank, second_rank)
    low = min(first_rank, second_rank)

    if high == low:
        return min(0.92, 0.50 + high / 14.0 * 0.40)

    suited_bonus = 0.035 if first[1] == second[1] else 0.0
    gap = high - low - 1
    connected_bonus = max(0.0, 0.04 - 0.01 * gap)
    broadway_bonus = 0.03 if low >= 10 else 0.0
    ace_bonus = 0.025 if high == 14 else 0.0
    strength = (
        0.24
        + high / 14.0 * 0.30
        + low / 14.0 * 0.18
        + suited_bonus
        + connected_bonus
        + broadway_bonus
        + ace_bonus
    )
    return max(0.25, min(0.78, strength))


class StrategyTable:
    """
    Sparse regret and strategy table keyed by abstract infoset.

    Each infoset stores action-indexed regrets, current strategy, and an average
    strategy accumulator. Actions are stored as strings so the table is JSON
    serializable without losing OpenSpiel's action ids.
    """

    def __init__(self, data=None):
        self.data = data or {}
        # When recording, every regret and average-strategy change is also
        # added here (same node layout), so parallel workers can share just
        # what they changed since the last sync. See take_delta().
        self.delta = None

    def start_recording(self):
        self.delta = {}

    def take_delta(self):
        """Return the changes recorded since the last call and start afresh."""
        delta, self.delta = self.delta, {}
        return delta

    def _delta_node(self, infoset):
        return self.delta.setdefault(
            infoset,
            {"regret": {}, "strategy": {}, "strategy_sum": {}, "visits": 0},
        )

    def ensure_infoset(self, infoset, legal_actions, action_keys=None):
        if not isinstance(infoset, str):
            infoset = sys.intern(str(infoset))
        if action_keys is None:
            action_keys = _action_keys(legal_actions)
        node = self.data.setdefault(
            infoset,
            {"regret": {}, "strategy": {}, "strategy_sum": {}, "visits": 0},
        )
        regret = node["regret"]
        strategy = node["strategy"]
        strategy_sum = node["strategy_sum"]
        for key in action_keys:
            regret.setdefault(key, 0.0)
            strategy.setdefault(key, 0.0)
            strategy_sum.setdefault(key, 0.0)
        return node

    def regret_matching(self, infoset, legal_actions):
        action_keys = _action_keys(legal_actions)
        node = self.ensure_infoset(infoset, legal_actions, action_keys)
        regret = node["regret"]
        positive_regrets = []
        normalizer = 0.0
        for key in action_keys:
            value = float(regret[key])
            positive = value if value > 0.0 else 0.0
            positive_regrets.append(positive)
            normalizer += positive

        if normalizer > 0:
            probs = [positive / normalizer for positive in positive_regrets]
        else:
            probs = [1.0 / len(legal_actions) for _ in legal_actions]

        strategy = node["strategy"]
        for key, prob in zip(action_keys, probs):
            strategy[key] = prob
        return probs

    def add_regret(self, infoset, action, amount, floor=None):
        key = _action_key(action)
        node = self.ensure_infoset(infoset, [action], [key])
        before = float(node["regret"][key])
        updated = before + amount
        if floor is not None:
            updated = max(floor, updated)
        node["regret"][key] = updated
        if self.delta is not None:
            delta_regret = self._delta_node(infoset)["regret"]
            delta_regret[key] = delta_regret.get(key, 0.0) + (updated - before)

    def batch_add_regret(self, infoset, action_keys, deltas, floor=None):
        node = self.data.get(infoset)
        if node is None:
            return
        regret = node["regret"]
        if self.delta is not None:
            delta_regret = self._delta_node(infoset)["regret"]
            for key, delta in zip(action_keys, deltas):
                before = float(regret[key])
                updated = before + delta
                if floor is not None:
                    updated = max(floor, updated)
                regret[key] = updated
                delta_regret[key] = delta_regret.get(key, 0.0) + (updated - before)
        elif floor is not None:
            for key, delta in zip(action_keys, deltas):
                regret[key] = max(floor, float(regret[key]) + delta)
        else:
            for key, delta in zip(action_keys, deltas):
                regret[key] = float(regret[key]) + delta

    def add_average_strategy(self, infoset, legal_actions, probs, weight):
        action_keys = _action_keys(legal_actions)
        node = self.ensure_infoset(infoset, legal_actions, action_keys)
        node["visits"] += 1
        strategy_sum = node["strategy_sum"]
        for key, prob in zip(action_keys, probs):
            strategy_sum[key] += weight * prob
        if self.delta is not None:
            delta_node = self._delta_node(infoset)
            delta_node["visits"] += 1
            delta_sum = delta_node["strategy_sum"]
            for key, prob in zip(action_keys, probs):
                delta_sum[key] = delta_sum.get(key, 0.0) + weight * prob

    def average_strategy(self):
        average = {}
        for infoset, node in self.data.items():
            total = sum(float(value) for value in node["strategy_sum"].values())
            if total <= 0:
                actions = list(node["strategy"].keys())
                if not actions:
                    average[infoset] = {}
                else:
                    average[infoset] = {
                        action: 1.0 / len(actions) for action in actions
                    }
                continue
            average[infoset] = {
                action: float(value) / total
                for action, value in node["strategy_sum"].items()
            }
        return average

    def current_strategy(self):
        return {
            infoset: {
                action: float(prob)
                for action, prob in node.get("strategy", {}).items()
            }
            for infoset, node in self.data.items()
        }


def merge_strategy_tables(tables):
    """
    Sum sparse table fields from independently trained workers.

    The merge is intentionally additive for regret, strategy_sum, and visits.
    Current strategy is copied as an aggregate weighted by strategy_sum when
    available, otherwise regret-matched on the merged action set.
    """
    merged = StrategyTable()
    for table in tables:
        source = table.data if isinstance(table, StrategyTable) else table
        for infoset, node in source.items():
            target = merged.data.setdefault(
                infoset,
                {"regret": {}, "strategy": {}, "strategy_sum": {}, "visits": 0},
            )
            target["visits"] += int(node.get("visits", 0))
            for field in ("regret", "strategy_sum"):
                for action, value in node.get(field, {}).items():
                    key = _action_key(action)
                    target[field][key] = float(target[field].get(key, 0.0)) + float(
                        value
                    )
                    target["strategy"].setdefault(key, 0.0)
            for action, value in node.get("strategy", {}).items():
                key = _action_key(action)
                target["strategy"].setdefault(key, float(value))

    for infoset, node in merged.data.items():
        actions = sorted(node["strategy"].keys(), key=_action_sort_key)
        total_strategy = sum(float(node["strategy_sum"].get(action, 0.0)) for action in actions)
        if total_strategy > 0:
            for action in actions:
                node["strategy"][action] = (
                    float(node["strategy_sum"].get(action, 0.0)) / total_strategy
                )
        else:
            regrets = [max(0.0, float(node["regret"].get(action, 0.0))) for action in actions]
            normalizer = sum(regrets)
            for action, regret in zip(actions, regrets):
                node["strategy"][action] = (
                    regret / normalizer if normalizer > 0 else 1.0 / len(actions)
                )
    return merged


def add_table_into(target, source, sign=1.0):
    """target += sign * source for regret, strategy_sum and visits, in place.

    target is a StrategyTable or its data dict; source likewise. Actions only
    in source are added to target (current strategy entries start at 0).
    """
    target_data = target.data if isinstance(target, StrategyTable) else target
    source_data = source.data if isinstance(source, StrategyTable) else source
    for infoset, node in source_data.items():
        target_node = target_data.setdefault(
            infoset,
            {"regret": {}, "strategy": {}, "strategy_sum": {}, "visits": 0},
        )
        target_node["visits"] += int(sign * int(node.get("visits", 0)))
        target_strategy = target_node["strategy"]
        for field in ("regret", "strategy_sum"):
            target_field = target_node[field]
            for action, value in node.get(field, {}).items():
                key = _action_key(action)
                target_field[key] = float(target_field.get(key, 0.0)) + sign * float(value)
                target_strategy.setdefault(key, 0.0)


def linear_weight(effective_iteration):
    """Weight LinearMCCFRTrainer gives iteration t: flat for 100, then linear."""
    return max(1, int(effective_iteration) - 99)


def linear_weight_total(iterations, start=0, stride=1):
    """Sum of linear_weight(start + t * stride) for t = 1..iterations.

    A trainer with iteration_stride=stride and start_iteration=start gives
    its t-th iteration that weight.
    """
    n, start, stride = int(iterations), int(start), int(stride)
    if n <= 0:
        return 0
    # linear_weight(g) is 1 until g reaches 100, then g - 99.
    first_linear = max(1, -(-(100 - start) // stride))
    flat = min(n, first_linear - 1)
    if first_linear > n:
        return flat
    count = n - first_linear + 1
    t_sum = (first_linear + n) * count // 2
    return flat + count * (start - 99) + stride * t_sum


def table_metrics(table, weight_total=None):
    """Summary numbers for a table.

    With weight_total (the summed iteration weights behind its regrets),
    avg_regret_bound is sum over infosets of max(0, max_a R(I, a)) divided by
    weight_total. CFR theory bounds the sum of both players' average regrets,
    and so twice the exploitability, by this number in a perfect-recall game.
    Here it is a convergence signal only: regrets are sampled, the card
    abstraction has imperfect recall, and pruning skips updates.
    """
    metrics = {
        "infosets": len(table.data),
        "total_visits": 0,
        "total_strategy_mass": 0.0,
        "positive_regret": 0.0,
        "negative_regret": 0.0,
        "max_positive_regret_sum": 0.0,
    }

    for bucket in ACTION_BUCKETS:
        metrics[f"strategy_mass_{bucket}"] = 0.0
        metrics[f"positive_regret_{bucket}"] = 0.0
        metrics[f"action_entries_{bucket}"] = 0

    for node in table.data.values():
        metrics["total_visits"] += int(node.get("visits", 0))
        strategy_sum = node.get("strategy_sum", {})
        regret = node.get("regret", {})
        if regret:
            metrics["max_positive_regret_sum"] += max(
                0.0, max(float(value) for value in regret.values())
            )
        for action, mass in strategy_sum.items():
            bucket = action_bucket(action)
            strategy_mass = float(mass)
            action_regret = float(regret.get(action, 0.0))
            metrics["total_strategy_mass"] += strategy_mass
            metrics[f"strategy_mass_{bucket}"] += strategy_mass
            metrics[f"action_entries_{bucket}"] += 1

            if action_regret >= 0:
                metrics["positive_regret"] += action_regret
                metrics[f"positive_regret_{bucket}"] += action_regret
            else:
                metrics["negative_regret"] += action_regret

    if weight_total:
        metrics["weight_total"] = weight_total
        metrics["avg_regret_bound"] = metrics["max_positive_regret_sum"] / weight_total
        if metrics["infosets"]:
            metrics["avg_regret_per_infoset"] = (
                metrics["avg_regret_bound"] / metrics["infosets"]
            )
    return metrics


class TrainingCheckpointer:
    def __init__(
        self,
        output_dir,
        checkpoint_iterations=None,
        save_full_table=True,
        metrics_filename="metrics.csv",
    ):
        self.output_dir = output_dir
        self.checkpoint_iterations = set(int(i) for i in checkpoint_iterations or [])
        self.save_full_table = save_full_table
        self.metrics_path = os.path.join(output_dir, metrics_filename)
        os.makedirs(output_dir, exist_ok=True)

    def maybe_save(self, trainer):
        if trainer.iteration not in self.checkpoint_iterations:
            return

        metrics = table_metrics(
            trainer.table,
            weight_total=linear_weight_total(trainer.start_iteration)
            + linear_weight_total(
                trainer.iteration, trainer.start_iteration, trainer.iteration_stride
            ),
        )
        metrics["iteration"] = trainer.iteration
        self._append_metrics(metrics)

        if self.save_full_table:
            path = os.path.join(
                self.output_dir,
                f"mccfr_table_iter_{trainer.iteration:08d}.json.gz",
            )
            save_table(trainer.table, path)

    def _append_metrics(self, metrics):
        fieldnames = ["iteration"] + [
            key for key in metrics.keys() if key != "iteration"
        ]
        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)


def _raise_range_from_seq(seq_str, big_blind, max_raise, no_bet_baseline=0):
    """Parse a single-street sequence string → (min_raise, max_raise, last_top).

    seq_str should contain only the current street (no '|' separators).
    max_raise is passed in (starting_stack for preflop, money[player] for postflop).
    no_bet_baseline: the virtual 'previous raise' when no raises yet appear in seq
    (big_blind for preflop because of the forced BB; 0 for postflop).
    """
    raise_amounts = []
    amt = ""
    for ch in seq_str:
        if ch == "r":
            amt = ""
        elif ch.isdigit():
            amt += ch
        elif amt:
            raise_amounts.append(int(amt))
            amt = ""
    if amt:
        raise_amounts.append(int(amt))

    if not raise_amounts:
        last_top, prev_top = no_bet_baseline, 0
    elif len(raise_amounts) == 1:
        last_top, prev_top = raise_amounts[0], no_bet_baseline
    else:
        last_top, prev_top = raise_amounts[-1], raise_amounts[-2]

    min_raise = last_top + (last_top - prev_top)
    min_raise = max(big_blind, min(min_raise, max_raise))
    return min_raise, max_raise, last_top


@lru_cache(maxsize=65536)
def _raise_amounts(seq_str):
    """Chip amounts of the raises in a sequence string, in order (a tuple)."""
    amounts = []
    amt = ""
    for ch in seq_str:
        if ch.isdigit():
            amt += ch
            continue
        if amt:
            amounts.append(int(amt))
            amt = ""
    if amt:
        amounts.append(int(amt))
    return tuple(amounts)


def _street_raise_range(parsed, big_blind, starting_stack):
    """Return (min_raise, max_raise, last_top) for the current street.

    Action ids are cumulative chips committed across the whole hand. Each street
    starts with both players committed to the last raise of the previous
    streets (preflop and unraised streets: the big blind). The minimum raise
    increment is the last increment on this street, and at least one big blind.
    """
    streets = (parsed.get("Sequences", "") or "").split("|")
    earlier = _raise_amounts("|".join(streets[:-1]))
    street_start = earlier[-1] if earlier else big_blind
    tops = (street_start,) + _raise_amounts(streets[-1])
    last_top = tops[-1]
    increment = max(big_blind, last_top - tops[-2]) if len(tops) > 1 else big_blind
    return min(last_top + increment, starting_stack), starting_stack, last_top


def _preflop_raise_range(parsed, big_blind, starting_stack=0):
    """Return (min_raise, max_raise, last_top) for the preflop street.

    In universal_poker fullgame, action ID N = total chips committed this street
    by the current player. max_raise = starting_stack (all-in = full initial stack,
    not remaining chips). min_raise uses the standard NLHE rule:
    last_top + (last_top - prev_top), with big_blind as the forced-blind baseline.
    """
    money = parsed.get("Money") or []
    player = int(parsed.get("Player", 0) or 0)
    sequence = parsed.get("Sequences", "") or ""
    max_raise = starting_stack if starting_stack > 0 else (
        int(money[player]) if len(money) > player else 0
    )
    return _raise_range_from_seq(sequence, big_blind, max_raise, no_bet_baseline=big_blind)




class ActionAbstractor:
    """
    Selects an interpretable subset of no-limit actions.

    Buckets are anchored to pot geometry and effective stack commitment, then
    translated to the nearest legal OpenSpiel action.
    """

    def __init__(
        self,
        pot_fractions=DEFAULT_POT_FRACTIONS,
        stack_fractions=DEFAULT_STACK_FRACTIONS,
        random_probe_count=0,
        max_actions=None,
        rng=None,
    ):
        self.pot_fractions = tuple(pot_fractions)
        self.stack_fractions = tuple(stack_fractions)
        self.random_probe_count = random_probe_count
        self.max_actions = max_actions
        self.rng = rng

    def select_actions(self, state, legal_actions, parsed=None):
        legal_actions = self._sorted_int_actions(legal_actions)
        if not legal_actions:
            return []

        parsed = parsed if parsed is not None else self._parse_state(state)
        raise_start = bisect_right(legal_actions, 1)
        base_actions = [
            action for action in legal_actions[:raise_start] if action in (0, 1)
        ]
        base_actions.append(legal_actions[-1])

        selected = list(base_actions)
        if raise_start < len(legal_actions):
            selected.append(legal_actions[raise_start])
            targets = self._target_amounts(parsed)
            for target in targets:
                selected.append(
                    self._nearest_action(target, legal_actions, start=raise_start)
                )
            if self.random_probe_count > 0:
                selected.extend(
                    self._random_probes(legal_actions[raise_start:], selected)
                )

        return self._dedupe_and_cap(selected)

    def describe_actions(self, state, legal_actions, selected_actions):
        parsed = self._parse_state(state)
        player = self._current_player(state, parsed)
        return {
            "pot": parsed.get("Pot"),
            "to_call": self._to_call(parsed, player),
            "active_stack": self._active_stack(parsed, player),
            "opponent_stack": self._opponent_stack(parsed, player),
            "effective_stack": self._effective_stack(parsed),
            "actions": [
                {
                    "action": int(action),
                    "label": self._action_label(state, action),
                }
                for action in selected_actions
            ],
        }

    def _parse_state(self, state):
        try:
            return parse_poker_string(state.information_state_string())
        except Exception:
            return {}

    def _base_actions(self, state, legal_actions):
        base = [action for action in legal_actions if int(action) in (0, 1)]
        if legal_actions:
            base.append(legal_actions[-1])
        return base

    def _raise_actions(self, state, legal_actions):
        return [action for action in legal_actions if int(action) > 1]

    def _split_base_and_raise_actions(self, legal_actions):
        base_actions = []
        raise_actions = []
        for action in legal_actions:
            if action in (0, 1):
                base_actions.append(action)
            elif action > 1:
                raise_actions.append(action)
        if legal_actions:
            base_actions.append(legal_actions[-1])
        return base_actions, raise_actions

    def _sorted_int_actions(self, legal_actions):
        if not legal_actions:
            return []
        if len(legal_actions) > 64 and isinstance(legal_actions[0], int):
            return legal_actions
        iterator = iter(legal_actions)
        previous = next(iterator)
        if not isinstance(previous, int):
            return sorted(int(action) for action in legal_actions)
        for action in iterator:
            if not isinstance(action, int):
                return sorted(int(item) for item in legal_actions)
            if action < previous:
                return sorted(legal_actions)
            previous = action
        return legal_actions

    def _target_amounts(self, parsed):
        pot = max(0, int(parsed.get("Pot", 0) or 0))
        player = self._current_player(None, parsed)
        to_call = self._to_call(parsed, player)
        pot_after_call = pot + to_call
        effective_stack = self._effective_stack(parsed)

        targets = []
        targets.extend(pot_after_call * fraction for fraction in self.pot_fractions)
        targets.extend(effective_stack * fraction for fraction in self.stack_fractions)
        if effective_stack:
            targets.append(effective_stack)
        return [target for target in targets if target > 0]

    def _current_player(self, state, parsed):
        if state is not None:
            try:
                return int(state.current_player())
            except Exception:
                pass
        try:
            return int(parsed.get("Player", 0))
        except Exception:
            return 0

    def _to_call(self, parsed, player):
        return max(
            0,
            self._active_stack(parsed, player) - self._opponent_stack(parsed, player),
        )

    def _active_stack(self, parsed, player):
        money = parsed.get("Money") or []
        if player < len(money):
            return int(money[player])
        return 0

    def _opponent_stack(self, parsed, player):
        money = parsed.get("Money") or []
        opponent = 1 - player
        if opponent < len(money):
            return int(money[opponent])
        return 0

    def _effective_stack(self, parsed):
        money = parsed.get("Money") or []
        if len(money) >= 2:
            return max(0, min(int(money[0]), int(money[1])))
        return 0

    def _nearest_action(self, target, legal_actions, start=0):
        index = bisect_left(legal_actions, target, lo=start)
        if index <= start:
            return legal_actions[start]
        if index >= len(legal_actions):
            return legal_actions[-1]
        lower = legal_actions[index - 1]
        upper = legal_actions[index]
        if abs(target - lower) <= abs(upper - target):
            return lower
        return upper

    def _random_probes(self, legal_actions, selected):
        if self.random_probe_count <= 0:
            return []
        remaining = [action for action in legal_actions if action not in selected]
        if not remaining:
            return []
        count = min(int(self.random_probe_count), len(remaining))
        rng = self.rng or np.random.default_rng()
        return [int(action) for action in rng.choice(remaining, size=count, replace=False)]

    def _dedupe_and_cap(self, actions):
        deduped = []
        seen = set()
        for action in actions:
            action = int(action)
            if action not in seen:
                deduped.append(action)
                seen.add(action)
        if self.max_actions is not None:
            return deduped[: int(self.max_actions)]
        return deduped

    def _action_label(self, state, action):
        try:
            return state.action_to_string(state.current_player(), int(action))
        except Exception:
            return str(int(action))


class PreflopActionAbstractor(ActionAbstractor):
    """
    Lossless hand abstraction with a deliberately small preflop betting menu.

    Targets are expressed in chips using the configured big blind. Each target
    is mapped to the nearest legal OpenSpiel action, so the trainer never emits
    an illegal action even when the full no-limit action space is large.
    """

    def __init__(
        self,
        big_blind=100,
        rng=None,
        max_actions=None,
        starting_stack=0,
        bet_sizing=DEFAULT_BET_SIZING,
    ):
        super().__init__(
            pot_fractions=(),
            stack_fractions=(),
            random_probe_count=0,
            max_actions=max_actions,
            rng=rng,
        )
        if bet_sizing not in BET_SIZINGS:
            raise ValueError(f"bet_sizing must be one of {BET_SIZINGS}, got {bet_sizing!r}")
        self.big_blind = int(big_blind)
        self.starting_stack = int(starting_stack)
        self.bet_sizing = bet_sizing
        # Preflop menus depend only on the betting so far (never the cards) and
        # repeat constantly, so they are cached. Postflop, pot-relative sizes
        # make nearly every betting state unique, so caching there only costs
        # memory.
        self._preflop_spec_cache = {}

    def select_actions(self, state, legal_actions, parsed=None):
        return [
            int(action)
            for action in self.select_action_specs(state, legal_actions, parsed=parsed)
        ]

    def select_action_specs(self, state, legal_actions, parsed=None):
        parsed = parsed if parsed is not None else self._parse_state(state)
        if int(parsed.get("Round", 0) or 0) != 0:
            return [
                DecisionAction(_action_key(action), int(action))
                for action in super().select_actions(state, legal_actions, parsed=parsed)
            ]

        legal_actions = self._sorted_int_actions(legal_actions)
        if not legal_actions:
            return []

        selected = []
        used_action_ids = set()
        raise_start = bisect_right(legal_actions, 1)
        for action in legal_actions[:raise_start]:
            label = self._preflop_base_label(state, action)
            self._append_spec(selected, used_action_ids, label, action)

        if raise_start < len(legal_actions):
            self._append_spec(
                selected, used_action_ids, "min_raise", legal_actions[raise_start]
            )
            for label, target in self._preflop_targets(parsed):
                self._append_spec(
                    selected,
                    used_action_ids,
                    label,
                    self._nearest_action(target, legal_actions, start=raise_start),
                )
            self._append_spec(selected, used_action_ids, "jam", legal_actions[-1])

        if self.max_actions is not None:
            return selected[: int(self.max_actions)]
        return selected

    def _append_spec(self, selected, used_action_ids, label, action):
        action = int(action)
        if action in used_action_ids:
            return
        selected.append(DecisionAction(label, action))
        used_action_ids.add(action)

    def _preflop_base_label(self, state, action):
        action = int(action)
        if action == 0:
            return "fold"
        if action == 1:
            return "call"
        label = self._action_label(state, action).lower()
        if "check" in label:
            return "check"
        return _action_key(action)

    def _preflop_base_actions(self, state, legal_actions):
        return [action for action in legal_actions if int(action) in (0, 1)]

    def _preflop_raise_actions(self, legal_actions):
        return [action for action in legal_actions if int(action) > 1]

    def _preflop_targets(self, parsed):
        effective_stack = self._effective_stack(parsed)
        previous_raise = self._previous_raise_amount(parsed)
        sequence = parsed.get("Sequences", "") or ""
        prefix = "raise" if previous_raise > 0 or sequence else "open"

        targets = [
            (f"{prefix}_2_5bb", 2.5 * self.big_blind),
            (f"{prefix}_3bb", 3.0 * self.big_blind),
            ("raise_25eff", 0.25 * effective_stack),
            ("raise_50eff", 0.50 * effective_stack),
        ]
        if previous_raise > 0:
            targets.extend(
                (
                    ("raise_3x", 3.0 * previous_raise),
                    ("raise_4x", 4.0 * previous_raise),
                )
            )
        return [(label, target) for label, target in targets if target > 0]

    def _previous_raise_amount(self, parsed):
        sequence = parsed.get("Sequences", "") or ""
        last_raise = 0
        amount = ""
        for char in sequence:
            if char == "r":
                amount = ""
            elif char.isdigit():
                amount += char
            elif amount:
                last_raise = int(amount)
                amount = ""
        if amount:
            last_raise = int(amount)
        return last_raise

    def select_action_specs_direct(self, parsed):
        """Compute action specs without calling state.legal_actions().

        Dispatches to preflop or postflop helper based on the Round field.
        Raise amounts are concrete chip totals clamped to [min_raise, max_raise],
        derived from the parsed info string alone. Returns a tuple, which
        preflop is cached and shared between callers.
        """
        if int(parsed.get("Round", 0) or 0) != 0:
            return tuple(self._postflop_specs_direct(parsed))
        key = (
            parsed.get("Player", 0),
            tuple(parsed.get("Money") or ()),
            parsed.get("Sequences", ""),
        )
        specs = self._preflop_spec_cache.get(key)
        if specs is None:
            specs = tuple(self._preflop_specs_direct(parsed))
            self._preflop_spec_cache[key] = specs
        return specs

    def _preflop_specs_direct(self, parsed):
        money = parsed.get("Money") or []
        player = int(parsed.get("Player", 0) or 0)
        to_call = (
            max(0, int(money[player]) - int(money[1 - player]))
            if len(money) >= 2
            else 0
        )
        # Legacy keeps the old parser, which drops all but the last of
        # back-to-back raises, so pre-fix tables still line up.
        raise_range = (
            _preflop_raise_range if self.bet_sizing == "legacy" else _street_raise_range
        )
        min_raise, max_raise, last_top = raise_range(
            parsed, self.big_blind, self.starting_stack
        )
        has_raises = max_raise > last_top
        capped = (
            self.bet_sizing in RAISE_CAP_SIZINGS
            and len(_raise_amounts(parsed.get("Sequences", "") or "")) >= PREFLOP_RAISE_CAP
        )

        specs = []
        seen = set()
        if to_call > 0:
            specs.append(DecisionAction("fold", 0))
            seen.add(0)
        specs.append(DecisionAction("call", 1))
        seen.add(1)

        if has_raises and capped:
            specs.append(DecisionAction("jam", max_raise))
        elif has_raises:
            specs.append(DecisionAction("min_raise", min_raise))
            seen.add(min_raise)
            for label, target in self._preflop_targets(parsed):
                amount = max(min_raise, min(max_raise, int(round(target))))
                if amount not in seen:
                    specs.append(DecisionAction(label, amount))
                    seen.add(amount)
            if max_raise not in seen:
                specs.append(DecisionAction("jam", max_raise))

        if self.max_actions is not None:
            return specs[: self.max_actions]
        return specs

    def _postflop_specs_direct(self, parsed):
        if self.bet_sizing == "legacy":
            return self._postflop_specs_direct_legacy(parsed)

        money = parsed.get("Money") or []
        player = int(parsed.get("Player", 0) or 0)
        to_call = (
            max(0, int(money[player]) - int(money[1 - player]))
            if len(money) >= 2
            else 0
        )
        min_raise, max_raise, last_top = _street_raise_range(
            parsed, self.big_blind, self.starting_stack
        )

        specs = []
        seen = set()
        if to_call > 0:
            specs.append(DecisionAction("fold", 0))
            seen.add(0)
        specs.append(DecisionAction("call", 1))
        seen.add(1)

        v3 = self.bet_sizing in RAISE_CAP_SIZINGS
        raises_this_street = len(
            _raise_amounts((parsed.get("Sequences", "") or "").rsplit("|", 1)[-1])
        )
        if max_raise > last_top and v3 and raises_this_street >= POSTFLOP_RAISE_CAP:
            specs.append(DecisionAction("jam", max_raise))
        elif max_raise > last_top:
            fractions = POSTFLOP_POT_FRACTIONS
            if v3 and raises_this_street > 0:
                fractions = POSTFLOP_FACING_BET_FRACTIONS
            else:
                specs.append(DecisionAction("min_raise", min_raise))
                seen.add(min_raise)
            # Sizes are fractions of the pot after calling, added on top of the
            # amount needed to call (last_top is also each player's total then).
            pot_after_call = 2 * last_top
            for label, fraction in fractions:
                amount = last_top + int(round(fraction * pot_after_call))
                amount = max(min_raise, min(max_raise, amount))
                if amount not in seen:
                    specs.append(DecisionAction(label, amount))
                    seen.add(amount)
            if max_raise not in seen:
                specs.append(DecisionAction("jam", max_raise))

        if self.max_actions is not None:
            return specs[: self.max_actions]
        return specs

    def _postflop_specs_direct_legacy(self, parsed):
        """Pre-fix sizing, kept so checkpoints trained with it can still be read.

        Its pot targets omit the chips already committed, so bet_pot is really
        a half-pot bet, bet_2x is 1.5x pot, and bet_33/bet_50 collapse into
        min_raise. Its min_raise carries the previous street's raise increment.
        """
        money = parsed.get("Money") or []
        player = int(parsed.get("Player", 0) or 0)
        to_call = (
            max(0, int(money[player]) - int(money[1 - player]))
            if len(money) >= 2
            else 0
        )
        min_raise, max_raise, last_top = _preflop_raise_range(
            parsed, self.big_blind, self.starting_stack
        )
        has_raises = max_raise > last_top

        pot = int(parsed.get("Pot", 0) or 0)

        specs = []
        seen = set()
        if to_call > 0:
            specs.append(DecisionAction("fold", 0))
            seen.add(0)
        specs.append(DecisionAction("call", 1))
        seen.add(1)

        if has_raises:
            specs.append(DecisionAction("min_raise", min_raise))
            seen.add(min_raise)

            targets = []
            if pot > 0:
                targets = [
                    ("bet_33", int(round(pot * 0.33))),
                    ("bet_50", int(round(pot * 0.50))),
                    ("bet_pot", pot),
                    ("bet_2x", pot * 2),
                ]
            for label, target in targets:
                amount = max(min_raise, min(max_raise, target))
                if amount not in seen:
                    specs.append(DecisionAction(label, amount))
                    seen.add(amount)

            if max_raise not in seen:
                specs.append(DecisionAction("jam", max_raise))

        if self.max_actions is not None:
            return specs[: self.max_actions]
        return specs


class LinearMCCFRTrainer:
    """
    External-sampling MCCFR with linear update weights and negative-regret pruning.

    This is still much smaller than Pluribus: it has no real-time subgame search,
    continuation values, belief updates, or six-player-specific optimizations. It
    does provide the core algorithmic pieces missing from the original prototype.
    """

    def __init__(
        self,
        game,
        table=None,
        infoset_fn=get_infostate,
        prune_threshold=DEFAULT_PRUNE_THRESHOLD,
        prune_probability=0.95,
        prune_after=200,
        regret_floor=DEFAULT_PRUNE_THRESHOLD - 10_000_000,
        max_traverser_actions=None,
        action_abstractor=None,
        cutoff_street=None,
        start_iteration=0,
        iteration_stride=1,
        rng=None,
    ):
        self.game = game
        self.table = table or StrategyTable()
        self.infoset_fn = infoset_fn
        self.prune_threshold = prune_threshold
        self.prune_probability = prune_probability
        self.prune_after = prune_after
        self.regret_floor = regret_floor
        self.max_traverser_actions = max_traverser_actions
        self.rng = rng or np.random.default_rng()
        if action_abstractor is None and max_traverser_actions is not None:
            action_abstractor = ActionAbstractor(
                random_probe_count=1,
                max_actions=max_traverser_actions,
                rng=self.rng,
            )
        elif action_abstractor is not None and action_abstractor.rng is None:
            action_abstractor.rng = self.rng
        self.action_abstractor = action_abstractor
        self._bet_sizing = getattr(action_abstractor, "bet_sizing", DEFAULT_BET_SIZING)
        self.cutoff_street = cutoff_street
        # Linear weights follow overall training progress: iteration t of this
        # trainer counts as start_iteration + t * iteration_stride. Parallel
        # workers set the stride to the worker count.
        self.start_iteration = int(start_iteration)
        self.iteration_stride = int(iteration_stride)
        self.iteration = 0
        self._chance_priority = None

    def train(
        self,
        iterations,
        save_every=None,
        save_path=None,
        checkpointer=None,
    ):
        for _ in range(iterations):
            self.iteration += 1
            # Flat weight for first 100 effective iterations, then linear.
            # start_iteration shifts the ramp so resumed runs continue smoothly.
            effective = self.start_iteration + self.iteration * self.iteration_stride
            weight = linear_weight(effective)
            for player in range(self.game.num_players()):
                state = self.game.new_initial_state()
                max_outcomes = getattr(self.game, "max_chance_outcomes", None)
                self._chance_priority = (
                    self.rng.random(max_outcomes()) if max_outcomes else None
                )
                self.mccfr(
                    state,
                    traverser=player,
                    linear_weight=weight,
                    allow_pruning=self._should_prune_iteration(),
                )

            if save_every and save_path and self.iteration % save_every == 0:
                save_table(self.table, save_path)
                print(self.iteration)
            if checkpointer:
                checkpointer.maybe_save(self)

        return self.table

    def mccfr(
        self,
        state,
        traverser,
        linear_weight=1.0,
        allow_pruning=False,
        parsed=None,
    ):
        if state.is_terminal():
            return float(state.returns()[traverser])

        if state.is_chance_node():
            state.apply_action(self._sample_chance(state))
            return self.mccfr(state, traverser, linear_weight, allow_pruning)

        parsed = self._parsed_state(state) if parsed is None else parsed

        if self._should_cutoff(state, parsed):
            return self._cutoff_value(state, traverser, parsed=parsed)

        current_player = state.current_player()
        infoset = self._infoset(state, parsed)
        if (
            self.action_abstractor is not None
            and hasattr(self.action_abstractor, "select_action_specs_direct")
        ):
            legal_actions = self.action_abstractor.select_action_specs_direct(parsed)
        else:
            legal_actions = self._decision_actions(
                state, list(state.legal_actions()), parsed=parsed
            )
        strategy = self.table.regret_matching(infoset, legal_actions)

        if current_player != traverser:
            self.table.add_average_strategy(
                infoset, legal_actions, strategy, linear_weight
            )
            action = self._sample_action(legal_actions, strategy)
            state.apply_action(int(action))
            return self.mccfr(state, traverser, linear_weight, allow_pruning)

        traverser_actions = self._traverser_actions(
            state, infoset, legal_actions, allow_pruning, parsed=parsed
        )
        if not traverser_actions:
            traverser_actions = legal_actions

        traverser_set = set(traverser_actions)
        action_values = [None] * len(legal_actions)
        node_value = 0.0
        selected_strategy_mass = 0.0
        for action, probability in zip(legal_actions, strategy):
            if action in traverser_set:
                selected_strategy_mass += probability

        traverser_count = len(traverser_actions)
        for index, action in enumerate(legal_actions):
            if action not in traverser_set:
                continue
            new_state = state.clone()
            new_state.apply_action(int(action))
            action_value = self.mccfr(
                new_state, traverser, linear_weight, allow_pruning
            )
            action_values[index] = action_value
            if selected_strategy_mass > 0:
                action_prob = strategy[index] / selected_strategy_mass
            else:
                action_prob = 1.0 / traverser_count
            node_value += action_prob * action_value

        # The average strategy is only accumulated at the other player's nodes,
        # where sampling visits an infoset in proportion to that player's own
        # reach probability. The traverser explores every action, so adding
        # here would weight its infosets by the wrong reach.

        action_keys = _action_keys(legal_actions)
        deltas = []
        batch_keys = []
        for index, action in enumerate(legal_actions):
            action_value = action_values[index]
            if action_value is None:
                continue
            batch_keys.append(action_keys[index])
            deltas.append(linear_weight * (action_value - node_value))
        self.table.batch_add_regret(infoset, batch_keys, deltas, floor=self.regret_floor)

        return node_value

    def _sample_chance(self, state):
        """Sample a chance outcome, dealing cards once per traversal.

        Card deals are uniform, so each traversal draws one random priority
        per card and every chance node takes its legal card with the lowest
        priority. All branches of the traversal then see the same cards, as if
        the deck were shuffled once at the start, which is how poker MCCFR
        usually samples chance. Values stay unbiased, and card-dependent work
        (like v5 card buckets) happens once per street, not once per branch.
        Non-uniform chance events are sampled directly.
        """
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        if self._chance_priority is not None and max(probs) - min(probs) < 1e-12:
            priority = self._chance_priority
            return int(min(actions, key=lambda action: priority[action]))
        return int(self.rng.choice(actions, p=probs))

    def _should_prune_iteration(self):
        return (
            self.iteration >= self.prune_after
            and self.rng.random() < self.prune_probability
        )

    def _traverser_actions(self, state, infoset, legal_actions, allow_pruning, parsed=None):
        if not allow_pruning or self._is_final_betting_round(state, parsed=parsed):
            return legal_actions

        node = self.table.ensure_infoset(infoset, legal_actions)
        kept = []
        for action in legal_actions:
            if self._action_reaches_terminal(state, action):
                kept.append(action)
                continue
            regret = float(node["regret"][_action_key(action)])
            if regret >= self.prune_threshold:
                kept.append(action)
        return kept

    def _decision_actions(self, state, actions, parsed=None):
        if self.action_abstractor is None:
            return actions
        if hasattr(self.action_abstractor, "select_action_specs"):
            try:
                selected = self.action_abstractor.select_action_specs(
                    state, actions, parsed=parsed
                )
            except TypeError:
                selected = self.action_abstractor.select_action_specs(state, actions)
        else:
            try:
                selected = self.action_abstractor.select_actions(
                    state, actions, parsed=parsed
                )
            except TypeError:
                selected = self.action_abstractor.select_actions(state, actions)
        selected = selected or actions
        return selected

    def _sample_action(self, actions, probs):
        threshold = float(self.rng.random())
        cumulative = 0.0
        for action, prob in zip(actions, probs):
            cumulative += float(prob)
            if cumulative >= threshold:
                return action
        return actions[-1]

    def _should_cutoff(self, state, parsed=None):
        if self.cutoff_street != "preflop":
            return False
        try:
            parsed = self._parsed_state(state) if parsed is None else parsed
            return bool(parsed.get("Public")) or int(parsed.get("Round", 0) or 0) > 0
        except Exception:
            return False

    def _cutoff_value(self, state, traverser, parsed=None):
        try:
            if parsed is None:
                parsed = self._parse_player_state(state, traverser)
            money = parsed.get("Money") or []
            pot = float(parsed.get("Pot", 0) or 0)
            private_cards = parsed.get("Private") or []
            if traverser >= len(money) or len(private_cards) != 2:
                return 0.0
            starting_stack = (sum(float(stack) for stack in money) + pot) / len(money)
            contribution = max(0.0, starting_stack - float(money[traverser]))
            equity = preflop_hand_strength(private_cards)
            return equity * pot - contribution
        except Exception:
            return 0.0

    def _parse_player_state(self, state, player):
        try:
            return parse_poker_string(state.information_state_string(int(player)))
        except TypeError:
            return parse_poker_string(state.information_state_string())

    def _parsed_state(self, state):
        try:
            return parse_poker_string(state.information_state_string())
        except Exception:
            return {}

    def _infoset(self, state, parsed):
        if self.infoset_fn is get_infostate:
            try:
                return _infoset_from_parsed(parsed, self._bet_sizing)
            except Exception:
                pass
        return self.infoset_fn(state)

    def _is_final_betting_round(self, state, parsed=None):
        try:
            if parsed is None:
                parsed = parse_poker_string(state.information_state_string())
            return int(parsed.get("Round", 0)) >= int(game_config["numRounds"]) - 1
        except Exception:
            return False

    def _action_reaches_terminal(self, state, action):
        new_state = state.clone()
        new_state.apply_action(int(action))
        return new_state.is_terminal()


def save_table(table, path=None):
    path = path or _default_path("mccfr_table.json")
    if str(path).endswith(".gz"):
        open_fn = lambda filename, mode: gzip.open(filename, mode, compresslevel=1)
    else:
        open_fn = open
    # Write to a temporary file and rename it into place, so a crash or kill
    # mid-save never leaves a truncated checkpoint behind.
    tmp_path = f"{path}.tmp"
    with open_fn(tmp_path, "wt") as out_file:
        json.dump(table.data, out_file)
    os.replace(tmp_path, path)


def _default_path(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def run_manifest(path):
    """A checkpoint's run_manifest.json (or {} if it has none)."""
    folder = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else path
    manifest_path = os.path.join(folder, "run_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as manifest_file:
        return json.load(manifest_file)


def game_config_for_checkpoint(path):
    """Game rules a checkpoint was trained under, from its run_manifest.json.

    Without a recorded config, the run predates the seat-order fix.
    """
    config = run_manifest(path).get("game_config")
    if config:
        return dict(config)
    return dict(game_config, firstPlayer=LEGACY_FIRST_PLAYER)


def bet_sizing_for_checkpoint(path):
    """Bet sizing a checkpoint was trained with, from its run_manifest.json.

    Runs from before the sizing fix have no "bet_sizing" entry (or no manifest)
    and used the legacy sizing.
    """
    return run_manifest(path).get("bet_sizing", "legacy")


def load_table(path=None):
    path = path or _default_path("mccfr_table.json")
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as in_file:
        return StrategyTable(json.load(in_file))


