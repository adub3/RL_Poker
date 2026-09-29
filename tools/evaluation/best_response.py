"""
Greedy best-response estimate of a fixed strategy's exploitability.

Works on any two-player zero-sum OpenSpiel game. The caller supplies:
  - describe(state) -> (infoset_key, actions): how the tested strategy sees a
    decision (its card and betting abstraction). actions are ints or objects
    with int() and an action key (see ai._action_key).
  - blueprint.get(infoset_key) -> {action_key: prob} or None. Missing or
    zero-mass entries are played uniformly at random, like the live bot.

For each seat, the best responder plays against the frozen blueprint:
  1. Training: sampled traversals estimate the value of every action at each
     responder infoset (all responder actions are expanded, chance and
     blueprint actions are sampled). Each node returns the value of the
     action the current estimates rank best, chosen before this visit's
     update so the value passed up is not biased by picking a lucky sample.
  2. Evaluation: the responder always plays its best-ranked action, on fresh
     deals, so the result is an unbiased estimate of what that fixed
     strategy wins. Infosets the responder never trained on are played
     uniformly at random and counted.

Because the responder can only be as good as a true best response, the
estimate is a lower bound on exploitability within the abstraction described
by `describe`, up to sampling error (reported as a standard error).
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_code"))

from ai import _action_key  # noqa: E402


class GreedyBestResponse:
    def __init__(self, describe, blueprint, seed=0):
        self.describe = describe
        self.blueprint = blueprint
        self.rng = np.random.default_rng(seed)

    def compute(self, game, train_iterations, eval_hands):
        """Estimate exploitability; values are in the game's payoff units."""
        seats = [self.best_respond(game, seat, train_iterations, eval_hands) for seat in (0, 1)]
        value = (seats[0]["value"] + seats[1]["value"]) / 2
        stderr = math.sqrt(seats[0]["stderr"] ** 2 + seats[1]["stderr"] ** 2) / 2
        return {
            "exploitability": value,
            "stderr": stderr,
            "ci95": [value - 1.96 * stderr, value + 1.96 * stderr],
            "seats": seats,
        }

    def best_respond(self, game, seat, train_iterations, eval_hands):
        q_table = {}
        for _ in range(train_iterations):
            self._traverse(game.new_initial_state(), seat, q_table)

        counts = {"blueprint_decisions": 0, "blueprint_missing": 0,
                  "responder_decisions": 0, "responder_untrained": 0}
        returns = np.array([
            self._play(game.new_initial_state(), seat, q_table, counts)
            for _ in range(eval_hands)
        ])
        return {
            "seat": seat,
            "value": float(returns.mean()),
            "stderr": float(returns.std(ddof=1) / math.sqrt(len(returns))),
            "trained_infosets": len(q_table),
            "blueprint_missing_rate": _rate(counts["blueprint_missing"], counts["blueprint_decisions"]),
            "responder_untrained_rate": _rate(counts["responder_untrained"], counts["responder_decisions"]),
            **counts,
        }

    # ------------------------------------------------------------------

    def _traverse(self, state, seat, q_table):
        while True:
            if state.is_terminal():
                return float(state.returns()[seat])
            if state.is_chance_node():
                self._apply_chance(state)
                continue
            infoset, actions = self.describe(state)
            if state.current_player() == seat:
                break
            state.apply_action(self._blueprint_action(infoset, actions)[0])

        node = q_table.setdefault(infoset, {})
        choice = self._greedy_index(node, actions)
        values = []
        for action in actions:
            child = state.clone()
            child.apply_action(int(action))
            values.append(self._traverse(child, seat, q_table))
        for action, value in zip(actions, values):
            entry = node.setdefault(_action_key(action), [0.0, 0])
            entry[0] += value
            entry[1] += 1
        return values[choice]

    def _play(self, state, seat, q_table, counts):
        while not state.is_terminal():
            if state.is_chance_node():
                self._apply_chance(state)
                continue
            infoset, actions = self.describe(state)
            if state.current_player() == seat:
                counts["responder_decisions"] += 1
                node = q_table.get(infoset)
                if not node:
                    counts["responder_untrained"] += 1
                state.apply_action(int(actions[self._greedy_index(node or {}, actions)]))
            else:
                counts["blueprint_decisions"] += 1
                action, missing = self._blueprint_action(infoset, actions)
                counts["blueprint_missing"] += missing
                state.apply_action(action)
        return float(state.returns()[seat])

    def _greedy_index(self, node, actions):
        best_index, best_value = None, -math.inf
        for index, action in enumerate(actions):
            entry = node.get(_action_key(action))
            if entry and entry[1] and entry[0] / entry[1] > best_value:
                best_index, best_value = index, entry[0] / entry[1]
        if best_index is None:
            return int(self.rng.integers(len(actions)))
        return best_index

    def _blueprint_action(self, infoset, actions):
        """Return (action id, 1 if the blueprint had no usable entry else 0)."""
        policy = self.blueprint.get(infoset)
        if policy:
            weights = np.array([float(policy.get(_action_key(a), 0.0)) for a in actions])
            total = weights.sum()
            if total > 0:
                return int(actions[self.rng.choice(len(actions), p=weights / total)]), 0
        return int(actions[int(self.rng.integers(len(actions)))]), 1

    def _apply_chance(self, state):
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(outcomes[self.rng.choice(len(outcomes), p=probs)]))


def _rate(part, whole):
    return part / whole if whole else 0.0
