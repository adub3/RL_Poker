"""
Real-Time Search (Pluribus-style) for heads-up no-limit Texas Hold'em.

At each decision point the bot:
  1. Runs a short external-sampling MCCFR on the current betting round,
     using a fresh local regret table.
  2. At the boundary (chance node = new street card about to be dealt),
     estimates leaf value by rolling out with the blueprint average strategy.
  3. Samples an action from the resulting local average strategy.

The blueprint dict ({infoset: {action_key: prob}}) is used read-only.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "new_code"))

from abstraction import parse_poker_string  # noqa: E402
from ai import (  # noqa: E402
    PreflopActionAbstractor,
    StrategyTable,
    _action_key,
    _action_keys,
    _infoset_from_parsed,
)

# Must match the values used when training the blueprint.
_BIG_BLIND = 100
_STARTING_STACK = 10_000


class RealTimeSearch:
    """Pluribus-style real-time subgame search for a single decision."""

    def __init__(
        self,
        blueprint: dict,
        iterations: int = 100,
        rollout_samples: int = 5,
        rng=None,
    ):
        self.blueprint = blueprint
        self.iterations = iterations
        self.rollout_samples = rollout_samples
        self._rng = rng if rng is not None else np.random.default_rng()
        self._abstractor = PreflopActionAbstractor(
            big_blind=_BIG_BLIND,
            starting_stack=_STARTING_STACK,
        )
        self._isa_cache: dict = {}  # cleared per select_action call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state) -> int:
        """Run search from the current decision state and return a sampled action."""
        assert not state.is_terminal() and not state.is_chance_node(), \
            "select_action called on non-decision state"

        self._isa_cache = {}
        local_table = StrategyTable()
        for i in range(self.iterations):
            self._mccfr(state.clone(), i % 2, local_table)
        self._isa_cache = {}

        root_infoset, root_actions = self._infoset_and_actions(state)
        if not root_actions:
            return int(state.legal_actions()[0])

        avg = local_table.average_strategy()
        policy = avg.get(root_infoset)
        return self._sample_from_policy(policy, root_actions)

    # ------------------------------------------------------------------
    # MCCFR core
    # ------------------------------------------------------------------

    def _mccfr(self, state, traverser: int, local_table: StrategyTable) -> float:
        if state.is_terminal():
            return float(state.returns()[traverser])

        if state.is_chance_node():
            # Search boundary: roll out the rest of the hand with blueprint.
            return self._blueprint_rollout(state, traverser)

        infoset, actions = self._infoset_and_actions(state)
        if not actions:
            return 0.0

        current_player = state.current_player()
        strategy = local_table.regret_matching(infoset, actions)
        local_table.add_average_strategy(infoset, actions, strategy, 1.0)

        if current_player != traverser:
            # Opponent node: sample one action (external sampling).
            action = self._sample_action(actions, strategy)
            state.apply_action(int(action))
            return self._mccfr(state, traverser, local_table)

        # Traverser node: expand every action and update regrets.
        action_values = []
        node_value = 0.0
        for action, prob in zip(actions, strategy):
            child = state.clone()
            child.apply_action(int(action))
            v = self._mccfr(child, traverser, local_table)
            action_values.append(v)
            node_value += prob * v

        deltas = [v - node_value for v in action_values]
        local_table.batch_add_regret(infoset, _action_keys(actions), deltas)
        return node_value

    # ------------------------------------------------------------------
    # Blueprint rollout (leaf value estimator)
    # ------------------------------------------------------------------

    def _blueprint_rollout(self, state, traverser: int) -> float:
        """Estimate EV from a chance/terminal state via blueprint rollouts."""
        if state.is_terminal():
            return float(state.returns()[traverser])

        total = 0.0
        for _ in range(self.rollout_samples):
            s = state.clone()
            while not s.is_terminal():
                # Deal any pending street cards before asking for a decision.
                while s.is_chance_node():
                    outcomes = s.chance_outcomes()
                    acts, probs = zip(*outcomes)
                    a = int(self._rng.choice(list(acts), p=list(probs)))
                    s.apply_action(a)
                if s.is_terminal():
                    break
                a = self._blueprint_action(s)
                s.apply_action(int(a))
            total += float(s.returns()[traverser])

        return total / self.rollout_samples

    def _blueprint_action(self, state) -> int:
        """Sample one action from blueprint strategy for the current decision state."""
        infoset, actions = self._infoset_and_actions(state)
        if not actions:
            legal = state.legal_actions()
            return int(self._rng.choice([int(a) for a in legal]))
        return self._sample_from_policy(self.blueprint.get(infoset), actions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infoset_and_actions(self, state):
        iss = state.information_state_string()
        cached = self._isa_cache.get(iss)
        if cached is not None:
            return cached
        parsed = parse_poker_string(iss)
        infoset = _infoset_from_parsed(parsed)
        actions = self._abstractor.select_action_specs_direct(parsed)
        result = (infoset, actions)
        self._isa_cache[iss] = result
        return result

    def _sample_from_policy(self, policy, actions) -> int:
        """Sample an action from a policy dict; falls back to uniform."""
        if policy:
            weighted = []
            total = 0.0
            for action in actions:
                key = _action_key(action)
                w = float(policy.get(key, 0.0))
                if w > 0:
                    weighted.append((action, w))
                    total += w
            if total > 0:
                r = self._rng.random() * total
                cum = 0.0
                for action, w in weighted:
                    cum += w
                    if cum >= r:
                        return int(action)
                return int(weighted[-1][0])
        # Uniform fallback
        idx = int(self._rng.integers(len(actions)))
        return int(actions[idx])

    def _sample_action(self, actions, strategy):
        r = self._rng.random()
        cum = 0.0
        for action, prob in zip(actions, strategy):
            cum += prob
            if cum >= r:
                return action
        return actions[-1]
