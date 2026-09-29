"""
Action translation: how the blueprint reads bets it never trained on.

The strategy table only has entries for betting lines built from the bot's own
action menu. When an opponent raises to another amount, the bot reads it as a
mix of the two nearest menu sizes A < x < B (as fractions of the pot after
calling), taking A with the pseudo-harmonic probability

    P(A) = (B - x)(1 + A) / ((B - A)(1 + x))

(Ganzfried & Sandholm, "Action Translation in Extensive-Form Games with Large
Action Spaces", 2013). Sizes below the smallest raise or above all-in map to
those. Every off-menu raise in the hand is translated, so a history becomes a
few weighted translated histories; the bot's strategy is the weighted mix of
the table's strategies for them, with each translated action mapped back to
the nearest real action.

Translated histories only differ from the real one in bet amounts, so the
cards (and each player's view of them) are the same.
"""

from ai import _street_raise_range


def pseudo_harmonic(x, a, b):
    """Probability of reading size x (a <= x <= b) as a rather than b."""
    return (b - x) * (1 + a) / ((b - a) * (1 + x))


def raise_fraction(amount, parsed, big_blind, stack):
    """A raise to `amount` as a fraction of the pot after calling."""
    last_top = _street_raise_range(parsed, big_blind, stack)[2]
    return (amount - last_top) / (2 * last_top)


class ActionTranslator:
    def __init__(self, game, abstractor, parse, min_weight=1e-3):
        """`abstractor` gives the bot's action menu; `parse` parses info strings."""
        self.game = game
        self.abstractor = abstractor
        self.parse = parse
        self.big_blind = abstractor.big_blind
        self.stack = abstractor.starting_stack
        self.min_weight = min_weight

    def specs(self, state):
        parsed = self.parse(state.information_state_string(state.current_player()))
        return parsed, self.abstractor.select_action_specs_direct(parsed)

    def fraction(self, amount, parsed):
        return raise_fraction(amount, parsed, self.big_blind, self.stack)

    def nearest(self, spec_key, amount, parsed, to_parsed, to_specs):
        """The action in `to_specs` matching `spec_key`, else the nearest raise.

        `amount` is the action's id in the spot `parsed` it comes from.
        """
        for spec in to_specs:
            if spec.key == spec_key:
                return int(spec)
        if amount in (0, 1):
            return amount
        raises = [int(spec) for spec in to_specs if int(spec) >= 2]
        if not raises:
            return 1
        x = self.fraction(amount, parsed)
        return min(raises, key=lambda a: abs(self.fraction(a, to_parsed) - x))

    def translate(self, action, parsed, specs, to_parsed, to_specs):
        """[(weight, action)] reading real `action` in the translated spot."""
        for spec in specs:
            if int(spec) == action:
                return [(1.0, self.nearest(spec.key, action, parsed, to_parsed, to_specs))]
        if action in (0, 1):
            return [(1.0, action)]
        x = self.fraction(action, parsed)
        sizes = sorted((self.fraction(int(spec), to_parsed), int(spec))
                       for spec in to_specs if int(spec) >= 2)
        if not sizes:
            return [(1.0, 1)]
        if x <= sizes[0][0]:
            return [(1.0, sizes[0][1])]
        if x >= sizes[-1][0]:
            return [(1.0, sizes[-1][1])]
        for (a, action_a), (b, action_b) in zip(sizes, sizes[1:]):
            if a <= x <= b:
                p = pseudo_harmonic(x, a, b)
                return [(p, action_a), (1 - p, action_b)]
        raise AssertionError("unreachable")

    def branches(self, state):
        """Weighted translated states at the same point as `state`.

        Branches where translation put someone all-in earlier than the real
        hand did can't follow it and are dropped. If none are left, the real
        state is returned unchanged.
        """
        real = self.game.new_initial_state()
        branches = [(1.0, self.game.new_initial_state())]
        for action in state.history():
            if real.is_chance_node():
                real.apply_action(action)
                for _, branch in branches:
                    branch.apply_action(action)
                continue
            parsed, specs = self.specs(real)
            player = real.current_player()
            following = []
            for weight, branch in branches:
                if branch.is_chance_node() or branch.is_terminal() \
                        or branch.current_player() != player:
                    continue
                to_parsed, to_specs = self.specs(branch)
                for share, translated in self.translate(action, parsed, specs,
                                                        to_parsed, to_specs):
                    if weight * share < self.min_weight:
                        continue
                    child = branch.clone()
                    child.apply_action(translated)
                    following.append((weight * share, child))
            real.apply_action(action)
            if not following:
                return [(1.0, state)]
            total = sum(weight for weight, _ in following)
            branches = [(weight / total, branch) for weight, branch in following]
        branches = [(w, b) for w, b in branches
                    if not b.is_terminal() and not b.is_chance_node()
                    and b.current_player() == state.current_player()]
        if not branches:
            return [(1.0, state)]
        total = sum(weight for weight, _ in branches)
        return [(weight / total, branch) for weight, branch in branches]
