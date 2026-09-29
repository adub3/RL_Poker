"""
Checks for the local best response evaluator.

    .venv/bin/python -B tools/evaluation/test_lbr.py
"""

import sys
from pathlib import Path

import numpy as np
import pyspiel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstraction import parse_poker_string  # noqa: E402
from ai import _infoset_from_parsed        # noqa: E402
from const import game_config              # noqa: E402
from ai import PreflopActionAbstractor     # noqa: E402
from lbr_exploitability import (           # noqa: E402
    CARD_ID,
    COMBO_NAMES,
    COMBOS,
    LocalBestResponse,
    betting_context,
)
from translation import ActionTranslator, pseudo_harmonic, raise_fraction  # noqa: E402

GAME = pyspiel.load_game("universal_poker", game_config)


class UniformBlueprint:
    """No entries: the bot plays uniformly at random over its menu."""

    def get(self, key):
        return None


class PairsContinueBlueprint:
    """Folds every non-pair preflop when it can; pairs always call."""

    def get(self, key):
        if key.startswith("[PF:") and key[4] != key[5]:
            return {"fold": 1.0, "call": 1e-9}
        return {"call": 1.0}


def lbr(blueprint, fractions=(1.0,), translate=True):
    return LocalBestResponse(GAME, blueprint, "v3", fractions, equity_samples=100,
                             translate=translate)


def deal(deck):
    state = GAME.new_initial_state()
    for card in deck[:4]:
        state.apply_action(int(card))
    return state


def random_decision_states(count, seed=0):
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < count:
        state = GAME.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = [o for o, _ in state.chance_outcomes()]
                state.apply_action(int(rng.choice(outcomes)))
                continue
            states.append(state.clone())
            legal = [action for action in state.legal_actions() if action != 0]
            # Mostly calls, some raises, so hands reach later streets.
            state.apply_action(int(legal[0] if rng.random() < 0.7 else rng.choice(legal)))
    return states


def test_card_keys_match_bot_infosets():
    """The range lookup builds exactly the key the bot would for each combo."""
    player = lbr(UniformBlueprint())
    rng = np.random.default_rng(1)
    for state in random_decision_states(60):
        parsed = parse_poker_string(state.information_state_string(0))
        public = tuple(parsed.get("Public") or ())
        keys, index = player.card_keys(public)
        context = betting_context(parsed)
        board = {CARD_ID[card] for card in public}
        for i in rng.choice(len(COMBOS), 40, replace=False):
            if board & set(COMBOS[i]):
                continue
            for private in (COMBO_NAMES[i], COMBO_NAMES[i][::-1]):
                expected = _infoset_from_parsed({**parsed, "Private": list(private)})
                assert keys[index[i]] + context == expected, (keys[index[i]], expected)


def test_decisions_ignore_bot_cards():
    """Same public history and LBR cards, different bot cards: same decision."""
    player = lbr(UniformBlueprint(), fractions=(0.5, 1.0))
    mine = (CARD_ID["Ah"], CARD_ID["Kd"])
    choices = set()
    for bot_cards in (("7c", "7s"), ("2d", "3h"), ("Qs", "Qc"), ("9h", "8h")):
        # P0 (the bot) is dealt first; P1 (LBR) acts first preflop.
        state = deal([CARD_ID[bot_cards[0]], CARD_ID[bot_cards[1]], *mine])
        hand = {"range": np.ones(len(COMBOS)), "cards": mine, "equity": {}, "seen": set()}
        hand["range"][[i for i, combo in enumerate(COMBOS) if set(combo) & set(mine)]] = 0
        choices.add(player.choose(state, 1, np.random.default_rng(5), hand))
    assert len(choices) == 1, choices


def test_range_update_follows_blueprint():
    """After the bot calls a raise, only combos that call stay in its range."""
    player = lbr(PairsContinueBlueprint())
    state = deal([CARD_ID[c] for c in ("2c", "3d", "Ah", "Kd")])
    state.apply_action(300)  # P1 raises; P0 (the bot) to act
    action_ids, probs, missing = player.bot_policy(state, 1)
    weights = probs[:, action_ids.index(1)]
    pairs = np.array([a // 4 == b // 4 for a, b in COMBOS])
    assert np.all(weights[pairs] > 0.99)
    assert np.all(weights[~pairs] < 1e-6)
    assert not missing.any()


def test_equity():
    player = LocalBestResponse(GAME, UniformBlueprint(), "v3", (1.0,), equity_samples=3000)
    rng = np.random.default_rng(0)
    active = np.zeros(len(COMBOS), dtype=bool)
    kk = COMBOS.index(tuple(sorted((CARD_ID["Kc"], CARD_ID["Kd"]))))
    active[kk] = True
    eq = player.equity((CARD_ID["Ah"], CARD_ID["As"]), (), active, rng)
    assert abs(eq[kk] - 0.82) < 0.03, eq[kk]

    # River: exact. Ah Kh on Qh Jh Th 2c 3d is a royal flush; a board-playing combo ties.
    board = tuple(CARD_ID[c] for c in ("Qh", "Jh", "Th", "2c", "3d"))
    active[:] = True
    eq = player.equity((CARD_ID["Ah"], CARD_ID["Kh"]), board, active, rng)
    assert eq[COMBOS.index(tuple(sorted((CARD_ID["4c"], CARD_ID["5d"]))))] == 1.0
    board = tuple(CARD_ID[c] for c in ("As", "Ks", "Qs", "Js", "Ts"))
    eq = player.equity((CARD_ID["2c"], CARD_ID["3d"]), board, active, rng)
    assert eq[COMBOS.index(tuple(sorted((CARD_ID["4c"], CARD_ID["5d"]))))] == 0.5


def test_beats_uniform_bot():
    """A bot that plays uniformly at random is hugely exploitable."""
    player = lbr(UniformBlueprint())
    stats = {"bot_decisions": 0, "bot_missing": 0}
    values = []
    for index in range(40):
        deck = np.random.default_rng((7, index)).permutation(52)
        for seat in (0, 1):
            values.append(player.play(GAME, deck, seat, np.random.default_rng((7, index, seat)), stats))
    assert np.mean(values) > 500, np.mean(values)  # > 5 bb per hand
    assert stats["bot_missing"] == stats["bot_decisions"]


def translator():
    abstractor = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="v3")
    return ActionTranslator(GAME, abstractor, parse_poker_string)


def test_pseudo_harmonic():
    assert pseudo_harmonic(0.5, 0.5, 1.0) == 1.0
    assert pseudo_harmonic(1.0, 0.5, 1.0) == 0.0
    assert abs(pseudo_harmonic(0.75, 0.5, 1.0) - 3 / 7) < 1e-12  # (0.25 * 1.5) / (0.5 * 1.75)


def test_menu_lines_are_not_translated():
    """A history made only of menu actions translates to itself."""
    t = translator()
    rng = np.random.default_rng(3)
    for _ in range(30):
        state = deal(rng.permutation(52))
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice([o for o, _ in state.chance_outcomes()])))
                continue
            (weight, branch), = t.branches(state)
            assert weight == 1.0 and branch.history() == state.history()
            _, specs = t.specs(state)
            state.apply_action(int(specs[rng.integers(len(specs))]))


def open_to(amount):
    """P1 (small blind) opens to `amount`; P0, the bot, to act."""
    state = deal([CARD_ID[c] for c in ("2c", "3d", "Ah", "Kd")])
    state.apply_action(amount)
    return state


def test_off_menu_raise_splits_between_neighbours():
    t = translator()
    parsed, specs = t.specs(deal([CARD_ID[c] for c in ("2c", "3d", "Ah", "Kd")]))
    menu = {int(spec): spec.key for spec in specs}
    # Pot after calling is 200: open_2_5bb (to 250) is 0.75 pot, open_3bb (to 300)
    # is 1.0 pot, and an open to 270 is 0.85 pot.
    assert menu[250] == "open_2_5bb" and menu[300] == "open_3bb"
    branches = t.branches(open_to(270))
    amounts = {branch.history()[-1]: weight for weight, branch in branches}
    assert set(amounts) == {250, 300}
    x = raise_fraction(270, parsed, 100, 10_000)
    assert abs(x - 0.85) < 1e-12
    assert abs(amounts[250] - pseudo_harmonic(x, 0.75, 1.0)) < 1e-9


def test_translation_uses_trained_entries():
    """Facing an off-menu open, the bot mixes the strategies it has for the
    neighbouring sizes instead of finding no entry and playing randomly."""
    class Table:
        def get(self, key):
            if key.endswith("[tc:1bb][eff:100bb][seq:r:2_5bb]"):
                return {"fold": 1.0}
            if key.endswith("[tc:2bb][eff:100bb][seq:r:3bb]"):
                return {"call": 1.0}
            return None

    state = open_to(270)
    action_ids, probs, missing = lbr(Table()).bot_policy(state, 1)
    queens = probs[COMBOS.index(tuple(sorted((CARD_ID["Qc"], CARD_ID["Qd"]))))]
    x = 0.85  # the open to 270, as a fraction of the pot after calling
    assert abs(queens[action_ids.index(0)] - pseudo_harmonic(x, 0.75, 1.0)) < 1e-9
    assert abs(queens[action_ids.index(1)] - (1 - pseudo_harmonic(x, 0.75, 1.0))) < 1e-9
    assert missing.max() == 0.0
    _, _, untranslated_missing = lbr(Table(), translate=False).bot_policy(state, 1)
    assert untranslated_missing.min() == 1.0


if __name__ == "__main__":
    test_card_keys_match_bot_infosets()
    test_decisions_ignore_bot_cards()
    test_range_update_follows_blueprint()
    test_equity()
    test_beats_uniform_bot()
    test_pseudo_harmonic()
    test_menu_lines_are_not_translated()
    test_off_menu_raise_splits_between_neighbours()
    test_translation_uses_trained_entries()
    print("lbr tests passed")
