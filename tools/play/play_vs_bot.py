"""
Play heads-up no-limit Texas Hold'em against the MCCFR bot.

Usage:
    .venv/bin/python tools/play/play_vs_bot.py
    .venv/bin/python tools/play/play_vs_bot.py --checkpoint checkpoints/fullgame_100bb/mccfr_table_iter_00815000.json.gz
    .venv/bin/python tools/play/play_vs_bot.py --seat 1          # you are BB
    .venv/bin/python tools/play/play_vs_bot.py --bot-only 20     # bot vs blueprint, N hands
    .venv/bin/python tools/play/play_vs_bot.py --iterations 500  # stronger (slower) search

Actions during play:
    f / fold
    c / check / call
    <number>   raise to that chip total (e.g. "300")
"""

import argparse
import random
import sys
import time
from pathlib import Path

import pyspiel

ROOT = Path(__file__).resolve().parents[2]
NEW_CODE = ROOT / "new_code"
sys.path.insert(0, str(NEW_CODE))

from abstraction import parse_poker_string  # noqa: E402
from ai import load_table  # noqa: E402
from const import game_config as _BASE_CONFIG  # noqa: E402
from rts import RealTimeSearch, _BIG_BLIND, _STARTING_STACK  # noqa: E402


RANK_NAMES = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}
SUIT_SYMBOLS = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}

STREETS = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


def _game_config():
    cfg = dict(_BASE_CONFIG)
    cfg["stack"] = f"{_STARTING_STACK} {_STARTING_STACK}"
    return cfg


def _fmt_card(c):
    rank = c[0]
    suit = c[1] if len(c) > 1 else ""
    rank = RANK_NAMES.get(rank, rank)
    suit = SUIT_SYMBOLS.get(suit, suit)
    return f"{rank}{suit}"


def _fmt_cards(cards):
    return " ".join(_fmt_card(c) for c in cards) if cards else "(none)"


def _display_state(state, human_seat: int):
    parsed = parse_poker_string(state.information_state_string(human_seat))
    private = parsed.get("Private") or []
    public = parsed.get("Public") or []
    money = parsed.get("Money") or [0, 0]
    pot = int(parsed.get("Pot") or 0)
    street = int(parsed.get("Round") or 0)

    stack = int(money[human_seat])  # remaining chips

    print()
    print(f"  Street : {STREETS.get(street, street)}")
    if public:
        print(f"  Board  : {_fmt_cards(public)}")
    print(f"  Hand   : {_fmt_cards(private)}")
    print(f"  Pot    : {pot}  |  Your stack: {stack}")


def _get_human_action(state, human_seat: int) -> int:
    legal = sorted(state.legal_actions())
    has_fold = 0 in legal
    has_call_check = 1 in legal
    raises = [a for a in legal if a > 1]

    parts = []
    if has_fold:
        parts.append("[f]old")
    if has_call_check:
        action1_label = "call" if has_fold else "check"
        parts.append(f"[c]{action1_label}")
    if raises:
        parts.append(f"raise [{raises[0]}–{raises[-1]}]")

    while True:
        print(f"  Actions: {' | '.join(parts)}")
        raw = input("  > ").strip().lower()

        if raw in ("f", "fold") and has_fold:
            return 0
        if raw in ("c", "call", "check") and has_call_check:
            return 1
        try:
            amount = int(raw)
            # Accept exact match or snap to nearest legal raise
            if amount in legal:
                return amount
            valid_raises = [a for a in raises if a >= amount]
            if valid_raises:
                snapped = min(valid_raises)
                print(f"  Snapping to nearest raise: {snapped}")
                return snapped
            if raises:
                print(f"  Max raise is {raises[-1]}, using that.")
                return raises[-1]
        except ValueError:
            pass
        print("  Invalid input. Try: f  c  <chip amount>")


def play_hand(game, rts, human_seat: int) -> tuple[float, float]:
    """Play one hand human vs bot. Returns (human_return, bot_return)."""
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, probs = zip(*outcomes)
            state.apply_action(random.choices(list(acts), weights=list(probs))[0])
            continue

        player = state.current_player()

        if player == human_seat:
            _display_state(state, human_seat)
            action = _get_human_action(state, human_seat)
        else:
            print("  Bot is thinking…", end=" ", flush=True)
            t0 = time.perf_counter()
            action = rts.select_action(state)
            elapsed = time.perf_counter() - t0
            label = state.action_to_string(player, int(action))
            move = label.split("move=")[-1] if "move=" in label else label
            print(f"Bot: {move}  ({elapsed:.1f}s)")

        state.apply_action(int(action))

    returns = state.returns()
    return float(returns[human_seat]), float(returns[1 - human_seat])


def play_hand_bot_only(game, policy0, policy1) -> tuple[float, float]:
    """Play one hand with both sides using a policy (no human)."""
    state = game.new_initial_state()
    policies = [policy0, policy1]

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, probs = zip(*outcomes)
            state.apply_action(random.choices(list(acts), weights=list(probs))[0])
            continue
        player = state.current_player()
        action = policies[player].select_action(state, list(state.legal_actions()))
        state.apply_action(int(action))

    returns = state.returns()
    return float(returns[0]), float(returns[1])


class BlueprintPolicy:
    """Wraps the RTS in a simple policy interface for bot-only mode."""

    def __init__(self, rts: RealTimeSearch):
        self._rts = rts

    def select_action(self, state, _legal_actions):
        return self._rts.select_action(state)


def main():
    parser = argparse.ArgumentParser(description="Play vs MCCFR bot (RTS)")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fullgame_100bb/mccfr_table_iter_00815000.json.gz",
    )
    parser.add_argument(
        "--seat", type=int, default=0, choices=[0, 1],
        help="Your seat: 0=SB, 1=BB",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="RTS search iterations per decision (default 100 ~1.7s; 200 ~4s; 500 ~16s)",
    )
    parser.add_argument(
        "--bot-only", type=int, default=0, metavar="N",
        help="Skip human play; run N bot-vs-bot hands as a smoke test",
    )
    args = parser.parse_args()

    print(f"Loading blueprint from {args.checkpoint} …", end=" ", flush=True)
    table = load_table(args.checkpoint)
    blueprint = table.average_strategy()
    print(f"done  ({len(blueprint):,} infosets)")

    rts0 = RealTimeSearch(blueprint, iterations=args.iterations)

    game = pyspiel.load_game("universal_poker", _game_config())

    # ------------------------------------------------------------------
    # Bot-only smoke test
    # ------------------------------------------------------------------
    if args.bot_only > 0:
        rts1 = RealTimeSearch(blueprint, iterations=args.iterations)
        p0 = BlueprintPolicy(rts0)
        p1 = BlueprintPolicy(rts1)
        total = [0.0, 0.0]
        for hand in range(1, args.bot_only + 1):
            r0, r1 = play_hand_bot_only(game, p0, p1)
            total[0] += r0
            total[1] += r1
            print(f"  Hand {hand:3d}: P0={r0:+.0f}  P1={r1:+.0f}  (sum={r0+r1:+.0f})")
        print(f"\nTotals: P0={total[0]:+.0f}  P1={total[1]:+.0f}")
        return

    # ------------------------------------------------------------------
    # Human vs bot
    # ------------------------------------------------------------------
    human_seat = args.seat
    print(f"\nYou are P{human_seat} ({'SB' if human_seat == 0 else 'BB'}).")
    print("Type  f=fold  c=call/check  <number>=raise amount\n")

    net = 0.0
    hand_num = 0
    while True:
        hand_num += 1
        print(f"{'─'*40}")
        print(f"  HAND {hand_num}   (net so far: {net:+.0f})")
        print(f"{'─'*40}")

        h_ret, b_ret = play_hand(game, rts0, human_seat)
        net += h_ret

        print()
        print(f"  Hand result:  You {h_ret:+.0f}   Bot {b_ret:+.0f}")
        print(f"  Running net:  {net:+.0f}")
        print()

        again = input("  Play another hand? [Y/n] ").strip().lower()
        if again in ("n", "no", "q", "quit"):
            break

    print(f"\nFinal net over {hand_num} hand(s): {net:+.0f}")


if __name__ == "__main__":
    main()
