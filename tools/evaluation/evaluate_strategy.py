import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

import pyspiel


ROOT = Path(__file__).resolve().parents[2]
NEW_CODE = ROOT / "new_code"
sys.path.insert(0, str(NEW_CODE))

from ai import PreflopActionAbstractor, StrategyTable, get_infostate  # noqa: E402
from const import game_config  # noqa: E402


class RandomPolicy:
    name = "random"

    def select_action(self, state, legal_actions):
        return random.choice(legal_actions)


class CallCheckPolicy:
    name = "call_check"

    def select_action(self, state, legal_actions):
        for action in legal_actions:
            label = action_label(state, action)
            if "call" in label or "check" in label:
                return action
        return min(legal_actions)


class TightPassivePolicy:
    name = "tight_passive"

    def select_action(self, state, legal_actions):
        for action in legal_actions:
            if "check" in action_label(state, action):
                return action
        for action in legal_actions:
            if "call" in action_label(state, action):
                return action
        for action in legal_actions:
            if "fold" in action_label(state, action):
                return action
        return min(legal_actions)


class StrategyTablePolicy:
    def __init__(self, strategy, fallback_policy=None):
        self.name = "strategy_table"
        self.strategy = strategy
        self.fallback_policy = fallback_policy or CallCheckPolicy()
        self.preflop_abstractor = PreflopActionAbstractor(big_blind=100)

    def select_action(self, state, legal_actions):
        infoset = get_infostate(state)
        policy = self.strategy.get(infoset)
        if not policy:
            return self.fallback_policy.select_action(state, legal_actions)

        weighted_actions = []
        total = 0.0
        for action in legal_actions:
            weight = float(policy.get(str(int(action)), 0.0))
            if weight > 0:
                weighted_actions.append((action, weight))
                total += weight

        if total <= 0:
            for spec in self.preflop_abstractor.select_action_specs(state, legal_actions):
                weight = float(policy.get(spec.key, 0.0))
                if weight > 0:
                    weighted_actions.append((int(spec), weight))
                    total += weight

        if total <= 0:
            return self.fallback_policy.select_action(state, legal_actions)

        threshold = random.random() * total
        cumulative = 0.0
        for action, weight in weighted_actions:
            cumulative += weight
            if cumulative >= threshold:
                return action
        return weighted_actions[-1][0]


def action_label(state, action):
    return state.action_to_string(state.current_player(), int(action)).lower()


def load_strategy(path):
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as strategy_file:
        data = json.load(strategy_file)

    if data and all(
        isinstance(node, dict) and "strategy_sum" in node and "regret" in node
        for node in data.values()
    ):
        return StrategyTable(data).average_strategy()
    return data


def make_policy(name, strategy_path=None):
    if name == "random":
        return RandomPolicy()
    if name == "call_check":
        return CallCheckPolicy()
    if name == "tight_passive":
        return TightPassivePolicy()
    if name == "strategy_table":
        if not strategy_path:
            raise ValueError("strategy_table policy requires --strategy")
        return StrategyTablePolicy(load_strategy(strategy_path))
    raise ValueError(f"Unknown policy: {name}")


def play_hand(game, policies):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions = [action for action, _ in outcomes]
            probs = [prob for _, prob in outcomes]
            state.apply_action(random.choices(actions, probs)[0])
            continue

        player = state.current_player()
        legal_actions = list(state.legal_actions())
        action = policies[player].select_action(state, legal_actions)
        state.apply_action(int(action))

    return state.returns()


def summarize(results):
    hands = len(results)
    total = sum(row["hero_return"] for row in results)
    wins = sum(1 for row in results if row["hero_return"] > 0)
    losses = sum(1 for row in results if row["hero_return"] < 0)
    ties = hands - wins - losses
    mean = total / hands if hands else 0.0
    return {
        "hands": hands,
        "total_return": total,
        "mean_return": mean,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def write_results(rows, path):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "hand",
                "hero_seat",
                "hero_return",
                "opponent_return",
                "bankroll",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary, path, hero_policy, opponent_policy):
    lines = [
        "# Evaluation Summary",
        "",
        f"Hero policy: {hero_policy.name}",
        f"Opponent policy: {opponent_policy.name}",
        "",
        f"Hands: {summary['hands']}",
        f"Total return: {summary['total_return']:.2f}",
        f"Mean return: {summary['mean_return']:.4f}",
        f"Wins: {summary['wins']}",
        f"Losses: {summary['losses']}",
        f"Ties: {summary['ties']}",
        "",
        "This is a high-variance bankroll simulation, not exploitability.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_charts(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    hands = [int(row["hand"]) for row in rows]
    bankroll = [float(row["bankroll"]) for row in rows]
    returns = [float(row["hero_return"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hands, bankroll, color="#3A6EA5")
    ax.set_title("Evaluation Bankroll Curve")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bankroll_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=30, color="#6B8E23")
    ax.set_title("Per-Hand Return Distribution")
    ax.set_xlabel("Hero return")
    ax.set_ylabel("Hands")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "return_histogram.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="new_code/blackjack_smoke.txt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--hero-seat", type=int, default=0)
    parser.add_argument("--output-dir", default="eval/smoke")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game = pyspiel.load_game("universal_poker", game_config)
    strategy_path = args.checkpoint or args.strategy
    hero_policy = make_policy("strategy_table", strategy_path)
    opponent_policy = make_policy(args.opponent)

    rows = []
    bankroll = 0.0
    for hand in range(1, args.hands + 1):
        policies = [opponent_policy, opponent_policy]
        policies[args.hero_seat] = hero_policy
        returns = play_hand(game, policies)
        hero_return = float(returns[args.hero_seat])
        opponent_return = float(returns[1 - args.hero_seat])
        bankroll += hero_return
        rows.append(
            {
                "hand": hand,
                "hero_seat": args.hero_seat,
                "hero_return": hero_return,
                "opponent_return": opponent_return,
                "bankroll": bankroll,
            }
        )

    summary = summarize(rows)
    write_results(rows, output_dir / "hands.csv")
    write_summary(summary, output_dir / "summary.md", hero_policy, opponent_policy)
    write_charts(rows, output_dir)

    print(f"Wrote evaluation to {output_dir}")
    print(f"Mean return per hand: {summary['mean_return']:.4f}")


if __name__ == "__main__":
    main()
