import sys
sys.setrecursionlimit(2000)
import random
from collections import defaultdict

class KuhnPokerCFR:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: {'pass': 0.0, 'bet': 0.0})
        self.strategy_sum = defaultdict(lambda: {'pass': 0.0, 'bet': 0.0})
        self.cards = [1, 2, 3]

    def get_strategy(self, infoset):
        """Compute the strategy for the given information set."""
        regrets = self.regret_sum[infoset]
        strategy = {}
        normalizing_sum = sum(max(0, regret) for regret in regrets.values())
        for action in ['pass', 'bet']:
            if normalizing_sum > 0:
                strategy[action] = max(0, regrets[action]) / normalizing_sum
            else:
                strategy[action] = 1 / 2  # Default equal probabilities
        self.strategy_sum[infoset]['pass'] += strategy['pass']
        self.strategy_sum[infoset]['bet'] += strategy['bet']
        return strategy

    def cfr(self, cards, history, probs):
        """Recursive CFR calculation."""
        player = len(history) % 2  # Current player (0 or 1)
        opponent = 1 - player      # Opponent
        infoset = f"{cards[player]}:{history}"
        print(f"Debug: Player {player}, History {history}, Infoset {infoset}")
        # Terminal states
        if history == 'pp':  # Both players passed
            return 1 if cards[player] > cards[opponent] else -1
        if history[-2:] == 'bf':  # Bet followed by a fold
            return 1 if history[-1] == 'f' else -1

        # Get strategy for current infoset
        strategy = self.get_strategy(infoset)

        # Compute utilities for each action
        utilities = {}
        node_utility = 0
        for action in ['pass', 'bet']:
            next_history = history + ('p' if action == 'pass' else 'b')
            next_probs = probs[:]
            next_probs[player] *= strategy[action]
            utilities[action] = -self.cfr(cards, next_history, next_probs)
            node_utility += strategy[action] * utilities[action]

        # Update regrets
        for action in ['pass', 'bet']:
            regret = utilities[action] - node_utility
            self.regret_sum[infoset][action] += probs[opponent] * regret
        
        return node_utility



    def train(self, iterations):
        """Train the CFR model."""
        for _ in range(iterations):
            random.shuffle(self.cards)
            self.cfr(self.cards, '', [1, 1])  # Start with equal probabilities for both players

    def get_average_strategy(self):
        """Compute the average strategy from the accumulated strategy sums."""
        avg_strategy = {}
        for infoset in self.strategy_sum:
            strategy_sum = self.strategy_sum[infoset]
            total = sum(strategy_sum.values())
            avg_strategy[infoset] = {
                action: strategy_sum[action] / total if total > 0 else 0.5
                for action in ['pass', 'bet']
            }
        return avg_strategy

# Example Usage
if __name__ == "__main__":
    cfr = KuhnPokerCFR()
    cfr.train(10000)  # Train for 10,000 iterations
    avg_strategy = cfr.get_average_strategy()

    print("Trained Average Strategy:")
    for infoset, strategy in avg_strategy.items():
        print(f"{infoset}: {strategy}")
