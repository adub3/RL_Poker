import random
from typing import List, Tuple
from treys import Card, Evaluator, Deck
evaluator = Evaluator()

class Player:
    def __init__(self, name: str, initial_stack: int):
        self.name = name
        self.hand = []
        self.stack = initial_stack
        self.current_bet = 0
        self.folded = False
        self.all_in = False

    def bet(self, amount: int):
        if amount > self.stack:
            raise ValueError("Not enough chips to bet")
        self.stack -= amount
        self.current_bet += amount
        return amount

    def receive_card(self, card):
        self.hand = card

    def action():
        
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

class PokerGame:
    def __init__(self, players: List[str], starting_stack: int = 1000):
        self.deck = Deck()
        self.players = [Player(name, starting_stack) for name in players]
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.button_index = 0
        self.evaluator = Evaluator()

    def _deal_initial_cards(self):
        for player in self.players:
            cards = self.deck.draw(2)
            player.receive_card(cards)
            print(player,  " cards:")
            Card.print_pretty_cards(player.hand)

    def _deal_community_cards(self, count: int):
        cards = self.deck.draw(count)
        self.community_cards = self.community_cards + cards


    def _evaluate_hand(self, player: Player) -> int:
        #player_hand = [card.to_treys() for card in player.hand]
        #community_cards = [card.to_treys() for card in self.community_cards]
        Card.print_pretty_cards(self.community_cards + player.hand)
        return self.evaluator.evaluate(self.community_cards, player.hand)

    def determine_winner(self) -> List[Player]:
        best_score = float('inf')
        best_score = 10000000
        winners = []
        for player in self.players:
            if not player.folded:
                score = self.evaluator.evaluate(self.community_cards, player.hand)
                if score < best_score:
                    best_score = score
                    winners = [player]
                elif score == best_score:
                    winners.append(player)

        return winners
        
    def _betting_round(self):
        # Count active players
        active_players = [player for player in self.players if not player.folded]
        if len(active_players) == 1:
            # If only one player remains, award them the pot and skip the round
            winner = active_players[0]
            winner.stack += self.pot
            print(f"{winner.name} is the only player remaining and wins the pot of {self.pot} chips!")
            self.pot = 0
            return  # Skip the rest of the round
        
        for player in self.players:
            if player.folded:
                continue  # Skip players who have already folded
            
            # Randomly choose an action
            action = random.choice(["fold", "call", "raise"])
            
            if action == "fold":
                player.folded = True
                print(f"{player.name} folds.")
            
            elif action == "call":
                call_amount = self.current_bet - player.current_bet
                if call_amount > 0:
                    try:
                        player.bet(call_amount)
                        self.pot += call_amount
                        print(f"{player.name} calls {call_amount}.")
                    except ValueError:
                        # Player cannot match the bet and folds
                        player.folded = True
                        print(f"{player.name} cannot call and folds.")
                else:
                    print(f"{player.name} checks.")
            
            elif action == "raise":
                raise_amount = random.randint(1, min(player.stack, 50))  # Raise by a random amount up to 50 or their stack
                new_bet = self.current_bet + raise_amount
                if new_bet > player.stack:
                    raise_amount = player.stack - self.current_bet
                
                if raise_amount > 0:
                    player.bet(raise_amount)
                    self.current_bet += raise_amount
                    self.pot += raise_amount
                    print(f"{player.name} raises by {raise_amount} to {self.current_bet}.")
                else:
                    print(f"{player.name} cannot raise and checks.")
        if len(active_players) == 1:
            # If only one player remains, award them the pot and skip the round
            winner = active_players[0]
            winner.stack += self.pot
            print(f"{winner.name} is the only player remaining and wins the pot of {self.pot} chips!")
            self.pot = 0
            return  # Skip the rest of the round

    """
    def _betting_round(self):
        for player in self.players:

            if not player.folded:
                if random.random() < 0.5:
                    bet_amount = min(self.current_bet, player.stack)
                    if bet_amount > 0:
                        player.bet(bet_amount)
                        self.pot += bet_amount
                else:
                    player.folded = True"""

    def state_check(self):
        """Display the current state of all players and the pot."""
        print("\n--- Game State ---")
        print(f"Pot: {self.pot} chips")
        totalchips = self.pot
        for player in self.players:
            hand_display = " ".join(str(card) for card in player.hand) if player.hand else "No cards"
            status = "Folded" if player.folded else "Active"
            totalchips = totalchips + player.stack
            print(f"{player.name}: {player.stack} chips, Hand: {hand_display}, Status: {status}")
        print("TOTAL CHIPS " + str(totalchips))

    def _distribute_pot(self, winners: List[Player]):
        if not winners:
            return
        split_pot = self.pot // len(winners)
        for winner in winners:
            winner.stack += split_pot
        self.pot = 0

    def play_round(self):
        self.deck = Deck()  # Create a fresh deck for each round
        self.community_cards = []
        self.pot = 0

        for player in self.players:
            player.clear_hand()

        self.state_check()
        self._deal_initial_cards()
        self._betting_round()
        
        self._deal_community_cards(3)
        self._betting_round()
        self._deal_community_cards(1)
        self._betting_round()
        self._deal_community_cards(1)
        self._betting_round()

        winners = self.determine_winner()
        print(f"Winners: {winners}")
        self._distribute_pot(winners)
        self.state_check()

    def play_game(self, num_rounds: int = 15):
        for _ in range(num_rounds):
            self.play_round()


# Example usage
if __name__ == "__main__":
    game = PokerGame(["Alice", "Bob", "Charlie"])
    game.play_game()
