#should mostly be working
#still issues with negativity
#all in is super confusing
import random
from collections import deque
from typing import List, Tuple
from treys import Card, Evaluator, Deck
evaluator = Evaluator()

class Player:
    def __init__(self, name: str, initial_stack: int):
        self.name = name
        self.hand = []
        self.stack = initial_stack
        self.current_bet = 0
        self.bets = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
        self.folded = False
        self.all_in = False
        self.moved = False

    def __repr__(self):
        return f"Player(name={self.name})"

    def receive_card(self, card):
        self.hand = card

    def response(self):
        x = random.randint(1,3)
        if x == 1:
            return "call"
        if x == 2:
            return "fold"
        if x == 3:
            num = random.randint(1,10)
            return f"raise {num}"
        

    def action(self, bet, round, decision = None):
        self.moved = False
        if decision != None:
            return
        while True:
            try:
                decision = self.response()
                if decision == "fold":
                    self.folded = True
                    self.bets[round] = 0
                    print(f"{self.name} folded.")
                    return ["Fold", 0]
                
                if decision == "call":
                    if bet < 0 or self.stack <= 0:
                        print("Invalid bet or insufficient chips.")
                        continue
                    if bet >= self.stack:
                        self.all_in = round
                        self.current_bet = self.stack
                        self.bets[round] = self.current_bet
                        print(f"{self.name} went all-in with {self.stack}.")
                        return ["All In", self.current_bet]
                    call_amount = min(self.stack, bet - self.current_bet)
                    if call_amount < 0:
                        print("Invalid call amount.")
                        continue
                    self.current_bet += call_amount
                    self.bets[round] = self.current_bet
                    print(f"{self.name} called with {call_amount}.")
                    return ["Call", self.current_bet]
                
                if decision == "all-in":
                    if self.stack <= 0:
                        print("No chips to go all-in.")
                        continue
                    self.current_bet += self.stack
                    self.all_in = round
                    self.bets[round] = self.current_bet
                    print(f"{self.name} went all-in with {self.stack}.")
                    return ["All In", self.current_bet]
                
                if decision.startswith("raise "):
                    if self.stack <= 0:
                        print("Not enough chips to raise.")
                        continue
                    try:
                        raise_amount = int(decision.split()[1])
                        if raise_amount <= 0:
                            print("Raise amount must be positive.")
                            continue
                        total_bet = self.current_bet + raise_amount
                        if total_bet <= bet:
                            print(f"Total bet must be greater than current bet {bet}")
                            continue
                        if total_bet > self.stack:
                            print(f"Cannot raise more than your stack ({self.stack}).")
                            continue
                        if total_bet == self.stack:
                            self.all_in = round
                            self.current_bet = self.stack
                            self.bets[round] = self.current_bet
                            print(f"{self.name} went all-in with {self.stack}.")
                            return ["All In", self.current_bet]
                        self.current_bet = total_bet
                        self.bets[round] = self.current_bet
                        print(f"{self.name} raised to {self.current_bet}.")
                        return ["Raise", self.current_bet]
                    except IndexError:
                        print("Invalid raise format. Use 'raise [amount]'")
                
                print("Invalid input. Use 'fold', 'call', 'all-in', or 'raise [amount]'")
            except ValueError:
                print("Invalid raise amount. Please enter a valid number.")

    
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.bets = {}
        self.folded = False
        self.all_in = False
        self.moved = False

class PokerGame:
    def __init__(self, players: List[str], starting_stack: int = 1000):
        self.deck = Deck()
        self.players = [Player(name, starting_stack) for name in players]
        self.current_bet = 0
        self.evaluator = Evaluator()
        self.community_cards = []
        self.pot = 0

    def _deal_initial_cards(self):
        for player in self.players:
            cards = self.deck.draw(2)
            player.receive_card(cards)

    def _deal_community_cards(self, count: int):
        cards = self.deck.draw(count)
        self.community_cards = self.community_cards + cards


    def _evaluate_hand(self, player: Player) -> int:
        #player_hand = [card.to_treys() for card in player.hand]
        #community_cards = [card.to_treys() for card in self.community_cards]
        Card.print_pretty_cards(self.community_cards + player.hand)
        return self.evaluator.evaluate(self.community_cards, player.hand)

    def determine_winner(self):
        active_players = [player for player in self.players if not player.folded]
        if not active_players:
            return []
        
        scored_players = [(player, self.evaluator.evaluate(self.community_cards, player.hand)) 
                        for player in active_players]
        scored_players.sort(key=lambda x: x[1])  # Lower scores are better
        
        for player, score in scored_players:
            total_won = 0
            for round_name in ['preflop', 'flop', 'turn', 'river']:
                if round_name not in player.bets:
                    continue
                player_bet = player.bets[round_name]
                if player_bet > 0:
                    round_winnings = 0
                    for other_player in self.players:
                        if round_name in other_player.bets:
                            win_amount = min(player_bet, other_player.bets[round_name])
                            round_winnings += win_amount
                            other_player.bets[round_name] -= win_amount
                            other_player.stack -= win_amount
                    total_won += round_winnings
            
            player.stack += total_won

        return [player for player, score in scored_players]

    def _betting_round(self, round, current_bet = 0):
        active_players = deque(player for player in self.players if not (player.folded or player.all_in))
        #for i in range(order):
        #    active_players.rotate(-1)
        betting = True
        while betting:
            if len(active_players) == 1:
                return
            for i in active_players:
                print (i.name, i.bets)
            try:
                current_player = active_players[0]
            except:
                betting = False
            if not current_player.moved:
                decision = current_player.action(current_bet, round)
                if decision[0] == "Fold":
                    current_player.folded = True
                    active_players.popleft()  # Remove the player from the deque
                elif decision[0] == "All In":
                    if current_bet < decision[1]:
                        current_bet = decision[1]
                        for player in active_players:
                            player.moved = False
                        current_player.moved = True
                    active_players.popleft()  # Move to the next player
                elif decision[0] == "Raise":
                    current_bet = decision[1]
                    for player in active_players:
                        player.moved = False
                    current_player.moved = True
                    active_players.rotate(-1)  # Move to the next player
                else:  # Default case for "Call" or other actions
                    current_player.moved = True
                    active_players.rotate(-1)  # Move to the next player
            else:
                betting = False
        for player in active_players:
            player.current_bet = 0
            player.moved = False
        return

    def state_check(self, round):
        #Display the current state of all players and the pot.
        print(f"\n--- Game State --- {round}")
        print(f"Pot: {self.pot} chips")
        totalchips = self.pot
        
        for player in self.players:
            status = "Folded" if player.folded else "Active"
            totalchips = totalchips + player.stack
            print(f"{player.name}: {player.stack} chips, Status: {status}")
            Card.print_pretty_cards(player.hand)
        print(f"Community: Total chips: {totalchips} Pot: {self.pot}")
        Card.print_pretty_cards(self.community_cards)
        print("TOTAL CHIPS " + str(totalchips))

    def play_round(self):
        self.deck = Deck()  # Create a fresh deck for each round
        self.community_cards = []
        self.pot = 0

        for player in self.players:
            player.clear_hand()

        self._deal_initial_cards()
        #still working on small big blind
        self.players[0].action(0, "preflop", decision = "raise 5")  # Small blind
        self.players[1].action(0, "preflop", decision = "raise 10")
        self._betting_round("preflop",10) #order diff
        self.state_check("preflop")
        self._deal_community_cards(3)
        self._betting_round("flop")
        self.state_check("flop")
        self._deal_community_cards(1)
        self._betting_round("turn")
        self.state_check("turn")
        self._deal_community_cards(1)
        self._betting_round("river")
        self.state_check("river")
        self.determine_winner()
        self.pot = 0
        self.state_check("end")

    def play_game(self, num_rounds: int = 3):
        self.players = [player for player in self.players if player.stack > 0]
        for _ in range(num_rounds):
            self.play_round()
            self.players.append(self.players.pop(0))


# Example usage

if __name__ == "__main__":
    game = PokerGame(["Alice", "Bob", "Charlie"])
    game.play_game()