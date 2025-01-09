import random
from collections import deque
from typing import List, Tuple
from treys import Card, Evaluator, Deck
#from player_logic import Player
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

    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.bets = {}
        self.folded = False
        self.all_in = False
        self.moved = False
        
    def action(self, bet, round, decision = None):
        self.moved = False
        if decision != None:
            return
        while True:
            try:
                decision = input(f"{self.name}, your stack: {self.stack}. Current bet: {bet}. Raised: {self.moved} Enter 'fold', 'call', 'all-in', or 'raise [amount]': ").strip().lower()
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

class PokerGame:
    def __init__(self, players: List[str], starting_stack: int = 1000):
        self.deck = Deck()
        self.players = deque(Player(name, starting_stack) for name in players)
        self.current_bet = 0
        self.evaluator = Evaluator()
        self.community_cards = []
        self.pot = 0
        self.active_players = self.players
        self.preflop, self.flop, self.turn, self.river = (True, True, True, True)

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
    
    def check_all_players_moved(self):
        for player in self.players:
            if not player.moved:
                return False
        return True
    
    def reset_actions(self):
        for player in self.players:
            player.current_bet = 0
            player.moved = False

    def nextplayer(self, decision, active_players):
        print("nextplayer called")
        #if len(active_players) == 1:
        #    active_players[0].moved = True
        #    return
        #for i in active_players:
        #    print (i.name, i.bets)
        current_player = active_players[0]
        print(current_player.name, current_player.moved)
        if not current_player.moved:
            print (decision[0] == "Fold")
            if decision[0] == "Fold":
                current_player.folded = True
                print(active_players)
                active_players.popleft()  # Remove the player from the deque
                print(active_players)
            elif decision[0] == "All In":
                if self.current_bet < decision[1]:
                    self.current_bet = decision[1]
                    for player in active_players:
                        player.moved = False
                    current_player.moved = True
                active_players.popleft()  # Move to the next player
            elif decision[0] == "Raise":
                self.current_bet = decision[1]
                for player in active_players:
                    player.moved = False
                current_player.moved = True
                active_players.rotate(-1)  # Move to the next player
            else:  # Default case for "Call" or other actions
                current_player.moved = True
                active_players.rotate(-1)  # Move to the next player

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

    def gamelogic(self):
        self.deck = Deck()  # Create a fresh deck for each round
        self.community_cards = []
        self.pot = 0
        self.active_players = deque(player for player in self.players if not (player.folded or player.all_in))
        for player in self.players:
            player.clear_hand()
        self._deal_initial_cards()
        #still working on small big blind
        #self.players[0].action(0, "preflop", decision = "raise 5")  # Small blind
        #self.players[1].action(0, "preflop", decision = "raise 10")
        if self.check_all_players_moved() or len(self.active_players) == 1 and self.preflop:
            self.reset_actions()
            self.preflop = False
            self.state_check("preflop")
        if self.check_all_players_moved() or len(self.active_players) == 1 and self.flop:
            self._deal_community_cards(3)
            self.reset_actions()
            self.flop = False
            self.state_check("flop")
        if self.check_all_players_moved() or len(self.active_players) == 1 and self.turn:
            self._deal_community_cards(1)
            self.reset_actions()
            self.turn = False
            self.state_check("turn")
        if self.check_all_players_moved() or len(self.active_players) == 1 and self.river:
            self._deal_community_cards(1)
            self.reset_actions()
            self.river = False
            self.state_check("river")
        if self.check_all_players_moved() or len(self.active_players) == 1 and not (self.preflop or self.flop or self.turn or self.river): ##
            self.determine_winner()
            self.reset_actions()
            self.pot = 0
            self.state_check("end")
            return True
        return False
        

    def play_game(self, num_rounds: int = 3):
        self.players = deque(player for player in self.players if player.stack > 0)
        for _ in range(num_rounds):
            self.reset_actions()
            
            self.preflop, self.flop, self.turn, self.river = (True, True, True, True)
            while not self.gamelogic():
                decision = ["Fold", 0]
                round = "flop"
                self.players[0].action(self.current_bet, round, decision)
                self.nextplayer(decision , self.players)
            self.players.append(self.players.popleft())

if __name__ == "__main__":
    game = PokerGame(["Alice", "Bob", "Charlie"])
    game.play_game()