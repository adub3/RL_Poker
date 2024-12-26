#should mostly be working

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

    def __repr__(self):
        return f"Player(name={self.name})"
    def bet(self, amount: int):
        if amount > self.stack:
            raise ValueError("Not enough chips to bet")
        self.stack -= amount
        self.current_bet += amount
        return amount

    def receive_card(self, card):
        self.hand = card

    def action(self, bet):
        self.current_bet = self.current_bet + int(input(f"{self.name} how much do you want to call?")) #variable\
        print(bet)
        print(self.current_bet)
        if self.current_bet == bet:
            print("call")
            return ["Call", self.current_bet]
        
        if self.current_bet > bet:
            return ["Raise", self.current_bet]
        
        else:
            self.folded = True
            print(self.name, " folded")
            return ["Fold", 0]
    
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

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

        for player in winners:
            print (player.name)
            player.stack = player.stack + self.pot/len(winners)
            self.pot = 0

        
    def _betting_round(self, order, currentbet):
        active_players = [player for player in self.players if not player.folded]
        if order > 0:
            for i in range(order):
                active_players.append(active_players.pop(0)) #shifting order
        
        print(active_players)
        currentbet = int(active_players[-1].current_bet)
        index = 0 + order #  add order so doesnt mess up order
        for player in active_players:
            print(player.name ,player.current_bet)
        for player in active_players:
            if player == active_players[-1]:
                return
            index = index + 1   #index to know how many to shift when raise
            decision = player.action(currentbet)
            if decision[0] == "Raise":
                print(f"Raised to {decision[1]}")
                print("LEVEL REC")
                self._betting_round(index, decision[1]) #if raise then shift by index
                print("Level exit")
                for player in active_players: #this is so scuffed
                    self.pot = self.pot + player.current_bet
                    player.stack = player.stack - player.current_bet
                    print(self.pot)
                    player.current_bet = 0
                return
        
                

    def state_check(self, round):
        """Display the current state of all players and the pot."""
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
        self._betting_round(2,0) #order diff
        self.state_check("preflop")
        self._deal_community_cards(3)
        self._betting_round(0,0) #
        self.state_check("flop")
        self._deal_community_cards(1)
        self._betting_round(0,0) #
        self.state_check("river")
        self._deal_community_cards(1)
        self._betting_round(0,0) # 
        self.state_check("turn")
        self.determine_winner()
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
