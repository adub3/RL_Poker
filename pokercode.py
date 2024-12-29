#should mostly be working
#still issues with negativity
#all in is super confusing
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
        self.bets = {}
        self.folded = False
        self.all_in = False
        self.raised = False

    def __repr__(self):
        return f"Player(name={self.name})"

    def receive_card(self, card):
        self.hand = card

    def action(self, bet, round):
        self.raised = False
        while True:
            try:
                # Prompt the player for input
                decision = input(f"{self.name}, your stack: {self.stack}. Current bet: {bet}. Raised: {self.raised}Enter 'fold', 'call', 'raise', or 'all-in': ").strip().lower()
                
                if decision == "fold":
                    self.folded = True
                    print(f"{self.name} folded.")
                    self.bets[round] = 0
                    return ["Fold", 0]
                
                elif decision == "call":
                    call_amount = min(self.stack, bet - self.current_bet)
                    self.current_bet += call_amount
                    self.bets[round] = self.current_bet
                    print(f"{self.name} called with {call_amount}.")
                    return ["Call", self.current_bet]
                
                elif decision == "all-in":
                    all_in_amount = self.stack
                    self.current_bet += all_in_amount
                    self.all_in = True
                    self.bets[round] = self.current_bet
                    print(f"{self.name} went all-in with {all_in_amount}.")
                    return ["All In", self.current_bet]
                
                elif decision == "raise":
                    raise_amount = int(input(f"Enter raise amount (must be greater than the current bet {bet}): "))
                    total_bet = self.current_bet + raise_amount
                    if total_bet < bet:
                        print(f"Raise amount (must be greater than the current bet {bet}): ")
                    elif total_bet > self.stack:
                        print(f"Invalid raise. You can only raise up to your stack ({self.stack}).")
                    elif total_bet == self.stack:
                        self.all_in = True
                        self.bets[round] = self.current_bet
                        print(f"{self.name} went all-in with {self.stack}.")
                        return ["All In", self.current_bet]
                    else:
                        self.current_bet += raise_amount
                        self.bets[round] = self.current_bet
                        print(f"{self.name} raised to {self.current_bet}.")
                        self.raised = True
                        return ["Raise", self.current_bet]
                
                else:
                    print("Invalid input. Please enter 'fold', 'call', 'raise', or 'all-in'.")
            except ValueError:
                print("Invalid input. Please enter a valid number for raise.")

    
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.bets = {}
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

        
    def _betting_round(self, order, round, currentbet = 0, Raised = False): #issue with raising fucks up the order super bad... especially when someone folds... current soln is not working where list is the same so the order still works for the shifting, but doesnt change orders correctly when raised as stack still has same person
        active_playerscheck = [player for player in self.players if not (player.folded or player.all_in)]
        active_players = self.players
        if order > 0:
            for i in range(order):
                print(active_players[0].name)
                while active_players[0].folded or active_players[0].all_in: #changed...

                    active_players.append(active_players.pop(0))
                    print("shifted ",active_players)
                active_players.append(active_players.pop(0)) #shifting order
                print("shifted ", active_players)
        index = order
        order = 0
        if len(active_players) <= 1:
            if len(active_players) == 1:
                # Last player wins the pot
                winner = active_players[0]
                winner.stack += self.pot
                self.pot = 0
            return

        print(active_playerscheck, "pineapple")
        print(active_players, "turn order")
        current_max_bet = max(player.current_bet for player in active_players)
        currentbet = max(currentbet, current_max_bet)
        index = order #  add order so doesnt mess up order
        for player in active_players:
            print(player.name , player.current_bet)
        for player in active_players:
            index = index + 1   #index to know how many to shift when raise
            if player.raised and Raised or (player.folded or player.all_in): #changed...
                print(player.name, "is out")
                pass
            else:
                #index = index + 1   #index to know how many to shift when raise
                decision = player.action(currentbet,round)
                if decision[0] == "Fold":
                    pass
                    #index = index - order//index + 1 #should reset the order no clue seems to be working
                elif decision[0] == "All In":
                    print('triggered all in')
                    #index = index - order//index + 1 #once all-in, same as folded other than A. if all in is higher acts as raise, else acts as call
                    print(decision[1], decision[1] > currentbet)
                    if decision[1] > currentbet:
                        self._betting_round(index, round, decision[1], True) #if raise then shift by index
                        for player in active_players: #this is so scuffed
                            self.pot = self.pot + player.current_bet
                            player.stack = player.stack - player.current_bet
                            player.current_bet = 0
                        return
                    else:
                        pass
                    print("Player is all-in")
                elif decision[0] == "Raise":
                    self._betting_round(index, round, decision[1], True) #if raise then shift by index
                    for player in active_players: #this is so scuffed
                        self.pot = self.pot + player.current_bet
                        player.stack = player.stack - player.current_bet
                        player.current_bet = 0
                    return
        if not Raised:
            for player in active_players: #this is so scuffed
                print(f"pot added: {player.current_bet}")
                self.pot = self.pot + player.current_bet
                player.stack = player.stack - player.current_bet

                player.current_bet = 0
        for player in active_players:
            player.raised = False
            print(f'{player.name} and {player.raised}')
        index = 0


                

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

        sb = 5
        bb = 10
        self._deal_initial_cards()
        self.players[0].current_bet = sb  # Small blind
        self.players[1].current_bet = bb
        self._betting_round(2, "preflop") #order diff
        self.state_check("preflop")
        self._deal_community_cards(3)
        self._betting_round(1, "flop") #
        self.state_check("flop")
        self._deal_community_cards(1)
        self._betting_round(1, "river") #
        self.state_check("river")
        self._deal_community_cards(1)
        self._betting_round(1, "turn") # 
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