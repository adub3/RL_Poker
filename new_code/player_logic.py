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