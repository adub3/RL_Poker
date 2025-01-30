import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Any
from treys import Card, Evaluator, Deck

class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, num_players=3, starting_stack=1000, big_blind=10):
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.big_blind = big_blind
        self.evaluator = Evaluator()
        
        # Action space: fold (0), call (1), raise (2), all-in (3)
        # For raises, we'll handle the amount in the step function
        self.action_space = spaces.Discrete(4)
        
        # Observation space:
        # - Player hands (2 cards per player, encoded as integers 0-51)
        # - Community cards (5 cards, encoded as integers 0-51)
        # - Player stacks (float values)
        # - Current bets (float values)
        # - Pot size (float value)
        # - Player folded status (boolean)
        # - Current round (0-3 for preflop, flop, turn, river)
        self.observation_space = spaces.Dict({
            'hand': spaces.Box(low=0, high=51, shape=(2,), dtype=np.int32),
            'community_cards': spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
            'player_stacks': spaces.Box(low=0, high=float('inf'), shape=(num_players,), dtype=np.float32),
            'current_bets': spaces.Box(low=0, high=float('inf'), shape=(num_players,), dtype=np.float32),
            'pot': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'player_folded': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.bool_),
            'current_round': spaces.Discrete(4),
            'current_player': spaces.Discrete(num_players)
        })
        
        self.reset()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Convert current game state to observation
        community_cards_arr = np.full(5, -1)  # Fill with -1 for unrevealed cards
        if self.community_cards:
            for i, card in enumerate(self.community_cards):
                community_cards_arr[i] = card
        
        return {
            'hand': np.array([self.players[self.current_player].hand[0], 
                            self.players[self.current_player].hand[1]]),
            'community_cards': community_cards_arr,
            'player_stacks': np.array([p.stack for p in self.players]),
            'current_bets': np.array([p.current_bet for p in self.players]),
            'pot': np.array([self.pot]),
            'player_folded': np.array([p.folded for p in self.players]),
            'current_round': self.current_round,
            'current_player': self.current_player
        }
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize game state
        self.deck = Deck()
        self.players = [Player(f"Player_{i}", self.starting_stack) for i in range(self.num_players)]
        self.community_cards = []
        self.pot = 0
        self.current_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.current_player = 0
        self.current_bet = self.big_blind
        
        # Deal initial cards
        for player in self.players:
            player.hand = self.deck.draw(2)
        
        # Set up blinds
        self.players[0].current_bet = self.big_blind / 2  # Small blind
        self.players[1].current_bet = self.big_blind      # Big blind
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0
        
        current_player = self.players[self.current_player]
        
        # Handle action
        if action == 0:  # Fold
            current_player.folded = True
            reward = -current_player.current_bet
        elif action == 1:  # Call
            call_amount = self.current_bet - current_player.current_bet
            if call_amount >= current_player.stack:
                current_player.current_bet += current_player.stack
                current_player.stack = 0
                current_player.all_in = True
            else:
                current_player.current_bet = self.current_bet
                current_player.stack -= call_amount
        elif action == 2:  # Raise
            raise_amount = min(self.current_bet * 2, current_player.stack)
            current_player.stack -= raise_amount
            current_player.current_bet += raise_amount
            self.current_bet = current_player.current_bet
        elif action == 3:  # All-in
            current_player.current_bet += current_player.stack
            self.current_bet = max(self.current_bet, current_player.current_bet)
            current_player.stack = 0
            current_player.all_in = True
        
        # Move to next player or next round
        active_players = [p for p in self.players if not (p.folded or p.all_in)]
        if len(active_players) <= 1:
            terminated = True
            reward = self._handle_round_end()
        else:
            self.current_player = (self.current_player + 1) % self.num_players
            while self.players[self.current_player].folded or self.players[self.current_player].all_in:
                self.current_player = (self.current_player + 1) % self.num_players
        
        # Check if round is complete
        if all(p.current_bet == self.current_bet or p.folded or p.all_in for p in self.players):
            if self.current_round < 3:
                self._advance_round()
            else:
                terminated = True
                reward = self._handle_round_end()
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _advance_round(self):
        self.current_round += 1
        if self.current_round == 1:  # Flop
            self.community_cards.extend(self.deck.draw(3))
        elif self.current_round in [2, 3]:  # Turn and River
            self.community_cards.extend(self.deck.draw(1))
        
        self.pot += sum(p.current_bet for p in self.players)
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
    
    def _handle_round_end(self) -> float:
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            return self.pot if winner == self.players[self.current_player] else -self.pot
        
        # Evaluate hands
        scores = [(p, self.evaluator.evaluate(self.community_cards, p.hand)) 
                 for p in active_players]
        winner = min(scores, key=lambda x: x[1])[0]
        winner.stack += self.pot
        
        return self.pot if winner == self.players[self.current_player] else -self.pot
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nRound: {['Preflop', 'Flop', 'Turn', 'River'][self.current_round]}")
            print(f"Pot: {self.pot}")
            print(f"Community cards: {[Card.int_to_str(c) for c in self.community_cards]}")
            for i, player in enumerate(self.players):
                status = "Folded" if player.folded else "All-in" if player.all_in else "Active"
                print(f"Player {i}: Stack={player.stack}, Bet={player.current_bet}, "
                      f"Status={status}, Hand={[Card.int_to_str(c) for c in player.hand]}")