import random
from enum import Enum, auto
from typing import List, Optional,Tuple
from collections import Counter

class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()
    SPADES = auto()

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    
    def __repr__(self):
        return f"{self.rank.name} of {self.suit.name}"
    
    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank



class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        if not self.cards:
            raise ValueError("No cards left in the deck")
        return self.cards.pop()

class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

class Player:
    def __init__(self, name: str, initial_stack: int):
        self.name = name
        self.hand: List[Card] = []
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
    
    def receive_card(self, card: Card):
        self.hand.append(card)
    
    def clear_hand(self):
        self.hand = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

class PokerGame:
    def __init__(self, players: List[str], starting_stack: int = 1000):
        self.deck = Deck()
        self.players = [Player(name, starting_stack) for name in players]
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.button_index = 0
        self.current_player_index = 0
    
    def _deal_initial_cards(self):
        # Deal 2 cards to each player
        for _ in range(2):
            for player in self.players:
                player.receive_card(self.deck.draw())
    
    def _deal_flop(self):
        # Burn a card and deal 3 community cards
        self.deck.draw()  # Burn card
        for _ in range(3):
            self.community_cards.append(self.deck.draw())
    
    def _deal_turn_or_river(self):
        # Burn a card and deal 1 community card
        self.deck.draw()  # Burn card
        self.community_cards.append(self.deck.draw())
    
    def _evaluate_hand(self) -> Tuple[HandRank, List[int]]:
        """Determine the hand's rank and supporting details for comparison."""
        is_flush = len(set(card.suit for card in self.cards)) == 1
        rank_values = [card.value for card in self.cards]
        rank_counts = Counter(rank_values)
        
        # Check for straight
        is_straight = (len(set(rank_values)) == 5 and 
                       max(rank_values) - min(rank_values) == 4)
        
        # Special case for Ace-low straight (A, 2, 3, 4, 5)
        if set(rank_values) == {14, 2, 3, 4, 5}:
            is_straight = True
            rank_values = [5, 4, 3, 2, 1]  # Adjust values for comparison
        
        # Royal and Straight Flush
        if is_flush and is_straight:
            if set(rank_values) == {10, 11, 12, 13, 14}:
                return HandRank.ROYAL_FLUSH, rank_values
            return HandRank.STRAIGHT_FLUSH, rank_values
        
        # Four of a Kind
        if 4 in rank_counts.values():
            four_value = [v for v, count in rank_counts.items() if count == 4][0]
            kicker = [v for v in rank_values if v != four_value][0]
            return HandRank.FOUR_OF_A_KIND, [four_value, kicker]
        
        # Full House
        if set(rank_counts.values()) == {2, 3}:
            three_value = [v for v, count in rank_counts.items() if count == 3][0]
            pair_value = [v for v, count in rank_counts.items() if count == 2][0]
            return HandRank.FULL_HOUSE, [three_value, pair_value]
        
        # Flush
        if is_flush:
            return HandRank.FLUSH, rank_values
        
        # Straight
        if is_straight:
            return HandRank.STRAIGHT, rank_values
        
        # Three of a Kind
        if 3 in rank_counts.values():
            three_value = [v for v, count in rank_counts.items() if count == 3][0]
            kickers = sorted([v for v in rank_values if v != three_value], reverse=True)
            return HandRank.THREE_OF_A_KIND, [three_value] + kickers
        
        # Two Pair
        pairs = [v for v, count in rank_counts.items() if count == 2]
        if len(pairs) == 2:
            pairs.sort(reverse=True)
            kicker = [v for v in rank_values if v not in pairs][0]
            return HandRank.TWO_PAIR, pairs + [kicker]
        
        # One Pair
        if 2 in rank_counts.values():
            pair_value = [v for v, count in rank_counts.items() if count == 2][0]
            kickers = sorted([v for v in rank_values if v != pair_value], reverse=True)
            return HandRank.PAIR, [pair_value] + kickers
        
        # High Card
        return HandRank.HIGH_CARD, rank_values
    
    def _evaluate_hand(self, cards):
        """
        Evaluate the best 5-card hand from a set of cards (player's hand + community cards)
        
        Args:
        cards (list): A list of Card objects to evaluate
        
        Returns:
        tuple: (hand_rank, hand_value) where hand_rank is the HandRank enum 
            and hand_value is a list of card values for tie-breaking
        """
        # Convert cards to PokerHand objects systematically
        best_hand = None
        for combo in itertools.combinations(cards, 5):
            hand = PokerHand(list(combo))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        
        return best_hand.rank, best_hand.rank_details
    def determine_winner(self) -> List[Player]:
        # Simplified winner determination
        best_hand_rank = HandRank.HIGH_CARD
        winners = []
        best_hand_value = 0
        
        for player in self.players:
            if not player.folded:
                full_hand = player.hand + self.community_cards
                hand_rank, hand_value = self._evaluate_hand(full_hand)
                if hand_rank > best_hand_rank or \
                   (hand_rank == best_hand_rank and hand_value > best_hand_value):
                    best_hand_rank = hand_rank
                    best_hand_value = hand_value
                    winners = [player]
                elif hand_rank == best_hand_rank and hand_value == best_hand_value:
                    winners.append(player)
        
        return winners
    
    def _print_game_state(self):
        """Print the current game state, including player hands and community cards."""
        print("\nCommunity Cards:")
        print(" ".join(str(card) for card in self.community_cards))
        print("\nPlayer Hands:")
        for player in self.players:
            full_hand = player.hand + self.community_cards
            ##
            print(full_hand[1].rank, full_hand[1].suit)
            if not player.folded:
                print(f"{player.name}: {' '.join(str(card) for card in player.hand)} (Stack: {player.stack})")
                print(player.hand)

    def play_round(self):
        # Reset for a new round
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        
        # Reset players
        for player in self.players:
            player.clear_hand()
        
        # Deal initial cards
        self._deal_initial_cards()
        
        # Print initial hands
        print("\n--- New Round ---")
        self._print_game_state()
        
        # Betting rounds
        # Preflop betting
        self._preflop_betting()
        
        # Flop
        self._deal_flop()
        print("\nFlop:")
        self._print_game_state()
        self._betting_round()
        
        # Turn
        self._deal_turn_or_river()
        print("\nTurn:")
        self._print_game_state()
        self._betting_round()
        
        # River
        self._deal_turn_or_river()
        print("\nRiver:")
        self._print_game_state()
        self._betting_round()
        
        # Showdown
        print("\nShowdown:")
        winners = self.determine_winner()
        
        # Distribute pot
        self._distribute_pot(winners)
        
        # Highlight winners
        print("\nWinners:")
        for winner in winners:
            print(f"{winner.name}: {' '.join(str(card) for card in winner.hand)} won {self.pot} chips")
        
        return winners
    
    def _preflop_betting(self):
        # Implement preflop betting logic with small and big blinds
        # This is a simplified version
        small_blind_index = (self.button_index + 1) % len(self.players)
        big_blind_index = (self.button_index + 2) % len(self.players)
        
        # Ensure players have enough chips for blinds
        small_blind = min(10, self.players[small_blind_index].stack)
        big_blind = min(20, self.players[big_blind_index].stack)
        
        self.players[small_blind_index].bet(small_blind)
        self.players[big_blind_index].bet(big_blind)
        
        self.pot += small_blind + big_blind
        self.current_bet = big_blind
        
        self._betting_round()
    
    def _betting_round(self):
        # Simplified betting round
        # In a real implementation, this would be much more complex
        for player in self.players:
            if not player.folded:
                # Simple decision logic - replace with more sophisticated AI
                if random.random() < 0.5:
                    # Bet or call
                    bet_amount = min(self.current_bet, player.stack)
                    if bet_amount > 0:
                        player.bet(bet_amount)
                        self.pot += bet_amount
                else:
                    # Fold
                    player.folded = True
    
    def _distribute_pot(self, winners: List[Player]):
        # Handle case where there are no winners (all fold except one)
        if not winners:
            # Find the last non-folded player and give them the pot
            non_folded_players = [p for p in self.players if not p.folded]
            if non_folded_players:
                winners = non_folded_players
        
        # Distribute pot equally among winners
        if winners:
            pot_split = self.pot // len(winners)
            for winner in winners:
                winner.stack += pot_split
        
        self.pot = 0
    
    def play_game(self, num_rounds: int = 10):
        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}")
            winners = self.play_round()
            print(f"Round winners: {[winner.name for winner in winners]}")
        
        # Print final standings
        print("\nFinal Standings:")
        for player in self.players:
            print(f"{player.name}: {player.stack} chips")

# Example usage
def main():
    game = PokerGame(["Alice", "Bob", "Charlie"], starting_stack=1000)
    game.play_game()

if __name__ == "__main__":
    main()