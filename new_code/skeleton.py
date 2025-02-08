'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import abstractbetting
from ai import load_strategy, MCCFR
from abstraction import abstractbettinge, abstractioncards

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.

        LOAD DICTIONARY
        '''
        self.strategy = load_strategy() # load dictionary from text file
        self.cards = {"Private" : [], "Public" : []}
        self.big_blind = False
        self.chips = 0
        self.state = ""
        self.log = ""

        pass

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        self.chips = game_state.bankroll
        
        self.cards["Private"] = round_state.hands[active]  # your cards
        self.card_string = abstractioncards(self.cards)
        self.big_blind = bool(active)  # True if you are the big blind
        temp = abstractbettinge()
        self.log = ""
        
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        self.cards["Public"] = []
        self.cards["Private"] = []

        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''

        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        #find opp action, add to log
        if(opp_pip > round_state.previous_state.pips[1-active]):
            self.log += 'r'
        else:
            self.log += 'c'

        # call abstraction to find scenario
        context = abstractbettinge(self.log, round_state, active) #change abstractbetting to just extract all the information itself low key
        self.cards["Public"] = board_cards
        self.state = abstractioncards(self.cards) + ' ' + context # create state string for lookup table

        #fold, call, bet, all in
        if self.state in self.strategy:
            ind = random.choices(len(self.strategy[context]), self.strategy[context], 1) # strategy holds the weight. chooses on strategy.
        else:
            ind = random.choices(4, [0.25, 0.25, 0.25, 0.25], 1) #pick at random if no data
        print(ind)

        min_raise, max_raise = round_state.raise_bounds() # the smallest and largest numbers of chips for a legal bet/raise
        
        
        if (ind[0] == 0 and FoldAction in legal_actions):
            return FoldAction
        if (ind[0] == 1 and CheckAction in legal_actions):
            self.log += 'c'
            return CheckAction
        if (ind[0] >= 2 and RaiseAction in legal_actions):
            self.log += 'r'
            if(ind[0] == 2):
                return RaiseAction(min_raise)
            else:
                return RaiseAction(max_raise)
        self.log += 'c'
        return CallAction()


if __name__ == '__main__':

    run_bot(Player(), parse_args())