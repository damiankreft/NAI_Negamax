""" This is the example featured in section 'A quick example' of the docs """

from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player


class Battleships(TwoPlayerGame):
    """Each player has one ship with 17 combat power. Goal of the game is to sink opponent's battleship before he sink our ship. 
        In turn, the players use one of two actions: strzal (-2 cp for enemy's ship and -1 cp for ours) or 
        salwa (-4 cp for enemy's ship and -3 cp for ours). The player who stays afloat wins"""

    def __init__(self, players=None):
        self.players = players
        self.b1 = 17
        self.b2 = 17
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        return ["strzal", "salwa"]

    def make_move(self, move):
        """Possible actions to perform in game. 
        "strzal" action subtracts 2 combat power points from ememy's cp pool and 1 cp from player's cp pool
        "salwa" action subtracts 4 combat power points from ememy's cp pool and 3 cp from player's cp pool
        """
        
        if move == "strzal":
            if self.current_player is 1: # shot fired by player b1
                self.b1 -= 1
                self.b2 -= 2
            else: # shot fired by player b2
                self.b2 -= 1
                self.b1 -= 2
        elif "salwa":
            if self.current_player is 1:
                self.b1 -= 3
                self.b2 -= 4
            else:
                self.b2 -= 3
                self.b1 -= 4

    # zrobiony
    def win(self):
        """If opponent ship's combat point pool is <= 0 then player wins
        """
        if self.current_player == 1:
            return self.b2 <= 0
        elif self.current_player == 2:
            return self.b1 <= 0
        
    # zrobiony
    def is_over(self):
        """Geme ends when combat power of one of the ships is <= 0"""
        return self.b1 <= 0 or self.b2 <= 0  # game stops when someone wins.

    # zrobiony
    def scoring(self):
        return 100 if self.win() else 0

    # todo wypisywac ile hp maja statki
    def show(self):
        """After each turn function prints actual value of combat power for both ships"""
        print("AI's battleship: %d, player's battleship: %d " % (self.b1, self.b2))

ai = Negamax(10) # The AI will think 10 moves in advance
game = Battleships( [ AI_Player(ai), Human_Player() ] )
history = game.play()