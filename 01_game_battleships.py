""" This is the example featured in section 'A quick example' of the docs """

from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player


class Battleships(TwoPlayerGame):
    """In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses."""

    # chyba zrobiony
    def __init__(self, players=None):
        self.players = players
        self.b1 = 17
        self.b2 = 17
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        return ["strzal", "salwa"]

    def make_move(self, move):
        if move == "strzal": # y 1, o 2
            if self.current_player is 1: # shot fired by player b1
                self.b1 -= 1
                self.b2 -= 2
            else: # shot fired by player b2
                self.b2 -= 1
                self.b1 -= 2
        elif "salwa": # y 3, o 4
            if self.current_player is 1:
                self.b1 -= 3
                self.b2 -= 4
            else:
                self.b2 -= 3
                self.b1 -= 4

    # zrobiony
    def win(self):
        if self.current_player == 1:
            return self.b2 <= 0
        elif self.current_player == 2:
            return self.b1 <= 0
        
    # zrobiony
    def is_over(self):
        return self.b1 <= 0 or self.b2 <= 0  # game stops when someone wins.

    # zrobiony
    def scoring(self):
        # return 100 if self.b1 <= 0 else 0
        return 100 if self.win() else 0

    # todo wypisywac ile hp maja statki
    def show(self):
        print("AI's battleship: %d, player's battleship: %d " % (self.b1, self.b2))

ai = Negamax(10) # The AI will think 10 moves in advance
game = Battleships( [ AI_Player(ai), Human_Player() ] )
history = game.play()