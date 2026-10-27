#!/usr/bin/env python3
"""
Minesweeper adapter (single player, hidden information; see general_move_patterns.py)

The state is a MinesweeperGame, which holds the hidden mines. Anything that
chooses moves must use observe(); apply() is the environment carrying out a
move, not something a player may look ahead with.
"""

from grid_vocabulary import Grid

from .minesweeper_game import HIDDEN, MinesweeperGame

PLAYER = 'player'
NATURE = 'nature'      # the environment: wins when a mine is revealed, never moves
GRID_HIDDEN = 9        # value of a hidden cell in to_grid (numbers are 0-8)


class MinesweeperAdapter:
    players = 1
    hidden_information = True

    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10):
        self.rows, self.cols, self.mines = rows, cols, mines

    def new_game(self, seed=None) -> MinesweeperGame:
        return MinesweeperGame(self.rows, self.cols, self.mines, seed)

    def legal_moves(self, game, color):
        return game.legal_moves()

    def apply(self, game, move):
        after = game.copy()
        after.reveal(move)
        return after

    def advantage(self, game, color):
        """Share of the safe cells revealed (observable progress toward the win)"""
        return len(game.revealed) / (game.rows * game.cols - game.n_mines)

    def observe(self, game, color):
        """The player's view: HIDDEN (-1) or the number shown, per cell"""
        return game.observation()

    def final_view(self, game, color):
        """After the game only: the view with the mines shown (MINE, -2), as the game shows at the end"""
        return game.final_view()

    def grid_values(self, color):
        """Values to_grid uses for revealed cells (the numbers)"""
        return set(range(9))

    def to_grid(self, game):
        """The observation as a square Grid: revealed cells 0-8, hidden cells GRID_HIDDEN"""
        return Grid([[GRID_HIDDEN if v == HIDDEN else v for v in row] for row in game.observation()], 'square')

    def is_over(self, game):
        return game.is_over()

    def winner(self, game):
        if game.won():
            return PLAYER
        if game.lost():
            return NATURE
        return None

    def to_move(self, game):
        return PLAYER

    def opponent(self, color):
        return NATURE if color == PLAYER else PLAYER
