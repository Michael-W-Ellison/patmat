#!/usr/bin/env python3
"""Othello adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .othello_board import Color
from .othello_game import OthelloGame
from .othello_scorer import OthelloScorer


class OthelloAdapter:
    """State is an OthelloGame; color is Color.BLACK / Color.WHITE"""

    def __init__(self):
        self.scorer = OthelloScorer()

    def legal_moves(self, game, color):
        # A side without a placement gets a single pass move (the game's own rule)
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = OthelloGame(game.board.copy())
        after.make_move(move)
        return after

    def advantage(self, game, color):
        """Disc difference (scorer's material = disc count)"""
        return (self.scorer.calculate_material(game.board, color)
                - self.scorer.calculate_material(game.board, self.opponent(color)))

    def grid_values(self, color):
        """Values that to_grid uses for color's pieces"""
        return {1} if color == Color.BLACK else {2}

    def to_grid(self, game):
        """
        8x8 square Grid in the board's own coordinates (row 0 is the top,
        rank 8 in the board's notation). Empty 0; black 1, white 2.
        """
        return Grid([[0 if d is None else (1 if d.color == Color.BLACK else 2) for d in row]
                     for row in game.board.board], 'square')

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        # The board switches turn after every move, including a pass
        return game.board.turn

    def opponent(self, color):
        return Color.WHITE if color == Color.BLACK else Color.BLACK
