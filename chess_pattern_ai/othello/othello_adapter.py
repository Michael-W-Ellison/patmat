#!/usr/bin/env python3
"""Othello adapter for game-independent pattern learning (see general_move_patterns.py)"""

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

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        # The board switches turn after every move, including a pass
        return game.board.turn

    def opponent(self, color):
        return Color.WHITE if color == Color.BLACK else Color.BLACK
