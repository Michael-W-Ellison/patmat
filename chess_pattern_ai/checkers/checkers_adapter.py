#!/usr/bin/env python3
"""Checkers adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .checkers_board import Color, PieceType
from .checkers_game import CheckersGame
from .checkers_scorer import CheckersScorer


class CheckersAdapter:
    """State is a CheckersGame; color is Color.RED / Color.BLACK"""

    def __init__(self):
        self.scorer = CheckersScorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = CheckersGame(game.board.copy())
        after.make_move(move)
        return after

    def advantage(self, game, color):
        """
        Material difference (scorer's piece values). The winning condition,
        leaving the opponent with no pieces or no legal move, ends the game
        and is valued as a win by general_move_patterns.evaluate(); the
        material count measures progress toward the "no pieces" part.

        A legal-move-count (mobility) term was tested as further progress
        toward the "no legal move" part and left out: it lowered the win rate
        (94.8-94.9% against 96.1% over 4 seeds x 300 games).
        """
        opp = self.opponent(color)
        return (self.scorer.calculate_material(game.board, color)
                - self.scorer.calculate_material(game.board, opp))

    def to_grid(self, game):
        """
        8x8 square Grid in the board's own coordinates (row 0 is the top,
        Black's starting side). Empty 0; red man 1, red king 2, black man 3,
        black king 4.
        """
        cells = [[0] * 8 for _ in range(8)]
        for piece in game.board.pieces:
            r, c = piece.position
            cells[r][c] = ((1 if piece.color == Color.RED else 3)
                           + (piece.type == PieceType.KING))
        return Grid(cells, 'square')

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.board.turn

    def opponent(self, color):
        return Color.BLACK if color == Color.RED else Color.RED
