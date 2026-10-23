#!/usr/bin/env python3
"""Connect Four adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .connect4_board import Color, Move, Piece
from .connect4_game import Connect4Game
from .connect4_scorer import Connect4Scorer


class Connect4Adapter:
    """State is a Connect4Game; color is Color.RED / Color.YELLOW"""

    def __init__(self):
        self.scorer = Connect4Scorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = Connect4Game(game.board.copy())
        # The board stores the move's own Piece object; place a fresh one so the
        # new state shares nothing mutable with the caller's move or state
        piece = Piece(move.piece.color, (move.row, move.column))
        after.make_move(Move(column=move.column, row=move.row, piece=piece))
        return after

    def advantage(self, game, color):
        """Threat difference (scorer's count of open lines of three)"""
        return (self.scorer.count_threats(game.board, color)
                - self.scorer.count_threats(game.board, self.opponent(color)))

    def to_grid(self, game):
        """
        6x7 square Grid in the board's own coordinates (row 0 is the top,
        row 5 the bottom where pieces land first). Empty 0; red 1, yellow 2.
        """
        return Grid([[0 if p is None else (1 if p.color == Color.RED else 2) for p in row]
                     for row in game.board.board], 'square')

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.board.turn

    def opponent(self, color):
        return Color.YELLOW if color == Color.RED else Color.RED
