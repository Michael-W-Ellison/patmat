#!/usr/bin/env python3
"""Gomoku adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .gomoku_board import Color, GomokuBoard
from .gomoku_game import GomokuGame
from .gomoku_scorer import GomokuScorer


class GomokuAdapter:
    """State is a GomokuGame; color is Color.BLACK / Color.WHITE"""

    def __init__(self):
        self.scorer = GomokuScorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def _copy_board(self, board):
        """Same result as board.copy(), without building an empty grid first"""
        new = GomokuBoard.__new__(GomokuBoard)
        new.size = board.size
        new.board = [row[:] for row in board.board]   # Stones are never mutated
        new.stones = set(board.stones)
        new.turn = board.turn
        new.move_history = board.move_history.copy()
        return new

    def apply(self, game, move):
        after = GomokuGame(self._copy_board(game.board))
        after.make_move(move)
        after._general_over = None   # memo: only on states made here, never changed again
        return after

    def advantage(self, game, color):
        """Weighted threat difference (scorer's open/blocked 2-3-4 line weights)"""
        return (self.scorer.count_threats(game.board, color)
                - self.scorer.count_threats(game.board, self.opponent(color)))

    def to_grid(self, game):
        """
        size x size square Grid in the board's own coordinates (row 0 is the
        top). Empty 0; black 1, white 2.
        """
        return Grid([[0 if s is None else (1 if s.color == Color.BLACK else 2) for s in row]
                     for row in game.board.board], 'square')

    def _over_and_winner(self, game):
        memo = getattr(game, '_general_over', False)
        if memo:
            return memo
        # Same result as game.is_game_over() (five in a row or full board), without
        # generating every legal move just to see whether one exists
        result = (game.board.is_game_over(), None)
        if result[0]:
            result = (True, game.board.get_winner())
        if memo is None:
            game._general_over = result
        return result

    def is_over(self, game):
        return self._over_and_winner(game)[0]

    def winner(self, game):
        return self._over_and_winner(game)[1]

    def to_move(self, game):
        return game.board.turn

    def opponent(self, color):
        return Color.WHITE if color == Color.BLACK else Color.BLACK
