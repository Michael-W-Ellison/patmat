#!/usr/bin/env python3
"""Chess adapter for game-independent pattern learning (see general_move_patterns.py)"""

import chess

from game_scorer import GameScorer

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


class ChessAdapter:
    """State is a chess.Board; color is chess.WHITE / chess.BLACK"""

    def __init__(self):
        values = GameScorer.PIECE_VALUES
        self.values = {pt: values[chess.piece_symbol(pt).upper()] for pt in _PIECE_TYPES}

    def legal_moves(self, board, color):
        return list(board.legal_moves) if board.turn == color else []

    def apply(self, board, move):
        after = board.copy()
        after.push(move)
        return after

    # In-place variants: copying a board copies its whole move stack
    def push(self, board, move):
        board.push(move)

    def pop(self, board):
        board.pop()

    def advantage(self, board, color):
        """Material difference (scorer's piece values)"""
        total = 0.0
        for pt, value in self.values.items():
            total += value * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color)))
        return total

    def is_over(self, board):
        return board.is_game_over()

    def winner(self, board):
        outcome = board.outcome()
        return outcome.winner if outcome else None

    def to_move(self, board):
        return board.turn

    def opponent(self, color):
        return not color
