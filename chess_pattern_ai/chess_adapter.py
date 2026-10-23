#!/usr/bin/env python3
"""Chess adapter for game-independent pattern learning (see general_move_patterns.py)"""

import chess

from game_scorer import GameScorer
from grid_vocabulary import Grid

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


class ChessAdapter:
    """State is a chess.Board; color is chess.WHITE / chess.BLACK"""

    # Win-condition weight, in the scorer's material units (pawn = 10)
    CHECK_VALUE = 5.0

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
        """
        Material difference (scorer's piece values) plus progress toward the
        winning condition, checkmate (the side to move is in check and has no
        legal move): giving check is worth CHECK_VALUE. A checkmate itself is
        valued as a win by general_move_patterns.evaluate().

        Counting the kings' available moves as a further win-condition term
        lowered the win rate in testing (60-68% against 74% over 600 games),
        so it is not included.
        """
        total = 0.0
        for pt, value in self.values.items():
            total += value * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color)))
        if board.is_check():
            # Only the side to move can be in check
            total += self.CHECK_VALUE if board.turn != color else -self.CHECK_VALUE
        return total

    def to_grid(self, board):
        """
        8x8 square Grid: row = rank index (row 0 is rank 1, White's back
        rank), column = file (0 = a-file). Empty 0; white pawn, knight,
        bishop, rook, queen, king 1..6; black pawn..king 7..12.
        """
        cells = [[0] * 8 for _ in range(8)]
        for sq, piece in board.piece_map().items():
            cells[chess.square_rank(sq)][chess.square_file(sq)] = (
                piece.piece_type + (0 if piece.color == chess.WHITE else 6))
        return Grid(cells, 'square')

    def is_over(self, board):
        return board.is_game_over()

    def winner(self, board):
        outcome = board.outcome()
        return outcome.winner if outcome else None

    def to_move(self, board):
        return board.turn

    def opponent(self, color):
        return not color
