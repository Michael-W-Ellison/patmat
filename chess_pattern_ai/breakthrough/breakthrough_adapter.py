#!/usr/bin/env python3
"""Breakthrough adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .breakthrough_board import BreakthroughBoard, Color, Piece
from .breakthrough_game import BreakthroughGame
from .breakthrough_scorer import BreakthroughScorer
from grid_vocabulary import Grid


class BreakthroughAdapter:
    """State is a BreakthroughGame; color is Color.WHITE / Color.BLACK"""

    def __init__(self):
        self.scorer = BreakthroughScorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = BreakthroughGame(self._copy_board(game.board, move))
        after.make_move(move)
        return after

    @staticmethod
    def _copy_board(board, move):
        """
        Same position as BreakthroughBoard.copy() (the grid is rebuilt from the
        piece set), but cheaper: pieces are shared with the original except the
        one that moves, which is the only piece make_move changes; the move
        history starts empty (nothing reads it, make_move only appends to it)
        """
        new_board = BreakthroughBoard.__new__(BreakthroughBoard)
        grid = [[None] * 8 for _ in range(8)]
        for piece in board.pieces:
            row, col = piece.position
            grid[row][col] = piece
        new_board.board = grid
        new_board.pieces = set(board.pieces)
        new_board.turn = board.turn
        new_board.move_history = []
        row, col = move.from_pos
        piece = grid[row][col]
        if piece is not None:
            moving = Piece(piece.color, piece.position)
            grid[row][col] = moving
            new_board.pieces.discard(piece)
            new_board.pieces.add(moving)
        return new_board

    def to_grid(self, game) -> Grid:
        """
        The board as a grid_vocabulary Grid (8x8, topology 'square'); cell
        (r, c) is board.board[r][c], row 0 = the board's row 0 (the top)

        Values: 0 = empty, 1 = Color.WHITE piece, 2 = Color.BLACK piece.
        White starts on rows 0-1 and moves toward row 7 (its goal side is
        'bottom'); Black starts on rows 6-7 and moves toward row 0 (goal side
        'top'). The game is not changed.
        """
        code = {Color.WHITE: 1, Color.BLACK: 2}
        return Grid([[0 if piece is None else code[piece.color] for piece in row]
                     for row in game.board.board], topology='square')

    def advantage(self, game, color):
        """Scorer's position score (material + advancement), color minus opponent"""
        return (self.scorer.calculate_position_score(game.board, color)
                - self.scorer.calculate_position_score(game.board, self.opponent(color)))

    @staticmethod
    def _has_moves(game):
        """Whether the side to move has a legal move (stops at the first one found)"""
        return any(game._get_piece_moves(piece) for piece in game.board.get_pieces(game.board.turn))

    def is_over(self, game):
        """Same as BreakthroughGame.is_game_over(), without generating every move"""
        return game.board.is_game_over() or not self._has_moves(game)

    def winner(self, game):
        """Same as BreakthroughGame.get_winner()"""
        if game.board.is_game_over():
            return game.board.get_winner()
        if not self._has_moves(game):
            return self.opponent(game.board.turn)
        return None

    def to_move(self, game):
        return game.board.turn

    def opponent(self, color):
        return Color.BLACK if color == Color.WHITE else Color.WHITE
