#!/usr/bin/env python3
"""Arimaa adapter for game-independent pattern learning (see general_move_patterns.py)

A "move" here is a single step, as the trainer selects steps one at a time.
A turn has up to 4 steps: while the same player continues the turn,
to_move() returns that player, so general_move_patterns classifies the step
as extra_turn; after the turn's last step the opponent is to move.
"""

from grid_vocabulary import Grid

from .arimaa_board import ArimaaBoard, Color
from .arimaa_scorer import ArimaaScorer


class ArimaaAdapter:
    """State is an ArimaaGame; color is Color.GOLD / Color.SILVER"""

    def __init__(self):
        self.scorer = ArimaaScorer()
        self._piece_values = {}  # (type, color, square) -> its term in scorer._evaluate_color

    def legal_moves(self, game, color):
        if game.current_player != color:
            game = game.copy()
            game.current_player = color
            game.steps_taken = 0
        return game.get_legal_steps()

    def apply(self, game, step):
        # ArimaaGame.make_step works on a copy and returns the new game
        return game.make_step(step)

    def advantage(self, game, color):
        """Scorer's value (material, advancement, trap control) of color minus the opponent's"""
        return (self._evaluate_color(game.board, color)
                - self._evaluate_color(game.board, self.opponent(color)))

    def _evaluate_color(self, board, color):
        """
        Same value as ArimaaScorer._evaluate_color, which (unless a rabbit is
        on its goal) is a sum of one term per piece depending only on the
        piece's type, color and square. Each piece's term is taken once from
        the scorer itself, on a board holding only that piece, and cached:
        the scorer's per-piece trap-adjacency loop made feature extraction
        about ten times slower than the rest of step selection.
        """
        if board.has_rabbit_on_goal(color):
            return self.scorer.RABBIT_ON_GOAL
        values = self._piece_values
        total = 0.0
        for piece in board.get_pieces(color):
            # Keyed by identity of the (singleton) enum members: hashing Piece
            # or the enums themselves goes through slow Python-level __hash__
            key = (id(piece.type), id(piece.color), piece.position)
            value = values.get(key)
            if value is None:
                value = values[key] = self._single_piece_value(piece)
            total += value
        return total

    def _single_piece_value(self, piece):
        lone = ArimaaBoard.__new__(ArimaaBoard)
        lone.board = [[None] * 8 for _ in range(8)]
        lone.board[piece.position[0]][piece.position[1]] = piece
        lone.pieces_gold = {piece} if piece.color == Color.GOLD else set()
        lone.pieces_silver = set() if piece.color == Color.GOLD else {piece}
        return self.scorer._evaluate_color(lone, piece.color)

    def grid_values(self, color):
        """Values that to_grid uses for color's pieces"""
        return {1, 2, 3, 4, 5, 6} if color == Color.GOLD else {7, 8, 9, 10, 11, 12}

    def to_grid(self, game):
        """
        8x8 square Grid in the board's own coordinates (row 0 is gold's
        rabbit row; gold's goal is the bottom row 7, silver's the top row 0).
        Empty 0; gold piece = its strength (rabbit 1, cat 2, dog 3, horse 4,
        camel 5, elephant 6); silver piece = 6 + strength (rabbit 7 ...
        elephant 12). Trap squares are not marked (an empty trap is 0).
        """
        return Grid([[0 if p is None else p.type.strength + (0 if p.color == Color.GOLD else 6)
                      for p in row] for row in game.board.board], 'square')

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        """Player to make the next step (the same player within a turn)"""
        return game.current_player

    def opponent(self, color):
        return color.opposite()
