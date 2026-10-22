#!/usr/bin/env python3
"""Nine Men's Morris adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .morris_board import Color
from .morris_scorer import MorrisScorer


class MorrisAdapter:
    """
    State is a MorrisGame; color is Color.WHITE / Color.BLACK.

    Moves are the game's own legal moves for the current phase: placements,
    adjacent moves or flying moves, and (as integers) the removal of an
    opponent piece after a mill. Forming a mill keeps the turn for that
    removal, so such moves are seen as giving an extra turn.
    """

    def __init__(self):
        self.scorer = MorrisScorer()

    def legal_moves(self, game, color):
        # MorrisGame generates the moves of the player to move (color is that player)
        return game.get_legal_moves()

    def apply(self, game, move):
        # MorrisGame.make_move works on a copy and returns the new state
        return game.make_move(move)

    def advantage(self, game, color):
        """Scorer's evaluation (material, mills, potential mills, phase bonuses), color minus opponent"""
        return self.scorer.score_game(game, color)

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.current_player

    def opponent(self, color):
        return color.opposite()
