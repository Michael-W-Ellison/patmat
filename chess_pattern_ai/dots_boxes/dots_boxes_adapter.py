#!/usr/bin/env python3
"""Dots and Boxes adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .dots_boxes_board import Color
from .dots_boxes_game import DotsBoxesGame
from .dots_boxes_scorer import DotsBoxesScorer


class DotsBoxesAdapter:
    """State is a DotsBoxesGame; color is Color.RED / Color.BLUE"""

    def __init__(self):
        self.scorer = DotsBoxesScorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = DotsBoxesGame(game.board.copy())
        after.current_player = game.current_player
        after.last_move_completed_box = game.last_move_completed_box
        after.make_move(move)
        return after

    def advantage(self, game, color):
        """Box advantage from the scorer (boxes and 3-edge boxes, color minus opponent)"""
        return self.scorer.calculate_box_advantage(game.board, color)

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        # Completing a box keeps the turn (DotsBoxesGame.make_move does not switch)
        return game.current_player

    def opponent(self, color):
        return Color.BLUE if color == Color.RED else Color.RED
