"""
Othello/Reversi Pattern Recognition AI

Observation-based learning for Othello/Reversi game
"""

from .othello_board import OthelloBoard, Disc, Move, Color
from .othello_game import OthelloGame
from .othello_scorer import OthelloScorer

__all__ = [
    'OthelloBoard',
    'Disc',
    'Move',
    'Color',
    'OthelloGame',
    'OthelloScorer',
]
