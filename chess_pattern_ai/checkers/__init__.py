"""
Checkers Pattern Recognition AI

Observation-based learning for American/English Checkers
"""

from .checkers_board import CheckersBoard, Piece, Move, Color, PieceType
from .checkers_game import CheckersGame
from .checkers_scorer import CheckersScorer

__all__ = [
    'CheckersBoard',
    'Piece',
    'Move',
    'Color',
    'PieceType',
    'CheckersGame',
    'CheckersScorer',
]
