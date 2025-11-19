"""
Breakthrough Pattern Recognition AI

Observation-based learning for Breakthrough game
"""

from .breakthrough_board import BreakthroughBoard, Piece, Move, Color
from .breakthrough_game import BreakthroughGame
from .breakthrough_scorer import BreakthroughScorer

__all__ = [
    'BreakthroughBoard',
    'Piece',
    'Move',
    'Color',
    'BreakthroughGame',
    'BreakthroughScorer',
]
