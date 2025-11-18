"""
Dots and Boxes Pattern Recognition AI

Observation-based learning for Dots and Boxes
"""

from .dots_boxes_board import DotsBoxesBoard, Edge, Color
from .dots_boxes_game import DotsBoxesGame
from .dots_boxes_scorer import DotsBoxesScorer

__all__ = [
    'DotsBoxesBoard',
    'Edge',
    'Color',
    'DotsBoxesGame',
    'DotsBoxesScorer',
]
