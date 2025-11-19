"""
Pentago game implementation for pattern learning AI.

Pentago is a 2-player strategy game where:
- 6x6 board divided into 4 quadrants (each 3x3)
- Players place a marble then rotate a quadrant 90°
- Goal: Get 5 in a row (horizontal, vertical, diagonal)

Perfect for testing pattern learning because:
- Traditional AI struggles with rotation evaluation
- Clear tactical patterns (rotation traps, threat creation)
- Fast training (games typically 15-25 moves)
"""

from .pentago_board import PentagoBoard, Marble, Color
from .pentago_game import PentagoGame
from .pentago_scorer import PentagoScorer

__all__ = ['PentagoBoard', 'Marble', 'Color', 'PentagoGame', 'PentagoScorer']
