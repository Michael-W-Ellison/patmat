"""
Arimaa game implementation for pattern learning AI.

Arimaa is a strategy game specifically designed to be difficult for computers:
- 8×8 board with 4 trap squares
- Each player has 16 pieces with strength hierarchy
- Each turn = up to 4 steps
- Can push/pull opponent pieces (stronger pushes weaker)
- Goal: Get rabbit to opponent's home row OR capture all opponent rabbits
- Branching factor: up to 16^4 (extremely high!)

Perfect for testing pattern learning because:
- Traditional AI performs poorly (designed to be hard for computers)
- Huge branching factor makes tree search impractical
- Push/pull mechanics create complex tactical patterns
- Trap control requires strategic understanding
- Piece strength relationships are learned through observation
"""

from .arimaa_board import ArimaaBoard, Piece, Color, PieceType
from .arimaa_game import ArimaaGame
from .arimaa_scorer import ArimaaScorer

__all__ = ['ArimaaBoard', 'Piece', 'Color', 'PieceType', 'ArimaaGame', 'ArimaaScorer']
