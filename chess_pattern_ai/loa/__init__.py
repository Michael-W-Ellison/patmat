"""
Lines of Action (LOA) game implementation for pattern learning AI.

Lines of Action is a connection game where:
- 8x8 board with pieces starting on edges
- Move distance = number of pieces in that line (yours + opponent's)
- Can jump over own pieces, NOT opponent pieces
- Can capture by landing on opponent piece
- Goal: Connect all your pieces into a single group

Perfect for testing pattern learning because:
- Traditional AI struggles with connectivity evaluation
- Unique movement rule (distance = pieces in line)
- Positions can change dramatically with one move
- Grouping patterns are naturally learned through observation
"""

from .loa_board import LOABoard, Piece, Color
from .loa_game import LOAGame
from .loa_scorer import LOAScorer

__all__ = ['LOABoard', 'Piece', 'Color', 'LOAGame', 'LOAScorer']
