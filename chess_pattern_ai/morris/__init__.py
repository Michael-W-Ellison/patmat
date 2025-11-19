"""
Nine Men's Morris (Mills) game implementation for pattern learning AI.

Nine Men's Morris is a classic strategy game with three phases:
1. Placement phase: Players place 9 pieces on empty positions
2. Movement phase: Players move pieces to adjacent positions
3. Flying phase: When reduced to 3 pieces, can "fly" to any position

Goal: Form "mills" (3 in a row) to remove opponent's pieces
Win condition: Reduce opponent to 2 pieces or block all moves

Perfect for testing pattern learning because:
- Traditional AI often misses subtle mill patterns
- Three distinct phases require different strategies
- Small state space (24 positions) allows thorough exploration
- Mill formation is naturally pattern-based
"""

from .morris_board import MorrisBoard, Piece, Color
from .morris_game import MorrisGame
from .morris_scorer import MorrisScorer

__all__ = ['MorrisBoard', 'Piece', 'Color', 'MorrisGame', 'MorrisScorer']
