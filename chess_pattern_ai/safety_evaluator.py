#!/usr/bin/env python3
"""
Safety Evaluator - Loads Discovered King Safety Patterns
Evaluates king safety using statistically discovered patterns.
No hardcoded chess knowledge - patterns learned from game analysis.
"""

import sqlite3
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SafetyEvaluator:
    """Evaluates king safety using discovered patterns"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.safety_weights = {}
        self.safety_patterns = []

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_safety_weights(self):
        """
        Load discovered safety weights from database
        These weights were learned from analyzing safe vs unsafe king positions
        """
        try:
            # Load safety weights discovered from game analysis
            self.cursor.execute('''
                SELECT king_safety_weight, piece_protection_weight, exposed_penalty, observation_count
                FROM discovered_safety_patterns
                ORDER BY id DESC
                LIMIT 1
            ''')

            result = self.cursor.fetchone()
            if result:
                king_safety_w, piece_protect_w, exposed_pen, obs_count = result
                self.safety_weights = {
                    'king_safety': king_safety_w,
                    'piece_protection': piece_protect_w,
                    'exposed_penalty': exposed_pen,
                    'observations': obs_count
                }
                logger.info(f"✓ Loaded discovered safety weights (from {obs_count} observations)")
            else:
                logger.warning("No discovered safety patterns found")
                self.safety_weights = {}

        except sqlite3.Error as e:
            logger.warning(f"Could not load safety patterns: {e}")
            # Minimal fallback - no patterns
            self.safety_weights = {}

    def evaluate_safety(self, fen: str) -> float:
        """
        Evaluate king safety for the position

        Args:
            fen: FEN string of position

        Returns:
            Safety score from white's perspective (positive = white safer)
        """
        if not self.safety_weights:
            self._load_safety_weights()

        # Parse FEN
        board_part, turn, castling, ep, halfmove, fullmove = fen.split()

        white_safety = 0.0
        black_safety = 0.0

        # Extract king positions
        white_king_square, black_king_square = self._find_kings(board_part)

        if white_king_square is None or black_king_square is None:
            return 0.0

        # Evaluate white king safety
        white_safety = self._evaluate_king_safety(
            board_part, white_king_square, 'white', castling
        )

        # Evaluate black king safety
        black_safety = self._evaluate_king_safety(
            board_part, black_king_square, 'black', castling
        )

        # Return difference (positive = white safer)
        return white_safety - black_safety

    def _find_kings(self, board_part: str) -> Tuple[int, int]:
        """Find king positions in FEN board representation"""
        white_king = None
        black_king = None
        square = 56  # Start at a8

        for rank in board_part.split('/'):
            file = 0
            for char in rank:
                if char.isdigit():
                    file += int(char)
                else:
                    if char == 'K':
                        white_king = square + file
                    elif char == 'k':
                        black_king = square + file
                    file += 1
            square -= 8

        return white_king, black_king

    def _evaluate_king_safety(self, board_part: str, king_square: int,
                             color: str, castling: str) -> float:
        """
        Evaluate safety of a specific king using discovered patterns

        Args:
            board_part: Board position from FEN
            king_square: Square index of king (0-63)
            color: 'white' or 'black'
            castling: Castling rights string

        Returns:
            Safety score (higher = safer)
        """
        safety = 0.0

        # Apply discovered safety weights
        if not self.safety_weights:
            return 0.0

        king_safety_w = self.safety_weights.get('king_safety', 1.0)
        piece_protect_w = self.safety_weights.get('piece_protection', 1.0)
        exposed_pen = self.safety_weights.get('exposed_penalty', 1.0)

        # Apply discovered weights to observable features
        # Count nearby friendly pieces (more = safer, discovered from observation)
        nearby_pieces = self._count_nearby_pieces(board_part, king_square, color)
        safety += piece_protect_w * nearby_pieces

        # Count pawn shield (discovered pattern)
        pawn_shield_count = self._count_pawn_shield(board_part, king_square, color)
        safety += king_safety_w * pawn_shield_count

        return safety

    def _count_nearby_pieces(self, board_part: str, king_square: int, color: str) -> int:
        """Count friendly pieces near king (observable feature)"""
        board = [None] * 64
        square = 56

        for rank in board_part.split('/'):
            file = 0
            for char in rank:
                if char.isdigit():
                    file += int(char)
                else:
                    board[square + file] = char
                    file += 1
            square -= 8

        king_file = king_square % 8
        king_rank = king_square // 8
        nearby_pieces = 0

        # Check 8 surrounding squares
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                check_rank = king_rank + dr
                check_file = king_file + df
                if 0 <= check_rank < 8 and 0 <= check_file < 8:
                    check_square = check_rank * 8 + check_file
                    piece = board[check_square]
                    if piece:
                        # Count friendly pieces
                        if color == 'white' and piece.isupper():
                            nearby_pieces += 1
                        elif color == 'black' and piece.islower():
                            nearby_pieces += 1

        return nearby_pieces

    def _count_pawn_shield(self, board_part: str, king_square: int, color: str) -> int:
        """Count pawns near king (observable feature, no assumptions about what's good)"""
        # Convert board to simple representation
        board = [None] * 64
        square = 56

        for rank in board_part.split('/'):
            file = 0
            for char in rank:
                if char.isdigit():
                    file += int(char)
                else:
                    board[square + file] = char
                    file += 1
            square -= 8

        # Count friendly pawns in all adjacent squares
        king_file = king_square % 8
        king_rank = king_square // 8

        pawn_char = 'P' if color == 'white' else 'p'
        pawn_count = 0

        # Check all 8 surrounding squares for pawns
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                check_rank = king_rank + dr
                check_file = king_file + df
                if 0 <= check_rank < 8 and 0 <= check_file < 8:
                    check_square = check_rank * 8 + check_file
                    if board[check_square] == pawn_char:
                        pawn_count += 1

        return pawn_count

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_safety_evaluator():
    """Test the safety evaluator"""
    print("=" * 70)
    print("SAFETY EVALUATOR TEST")
    print("=" * 70)

    evaluator = SafetyEvaluator()
    evaluator._load_safety_weights()

    print("\nDiscovered Safety Patterns:")
    for pattern, data in sorted(evaluator.safety_weights.items(),
                                key=lambda x: abs(x[1]['weight']), reverse=True):
        print(f"  {pattern:30s}: weight={data['weight']:+.2f}, conf={data['confidence']:.2f}")

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1", "White king moved"),
        ("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black king moved"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "Both kings central"),
    ]

    print("\n" + "=" * 70)
    print("Position Evaluations:")
    print("=" * 70)

    for fen, description in test_positions:
        score = evaluator.evaluate_safety(fen)
        print(f"\n{description}")
        print(f"  Safety score: {score:+.2f} (positive = white safer)")

    evaluator.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_safety_evaluator()
