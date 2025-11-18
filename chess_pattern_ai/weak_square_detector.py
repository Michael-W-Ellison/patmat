#!/usr/bin/env python3
"""
Weak Square Detector - Loads Discovered Weak Square Patterns
Detects weak squares using statistically discovered patterns.
"""

import sqlite3
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeakSquareDetector:
    """Detects weak squares using discovered patterns"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.weak_square_weights = {}

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_weak_square_weights(self):
        """
        Load discovered weak square patterns from database
        """
        try:
            self.cursor.execute('''
                SELECT pattern_name, weight, confidence
                FROM weak_square_weights
                ORDER BY ABS(weight) DESC
            ''')

            patterns = self.cursor.fetchall()
            for pattern_name, weight, confidence in patterns:
                self.weak_square_weights[pattern_name] = {
                    'weight': weight,
                    'confidence': confidence
                }

            if self.weak_square_weights:
                logger.info(f"✓ Loaded {len(self.weak_square_weights)} weak square patterns")
            else:
                logger.warning("No weak square patterns found")

        except sqlite3.Error as e:
            logger.warning(f"Could not load weak square patterns: {e}")
            self.weak_square_weights = {}

    def evaluate_weak_squares(self, fen: str) -> float:
        """
        Evaluate weak square patterns

        Args:
            fen: FEN string of position

        Returns:
            Score based on weak square patterns (positive = white better)
        """
        if not self.weak_square_weights:
            self._load_weak_square_weights()

        # Parse FEN
        board_part = fen.split()[0]

        score = 0.0

        # Check discovered weak square patterns
        for pattern_name, data in self.weak_square_weights.items():
            weight = data['weight']
            confidence = data['confidence']

            if self._pattern_matches(pattern_name, board_part):
                score += weight * confidence

        return score

    def _pattern_matches(self, pattern_name: str, board_part: str) -> bool:
        """
        Check if a weak square pattern matches

        Args:
            pattern_name: Name of the pattern
            board_part: Board position from FEN

        Returns:
            True if pattern matches
        """
        pattern_lower = pattern_name.lower()

        # Placeholder pattern matching
        # Real implementation would check for:
        # - Holes in pawn structure
        # - Squares not controlled by pawns
        # - Outpost squares for knights
        # - Weak color complex

        if "weak_color" in pattern_lower:
            return self._has_weak_color_complex(board_part)

        elif "hole" in pattern_lower:
            return self._has_pawn_holes(board_part)

        return False

    def _has_weak_color_complex(self, board_part: str) -> bool:
        """Check for weak color complex (e.g., missing light/dark squared bishop)"""
        # Simple check: missing a bishop
        white_bishops = board_part.count('B')
        black_bishops = board_part.count('b')

        return white_bishops < 2 or black_bishops < 2

    def _has_pawn_holes(self, board_part: str) -> bool:
        """Check for holes in pawn structure"""
        board = self._parse_board(board_part)

        # Check each file for pawn presence
        for file in range(8):
            has_white_pawn = False
            has_black_pawn = False

            for rank in range(8):
                sq = rank * 8 + file
                piece = board[sq]
                if piece == 'P':
                    has_white_pawn = True
                elif piece == 'p':
                    has_black_pawn = True

            # Files without pawns can indicate weak squares
            if not has_white_pawn or not has_black_pawn:
                return True

        return False

    def _parse_board(self, board_part: str) -> List:
        """Parse FEN board part into 64-square array"""
        board = [None] * 64
        square = 56  # Start at a8

        for rank in board_part.split('/'):
            file = 0
            for char in rank:
                if char.isdigit():
                    file += int(char)
                else:
                    board[square + file] = char
                    file += 1
            square -= 8

        return board

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_weak_square_detector():
    """Test the weak square detector"""
    print("=" * 70)
    print("WEAK SQUARE DETECTOR TEST")
    print("=" * 70)

    detector = WeakSquareDetector()
    detector._load_weak_square_weights()

    print("\nDiscovered Weak Square Patterns:")
    for pattern, data in sorted(detector.weak_square_weights.items(),
                                key=lambda x: abs(x[1]['weight']), reverse=True):
        print(f"  {pattern:30s}: weight={data['weight']:+.2f}, conf={data['confidence']:.2f}")

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Missing bishop"),
        ("rnbqkbnr/p1pppppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1", "Files without pawns"),
    ]

    print("\n" + "=" * 70)
    print("Weak Square Evaluations:")
    print("=" * 70)

    for fen, description in test_positions:
        score = detector.evaluate_weak_squares(fen)
        print(f"\n{description}")
        print(f"  Weak square score: {score:+.2f}")

    detector.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_weak_square_detector()
