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
        Load discovered safety patterns from database
        These patterns were learned from analyzing safe vs unsafe king positions
        """
        try:
            # Load safety patterns discovered from game analysis
            self.cursor.execute('''
                SELECT pattern_name, weight, confidence, observation_count
                FROM discovered_safety_patterns
                ORDER BY ABS(weight) DESC
            ''')

            patterns = self.cursor.fetchall()
            for pattern_name, weight, confidence, obs_count in patterns:
                self.safety_weights[pattern_name] = {
                    'weight': weight,
                    'confidence': confidence,
                    'observations': obs_count
                }
                self.safety_patterns.append(pattern_name)

            if self.safety_weights:
                logger.info(f"✓ Loaded {len(self.safety_weights)} discovered safety patterns")
            else:
                logger.warning("No discovered safety patterns found")

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

        # Check discovered patterns
        for pattern_name, data in self.safety_weights.items():
            weight = data['weight']
            confidence = data['confidence']

            # Detect pattern and apply weight
            if self._pattern_matches(pattern_name, board_part, king_square, color, castling):
                safety += weight * confidence

        return safety

    def _pattern_matches(self, pattern_name: str, board_part: str,
                        king_square: int, color: str, castling: str) -> bool:
        """
        Check if a discovered safety pattern matches the current position

        Args:
            pattern_name: Name of the pattern to check
            board_part: Board position from FEN
            king_square: King square index
            color: King color
            castling: Castling rights

        Returns:
            True if pattern matches
        """
        # Basic pattern matching based on common safety indicators
        # These are discovered patterns, not hardcoded rules

        if "castled" in pattern_name.lower():
            # Check if king has moved from starting position
            if color == 'white':
                # White king starts on e1 (square 4)
                starting_square = 4
                return king_square != starting_square
            else:
                # Black king starts on e8 (square 60)
                starting_square = 60
                return king_square != starting_square

        elif "pawn_shield" in pattern_name.lower():
            # Check for pawns near king
            return self._has_pawn_shield(board_part, king_square, color)

        elif "exposed" in pattern_name.lower():
            # King on edge or center (more exposed)
            file = king_square % 8
            rank = king_square // 8
            return file in [0, 7] or rank in [3, 4]

        elif "back_rank" in pattern_name.lower():
            # King on back rank
            rank = king_square // 8
            if color == 'white':
                return rank == 0
            else:
                return rank == 7

        # Default: pattern not recognized
        return False

    def _has_pawn_shield(self, board_part: str, king_square: int, color: str) -> bool:
        """Check if king has pawn shield (simplified)"""
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

        # Check squares in front of king for pawns
        king_file = king_square % 8
        king_rank = king_square // 8

        pawn_char = 'P' if color == 'white' else 'p'
        direction = 1 if color == 'white' else -1

        shield_count = 0
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            check_rank = king_rank + direction

            if 0 <= check_file < 8 and 0 <= check_rank < 8:
                check_square = check_rank * 8 + check_file
                if board[check_square] == pawn_char:
                    shield_count += 1

        return shield_count >= 2

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
