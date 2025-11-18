#!/usr/bin/env python3
"""
Material Evaluator - Loads Discovered Piece Values
Evaluates material balance using statistically discovered piece values.
No hardcoded chess knowledge - values learned from game observations.
"""

import sqlite3
import logging
from typing import Dict
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaterialEvaluator:
    """Evaluates material balance using discovered piece values"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.piece_values = {}

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_piece_values(self):
        """
        Load discovered piece values from database
        These values were learned from observing exchanges and game outcomes
        """
        try:
            self.cursor.execute('''
                SELECT piece_type, discovered_value, confidence
                FROM discovered_piece_values
                ORDER BY piece_type
            ''')

            for piece_type, value, confidence in self.cursor.fetchall():
                self.piece_values[piece_type] = value
                logger.debug(f"Loaded {piece_type} = {value} (confidence: {confidence:.2f})")

            if self.piece_values:
                logger.info(f"✓ Loaded {len(self.piece_values)} discovered piece values")
            else:
                # Fallback to minimal values if discovery hasn't run yet
                logger.warning("No discovered piece values found, using minimal fallback")
                self.piece_values = {'P': 1.0, 'N': 3.0, 'B': 3.0, 'R': 5.0, 'Q': 9.0, 'K': 0.0}

        except sqlite3.Error as e:
            logger.warning(f"Could not load piece values: {e}, using fallback")
            self.piece_values = {'P': 1.0, 'N': 3.0, 'B': 3.0, 'R': 5.0, 'Q': 9.0, 'K': 0.0}

    def evaluate_material(self, fen: str) -> float:
        """
        Evaluate material balance for the position

        Args:
            fen: FEN string of position

        Returns:
            Material score from white's perspective (positive = white ahead)
        """
        if not self.piece_values:
            self._load_piece_values()

        # Parse FEN to get piece placement (first field)
        board_part = fen.split()[0]

        white_material = 0.0
        black_material = 0.0

        # Count pieces from FEN
        for char in board_part:
            if char.isupper():  # White piece
                piece_type = char
                value = self.piece_values.get(piece_type, 0.0)
                white_material += value
            elif char.islower():  # Black piece
                piece_type = char.upper()
                value = self.piece_values.get(piece_type, 0.0)
                black_material += value

        # Return difference (positive = white ahead)
        return white_material - black_material

    def get_piece_value(self, piece_type: str) -> float:
        """Get discovered value for a piece type"""
        if not self.piece_values:
            self._load_piece_values()
        return self.piece_values.get(piece_type.upper(), 0.0)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_material_evaluator():
    """Test the material evaluator"""
    print("=" * 70)
    print("MATERIAL EVALUATOR TEST")
    print("=" * 70)

    evaluator = MaterialEvaluator()
    evaluator._load_piece_values()

    print("\nDiscovered Piece Values:")
    for piece, value in sorted(evaluator.piece_values.items()):
        print(f"  {piece}: {value}")

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1", "White missing rook"),
        ("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black missing knight"),
        ("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1", "White has huge material advantage"),
    ]

    print("\n" + "=" * 70)
    print("Position Evaluations:")
    print("=" * 70)

    for fen, description in test_positions:
        score = evaluator.evaluate_material(fen)
        print(f"\n{description}")
        print(f"  FEN: {fen[:50]}...")
        print(f"  Material score: {score:+.1f} (positive = white ahead)")

    evaluator.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_material_evaluator()
