#!/usr/bin/env python3
"""
Opening Evaluator - Loads Discovered Opening Patterns
Evaluates opening positions using statistically discovered patterns.
No hardcoded opening theory - patterns learned from game analysis.
"""

import sqlite3
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpeningEvaluator:
    """Evaluates opening positions using discovered patterns"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.opening_weights = {}

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_opening_weights(self):
        """
        Load discovered opening weights from database
        These weights were learned from analyzing successful openings
        """
        try:
            # Load opening weights discovered from game analysis
            self.cursor.execute('''
                SELECT development_weight, center_control_weight, repetition_penalty, observation_count
                FROM discovered_opening_weights
                ORDER BY id DESC
                LIMIT 1
            ''')

            result = self.cursor.fetchone()
            if result:
                dev_w, center_w, rep_pen, obs_count = result
                self.opening_weights = {
                    'development': dev_w,
                    'center_control': center_w,
                    'repetition_penalty': rep_pen,
                    'observations': obs_count
                }
                logger.info(f"✓ Loaded discovered opening weights (from {obs_count} observations)")
            else:
                logger.warning("No discovered opening patterns found")
                self.opening_weights = {}

        except sqlite3.Error as e:
            logger.warning(f"Could not load opening patterns: {e}")
            self.opening_weights = {}

    def evaluate_opening(self, fen: str) -> float:
        """
        Evaluate opening position using discovered patterns

        Args:
            fen: FEN string of position

        Returns:
            Opening score from white's perspective (positive = white better opening)
        """
        if not self.opening_weights:
            self._load_opening_weights()

        # Parse FEN
        board_part, turn, castling, ep, halfmove, fullmove = fen.split()

        # Only evaluate in opening phase (first ~15 moves)
        move_num = int(fullmove)
        if move_num > 15:
            return 0.0  # Not in opening anymore

        score = 0.0

        # Apply discovered opening weights
        if not self.opening_weights:
            return 0.0

        dev_w = self.opening_weights.get('development', 1.0)
        center_w = self.opening_weights.get('center_control', 1.0)

        # Apply discovered weights to observable features
        # Count piece activity (how many pieces have moved)
        white_activity = self._count_piece_activity(board_part, 'white')
        black_activity = self._count_piece_activity(board_part, 'black')

        # Apply development weight to activity difference
        score += dev_w * (white_activity - black_activity)

        return score

    def _count_piece_activity(self, board_part: str, color: str) -> int:
        """Count pieces away from starting positions (observable feature)"""
        # Parse board
        board = self._parse_board(board_part)

        # Count pieces that have moved (observable, no assumptions about "good" squares)
        piece_activity = 0

        # Count total pieces for this color to measure development
        for sq in range(64):
            piece = board[sq]
            if piece:
                if color == 'white' and piece.isupper():
                    # Any white piece not on rank 0 or 1 has moved forward
                    rank = sq // 8
                    if rank > 1:
                        piece_activity += 1
                elif color == 'black' and piece.islower():
                    # Any black piece not on rank 6 or 7 has moved forward
                    rank = sq // 8
                    if rank < 6:
                        piece_activity += 1

        return piece_activity

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


def test_opening_evaluator():
    """Test the opening evaluator"""
    print("=" * 70)
    print("OPENING EVALUATOR TEST")
    print("=" * 70)

    evaluator = OpeningEvaluator()
    evaluator._load_opening_weights()

    print("\nDiscovered Opening Patterns:")
    for pattern, data in sorted(evaluator.opening_weights.items(),
                                key=lambda x: abs(x[1]['weight']), reverse=True):
        print(f"  {pattern:40s}: weight={data['weight']:+.2f}, conf={data['confidence']:.2f}")

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e4 opening"),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "After Nf6"),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2", "After Nf3"),
    ]

    print("\n" + "=" * 70)
    print("Opening Evaluations:")
    print("=" * 70)

    for fen, description in test_positions:
        score = evaluator.evaluate_opening(fen)
        print(f"\n{description}")
        print(f"  Opening score: {score:+.2f}")

    evaluator.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_opening_evaluator()
