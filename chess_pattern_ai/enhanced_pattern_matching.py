#!/usr/bin/env python3
"""
Enhanced Pattern Matching - Advanced Pattern Recognition
Provides enhanced pattern matching capabilities for move evaluation.
"""

import sqlite3
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPatternMatcher:
    """Enhanced pattern matching for move evaluation"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.stats = {
            'total_queries': 0,
            'exact_matches': 0,
            'pattern_matches': 0
        }

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def get_move_bonus(self, board, move, perspective) -> float:
        """
        Get pattern-based bonus for a move

        Args:
            board: chess.Board object
            move: chess.Move object
            perspective: chess.Color

        Returns:
            Bonus score based on patterns
        """
        self.stats['total_queries'] += 1

        bonus = 0.0

        # Get FEN before move
        fen_before = board.fen()

        # Make move temporarily
        board.push(move)
        fen_after = board.fen()
        board.pop()

        # Check for exact position match in learned tactics
        try:
            self.cursor.execute('''
                SELECT material_gained, success_rate, times_seen
                FROM learned_tactics
                WHERE fen_before = ? AND move_sequence LIKE ?
                AND success_rate > 0.5
                ORDER BY material_gained DESC
                LIMIT 1
            ''', (fen_before, f"{move.uci()}%"))

            result = self.cursor.fetchone()
            if result:
                mat_gain, success_rate, times_seen = result
                confidence = min(1.0, times_seen / 10.0)
                bonus += mat_gain * success_rate * confidence
                self.stats['exact_matches'] += 1

        except Exception as e:
            logger.debug(f"Pattern query error: {e}")

        # Check for known mistakes to avoid
        try:
            self.cursor.execute('''
                SELECT material_lost
                FROM learned_mistakes
                WHERE fen_before = ? AND move_made = ?
                LIMIT 1
            ''', (fen_before, move.uci()))

            result = self.cursor.fetchone()
            if result:
                material_lost = result[0]
                bonus -= material_lost * 2.0  # Heavy penalty
                self.stats['pattern_matches'] += 1

        except Exception as e:
            logger.debug(f"Mistake query error: {e}")

        return bonus

    def get_stats(self) -> Dict:
        """Get pattern matching statistics"""
        exact_rate = 0.0
        pattern_rate = 0.0

        if self.stats['total_queries'] > 0:
            exact_rate = self.stats['exact_matches'] / self.stats['total_queries'] * 100
            pattern_rate = self.stats['pattern_matches'] / self.stats['total_queries'] * 100

        return {
            'total_queries': self.stats['total_queries'],
            'exact_matches': self.stats['exact_matches'],
            'pattern_matches': self.stats['pattern_matches'],
            'exact_match_rate': exact_rate,
            'pattern_match_rate': pattern_rate
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_enhanced_pattern_matcher():
    """Test the enhanced pattern matcher"""
    print("=" * 70)
    print("ENHANCED PATTERN MATCHER TEST")
    print("=" * 70)

    matcher = EnhancedPatternMatcher()

    # Check database content
    print("\nDatabase Content:")

    try:
        matcher.cursor.execute("SELECT COUNT(*) FROM learned_tactics")
        tactic_count = matcher.cursor.fetchone()[0]
        print(f"  Learned tactics: {tactic_count:,}")

        matcher.cursor.execute("SELECT COUNT(*) FROM learned_mistakes")
        mistake_count = matcher.cursor.fetchone()[0]
        print(f"  Learned mistakes: {mistake_count:,}")

    except Exception as e:
        print(f"  Error querying database: {e}")

    print("\nPattern matching statistics:")
    stats = matcher.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    matcher.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_enhanced_pattern_matcher()
