#!/usr/bin/env python3
"""
Pattern Decay Manager - Recency-Weighted Learning

Solves the problem: Patterns that worked at Level 0 shouldn't dominate at Level 10

Features:
- Recency weighting (recent games count more)
- Exponential decay (old patterns fade)
- Configurable decay rate
- Compatible with existing database
"""

import sqlite3
import math
from datetime import datetime, timedelta


class PatternDecayManager:
    """Manages pattern priorities with recency weighting"""

    def __init__(self, db_path, decay_rate=0.95, window_size=100):
        """
        Args:
            db_path: Path to pattern database
            decay_rate: How fast old games fade (0.9 = faster, 0.99 = slower)
            window_size: Number of recent games to emphasize
        """
        self.db_path = db_path
        self.decay_rate = decay_rate
        self.window_size = window_size

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self._init_decay_tables()

    def _init_decay_tables(self):
        """Create tables for tracking game history per pattern"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_game_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                distance_from_start INTEGER NOT NULL,
                game_phase TEXT NOT NULL,
                game_number INTEGER NOT NULL,
                result TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(piece_type, move_category, distance_from_start, game_phase, game_number)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pattern_game_history_lookup
            ON pattern_game_history(piece_type, move_category, distance_from_start, game_phase)
        ''')

        self.conn.commit()

    def record_pattern_result(self, piece_type, move_category, distance, phase, game_number, result, score):
        """Record individual game result for a pattern"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO pattern_game_history
                (piece_type, move_category, distance_from_start, game_phase,
                 game_number, result, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (piece_type, move_category, distance, phase, game_number, result, score))

        self.conn.commit()

    def calculate_weighted_priority(self, piece_type, move_category, distance, phase):
        """
        Calculate priority with exponential decay weighting

        Recent games have higher weight:
        - Game N (most recent): weight = 1.0
        - Game N-1: weight = 0.95
        - Game N-2: weight = 0.95^2 = 0.9025
        - Game N-10: weight = 0.95^10 = 0.599
        - Game N-50: weight = 0.95^50 = 0.077
        """
        self.cursor.execute('''
            SELECT game_number, score
            FROM pattern_game_history
            WHERE piece_type = ? AND move_category = ?
              AND distance_from_start = ? AND game_phase = ?
            ORDER BY game_number DESC
            LIMIT ?
        ''', (piece_type, move_category, distance, phase, self.window_size))

        results = self.cursor.fetchall()

        if not results:
            return 40.0, 0.0  # Default priority, no confidence

        # Calculate weighted average
        weighted_sum = 0.0
        weight_total = 0.0

        max_game = results[0][0]  # Most recent game number

        for game_num, score in results:
            age = max_game - game_num
            weight = math.pow(self.decay_rate, age)

            weighted_sum += score * weight
            weight_total += weight

        weighted_avg = weighted_sum / weight_total if weight_total > 0 else 0

        # Confidence based on number of games (but capped)
        confidence = min(1.0, len(results) / 50.0)

        # Priority calculation (same as original)
        normalized_score = (weighted_avg + 1500) / 31
        priority = normalized_score * confidence

        return priority, confidence

    def get_pattern_statistics(self, piece_type, move_category, distance, phase, limit=20):
        """Get recent performance statistics for a pattern"""
        self.cursor.execute('''
            SELECT game_number, result, score, timestamp
            FROM pattern_game_history
            WHERE piece_type = ? AND move_category = ?
              AND distance_from_start = ? AND game_phase = ?
            ORDER BY game_number DESC
            LIMIT ?
        ''', (piece_type, move_category, distance, phase, limit))

        return self.cursor.fetchall()

    def get_pattern_trend(self, piece_type, move_category, distance, phase):
        """
        Determine if pattern is improving, stable, or declining

        Returns: ('improving' | 'stable' | 'declining', trend_score)
        """
        stats = self.get_pattern_statistics(piece_type, move_category, distance, phase, limit=20)

        if len(stats) < 10:
            return 'unknown', 0.0

        # Compare recent 5 vs previous 5
        recent_5 = [s[2] for s in stats[:5]]  # scores
        previous_5 = [s[2] for s in stats[5:10]]

        recent_avg = sum(recent_5) / len(recent_5)
        previous_avg = sum(previous_5) / len(previous_5)

        diff = recent_avg - previous_avg

        if diff > 200:
            return 'improving', diff
        elif diff < -200:
            return 'declining', diff
        else:
            return 'stable', diff

    def recompute_all_priorities(self):
        """Recompute priorities for all patterns using decay weighting"""
        # Get all unique patterns
        self.cursor.execute('''
            SELECT DISTINCT piece_type, move_category, distance_from_start, game_phase
            FROM pattern_game_history
        ''')

        patterns = self.cursor.fetchall()

        print(f"Recomputing priorities for {len(patterns)} patterns...")

        for piece_type, move_category, distance, phase in patterns:
            priority, confidence = self.calculate_weighted_priority(
                piece_type, move_category, distance, phase
            )

            # Update in main pattern table
            self.cursor.execute('''
                UPDATE learned_move_patterns
                SET priority_score = ?,
                    confidence = ?,
                    updated_at = datetime('now')
                WHERE piece_type = ? AND move_category = ?
                  AND distance_from_start = ? AND game_phase = ?
            ''', (priority, confidence, piece_type, move_category, distance, phase))

        self.conn.commit()
        print(f"✓ Priorities recomputed with decay rate {self.decay_rate}")

    def get_declining_patterns(self, threshold=-300, limit=10):
        """Get patterns that are declining in performance"""
        # Get all unique patterns
        self.cursor.execute('''
            SELECT DISTINCT piece_type, move_category, distance_from_start, game_phase
            FROM pattern_game_history
        ''')

        patterns = self.cursor.fetchall()
        declining = []

        for piece_type, move_category, distance, phase in patterns:
            trend, score = self.get_pattern_trend(piece_type, move_category, distance, phase)

            if trend == 'declining' and score < threshold:
                priority, confidence = self.calculate_weighted_priority(
                    piece_type, move_category, distance, phase
                )
                declining.append({
                    'piece_type': piece_type,
                    'move_category': move_category,
                    'distance': distance,
                    'phase': phase,
                    'trend_score': score,
                    'priority': priority,
                    'confidence': confidence
                })

        # Sort by trend score (most declining first)
        declining.sort(key=lambda x: x['trend_score'])

        return declining[:limit]

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Demonstrate pattern decay management"""
    import argparse

    parser = argparse.ArgumentParser(description='Pattern Decay Manager')
    parser.add_argument('database', help='Path to pattern database')
    parser.add_argument('--recompute', action='store_true',
                       help='Recompute all priorities with decay')
    parser.add_argument('--decay-rate', type=float, default=0.95,
                       help='Decay rate (0.9=fast, 0.99=slow)')
    parser.add_argument('--window', type=int, default=100,
                       help='Window size for recent games')
    parser.add_argument('--show-declining', type=int, default=0, metavar='N',
                       help='Show N declining patterns')

    args = parser.parse_args()

    manager = PatternDecayManager(args.database, args.decay_rate, args.window)

    try:
        if args.recompute:
            manager.recompute_all_priorities()

        if args.show_declining > 0:
            print("\n" + "=" * 70)
            print(f"TOP {args.show_declining} DECLINING PATTERNS")
            print("=" * 70)

            declining = manager.get_declining_patterns(limit=args.show_declining)

            if not declining:
                print("No declining patterns found!")
            else:
                print(f"{'Piece':<10} {'Category':<14} {'Phase':<12} {'Trend':<10} {'Priority':<10}")
                print("-" * 70)

                for p in declining:
                    print(f"{p['piece_type']:<10} {p['move_category']:<14} "
                          f"{p['phase']:<12} {p['trend_score']:<+10.0f} {p['priority']:<10.1f}")

            print("=" * 70)

    finally:
        manager.close()


if __name__ == '__main__':
    main()
