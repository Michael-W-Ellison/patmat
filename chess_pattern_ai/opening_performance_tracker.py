#!/usr/bin/env python3
"""
Opening Performance Tracker

Tracks actual game outcomes for each opening move, allowing the AI to:
- Learn which openings work best in practice
- Avoid repeatedly trying losing openings
- Converge on successful strategies over time

This solves the problem where caching makes the AI faster but doesn't
make it smarter - it keeps trying the same losing moves because they
evaluate well but perform poorly.
"""

import sqlite3
import chess
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class OpeningPerformanceTracker:
    """
    Tracks win/loss/draw statistics for opening moves

    Adjusts move scores based on historical performance
    """

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self._init_tables()

    def _init_tables(self):
        """Create table for tracking opening performance"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS opening_performance (
                fen TEXT,
                move_uci TEXT,
                times_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                avg_game_result REAL DEFAULT 0.0,
                last_played TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fen, move_uci)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_opening_perf_fen
            ON opening_performance(fen)
        ''')

        self.conn.commit()

    def record_opening_result(self, move_history: List[Tuple], result: str, ai_color: chess.Color):
        """
        Record the outcome of a game for opening moves

        Args:
            move_history: List of (fen, move_uci, move_san) tuples
            result: 'win', 'loss', or 'draw'
            ai_color: Color the AI was playing
        """
        # Convert result to numerical score
        if result == 'win':
            game_score = 1.0
        elif result == 'draw':
            game_score = 0.5
        else:  # loss
            game_score = 0.0

        # Record first 5 moves (opening)
        moves_to_record = min(5, len(move_history))

        for i, (fen, move_uci, move_san) in enumerate(move_history[:moves_to_record]):
            board = chess.Board(fen)

            # Only record our moves
            if board.turn != ai_color:
                continue

            # Upsert the performance record
            self.cursor.execute('''
                INSERT INTO opening_performance (fen, move_uci, times_played, wins, draws, losses, avg_game_result)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(fen, move_uci) DO UPDATE SET
                    times_played = times_played + 1,
                    wins = wins + ?,
                    draws = draws + ?,
                    losses = losses + ?,
                    avg_game_result = (avg_game_result * times_played + ?) / (times_played + 1),
                    last_played = CURRENT_TIMESTAMP
            ''', (
                fen, move_uci,
                1 if result == 'win' else 0,
                1 if result == 'draw' else 0,
                1 if result == 'loss' else 0,
                game_score,
                1 if result == 'win' else 0,
                1 if result == 'draw' else 0,
                1 if result == 'loss' else 0,
                game_score
            ))

        self.conn.commit()

    def get_opening_adjustment(self, fen: str, move_uci: str) -> float:
        """
        Get score adjustment based on historical performance

        Returns:
            Bonus/penalty to add to move evaluation
            - Positive for moves with good historical performance
            - Negative for moves that have lost repeatedly
            - 0 for unexplored moves
        """
        self.cursor.execute('''
            SELECT times_played, wins, draws, losses, avg_game_result
            FROM opening_performance
            WHERE fen = ? AND move_uci = ?
        ''', (fen, move_uci))

        row = self.cursor.fetchone()

        if not row or row[0] == 0:
            # Never tried this move - no adjustment
            return 0.0

        times_played, wins, draws, losses, avg_result = row

        # Calculate adjustment based on performance
        # avg_result is 0.0 (all losses) to 1.0 (all wins)
        # Map to -100 (terrible) to +100 (excellent)

        # Center around 0.5 (expected with random play)
        performance_delta = avg_result - 0.5  # Range: -0.5 to +0.5

        # Scale by confidence (more games = more confident)
        confidence = min(1.0, times_played / 10.0)  # Full confidence after 10 games

        # Base adjustment: -50 to +50 points
        base_adjustment = performance_delta * 100

        # Scale by confidence
        adjustment = base_adjustment * confidence

        # Add exploration bonus for rarely-tried moves
        if times_played < 3:
            exploration_bonus = (3 - times_played) * 5  # +10 for never tried, +5 for tried once
            adjustment += exploration_bonus

        return adjustment

    def get_opening_stats(self, fen: str) -> List[Tuple]:
        """
        Get performance statistics for all moves from a position

        Returns:
            List of (move_uci, times_played, win_rate, avg_result, adjustment)
        """
        self.cursor.execute('''
            SELECT move_uci, times_played, wins, draws, losses, avg_game_result
            FROM opening_performance
            WHERE fen = ?
            ORDER BY times_played DESC
        ''', (fen,))

        results = []
        for row in self.cursor.fetchall():
            move_uci, times_played, wins, draws, losses, avg_result = row

            if times_played > 0:
                win_rate = wins / times_played * 100
            else:
                win_rate = 0.0

            adjustment = self.get_opening_adjustment(fen, move_uci)

            results.append((move_uci, times_played, win_rate, avg_result, adjustment))

        return results

    def print_opening_statistics(self, fen: str = None):
        """Print opening performance statistics"""
        if fen is None:
            # Use standard starting position
            fen = chess.Board().fen()

        print("=" * 70)
        print("OPENING PERFORMANCE STATISTICS")
        print("=" * 70)
        print(f"\nPosition: {fen}\n")

        stats = self.get_opening_stats(fen)

        if not stats:
            print("No games played from this position yet.")
            return

        print(f"{'Move':<8} {'Played':<8} {'W-D-L':<12} {'Win%':<8} {'Avg':<8} {'Adjustment':<12}")
        print("-" * 70)

        for move_uci, times_played, win_rate, avg_result, adjustment in stats:
            # Get W-D-L record
            self.cursor.execute('''
                SELECT wins, draws, losses
                FROM opening_performance
                WHERE fen = ? AND move_uci = ?
            ''', (fen, move_uci))

            wins, draws, losses = self.cursor.fetchone()
            record = f"{wins}-{draws}-{losses}"

            adj_str = f"{adjustment:+.1f}" if adjustment != 0 else "—"

            print(f"{move_uci:<8} {times_played:<8} {record:<12} {win_rate:<7.1f}% {avg_result:<8.3f} {adj_str:<12}")

        print("=" * 70)


def test_opening_tracker():
    """Test the opening performance tracker"""
    tracker = OpeningPerformanceTracker()

    # Simulate some games
    board = chess.Board()
    start_fen = board.fen()

    # Game 1: e4, lost
    move_history = [(start_fen, "e2e4", "e4")]
    tracker.record_opening_result(move_history, 'loss', chess.WHITE)

    # Game 2: e4, lost again
    tracker.record_opening_result(move_history, 'loss', chess.WHITE)

    # Game 3: d4, won
    move_history2 = [(start_fen, "d2d4", "d4")]
    tracker.record_opening_result(move_history2, 'win', chess.WHITE)

    # Check adjustments
    print("\nAfter 3 games:")
    print(f"e4 adjustment: {tracker.get_opening_adjustment(start_fen, 'e2e4'):+.1f}")
    print(f"d4 adjustment: {tracker.get_opening_adjustment(start_fen, 'd2d4'):+.1f}")
    print(f"Nf3 adjustment (never tried): {tracker.get_opening_adjustment(start_fen, 'g1f3'):+.1f}")

    tracker.print_opening_statistics()


if __name__ == '__main__':
    test_opening_tracker()
