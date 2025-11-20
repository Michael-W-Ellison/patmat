#!/usr/bin/env python3
"""
Import PGN Patterns - Extract Patterns from Chess Game Database

This script processes large PGN files (e.g., 1 million games) and extracts
patterns to build the AI's training database without playing games.

Features:
- Processes PGN files in batches
- Tracks win/loss patterns from real games
- Updates learned_move_patterns database
- Shows progress during processing
- Handles large files efficiently

Usage:
    python chess_pattern_ai/import_pgn_patterns.py games.pgn
    python chess_pattern_ai/import_pgn_patterns.py --db chess_training.db --games games.pgn
    python chess_pattern_ai/import_pgn_patterns.py --limit 10000 games.pgn  # Process first 10K games
"""

import sqlite3
import re
import sys
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MinimalPGNParser:
    """
    Minimal PGN parser without requiring python-chess

    Parses PGN format and extracts:
    - Game results (1-0, 0-1, 1/2-1/2)
    - Move notation (SAN format)
    - Move characteristics (captures, checks, piece types)
    """

    def __init__(self):
        self.piece_symbols = {'K': 'king', 'Q': 'queen', 'R': 'rook',
                             'B': 'bishop', 'N': 'knight', '': 'pawn'}

    def parse_file(self, filename: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Parse PGN file and extract games

        Args:
            filename: Path to PGN file
            limit: Maximum number of games to process (None = all)

        Returns:
            List of game dictionaries with moves and results
        """
        games = []
        current_game = {'headers': {}, 'moves': []}
        in_moves = False

        print(f"Reading {filename}...")

        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        # Empty line marks end of game
                        if current_game['moves']:
                            games.append(current_game)
                            current_game = {'headers': {}, 'moves': []}
                            in_moves = False

                            if limit and len(games) >= limit:
                                break

                            if len(games) % 1000 == 0:
                                print(f"  Parsed {len(games):,} games...", end='\r')
                        continue

                    # Parse headers
                    if line.startswith('['):
                        match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
                        if match:
                            key, value = match.groups()
                            current_game['headers'][key] = value
                    else:
                        # Parse moves
                        in_moves = True
                        # Remove move numbers and annotations
                        moves_text = re.sub(r'\d+\.+', '', line)  # Remove move numbers
                        moves_text = re.sub(r'\{[^}]*\}', '', moves_text)  # Remove comments
                        moves_text = re.sub(r'\([^)]*\)', '', moves_text)  # Remove variations

                        # Split into individual moves
                        tokens = moves_text.split()
                        for token in tokens:
                            # Check for game result
                            if token in ['1-0', '0-1', '1/2-1/2', '*']:
                                current_game['result'] = token
                            elif token and not token.startswith('$'):
                                current_game['moves'].append(token)

                # Add last game if exists
                if current_game['moves']:
                    games.append(current_game)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found!")
            print("\nPlease provide the full path to your PGN file:")
            print("  python chess_pattern_ai/import_pgn_patterns.py /path/to/games.pgn")
            sys.exit(1)

        print(f"\n✓ Parsed {len(games):,} games from {filename}")
        return games

    def classify_move(self, move_san: str, move_num: int, total_moves: int) -> Dict:
        """
        Classify a move from SAN notation

        Args:
            move_san: Move in Standard Algebraic Notation (e.g., 'Nf3', 'exd5+')
            move_num: Current move number
            total_moves: Total moves in game

        Returns:
            Dict with piece_type, move_category, game_phase
        """
        # Identify piece type
        piece_type = 'pawn'
        if move_san[0] in 'KQRBN':
            piece_type = self.piece_symbols[move_san[0]]

        # Identify move category
        is_capture = 'x' in move_san
        is_check = '+' in move_san or '#' in move_san

        if is_capture and is_check:
            move_category = 'capture_check'
        elif is_capture:
            move_category = 'capture'
        elif is_check:
            move_category = 'check'
        else:
            # Heuristic: development vs quiet
            # Early non-pawn moves are likely development
            if move_num <= 12 and piece_type != 'pawn':
                move_category = 'development'
            else:
                move_category = 'quiet'

        # Determine game phase
        if move_num <= 12:
            game_phase = 'opening'
        elif total_moves - move_num < 15:
            game_phase = 'endgame'
        else:
            game_phase = 'middlegame'

        # Estimate distance from start (simplified heuristic)
        # In real games, pieces typically move 2-4 squares from start
        distance_from_start = min(move_num // 3, 4) if piece_type != 'pawn' else 2

        return {
            'piece_type': piece_type,
            'move_category': move_category,
            'game_phase': game_phase,
            'distance_from_start': distance_from_start
        }


class PGNPatternImporter:
    """
    Import patterns from PGN games into training database

    Analyzes games and updates learned_move_patterns with:
    - Win/loss statistics per move type
    - Confidence scores
    - Priority scores
    """

    def __init__(self, db_path: str = 'headless_training.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.parser = MinimalPGNParser()

        self._init_database()

    def _init_database(self):
        """Initialize database tables if needed"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_move_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                distance_from_start INTEGER,
                game_phase TEXT,
                times_seen INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                games_lost INTEGER DEFAULT 0,
                games_drawn INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_score REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(piece_type, move_category, distance_from_start, game_phase)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_move_pattern_priority
            ON learned_move_patterns(priority_score DESC, confidence DESC)
        ''')

        self.conn.commit()

    def import_games(self, pgn_file: str, limit: Optional[int] = None,
                     color_filter: Optional[str] = None):
        """
        Import patterns from PGN file

        Args:
            pgn_file: Path to PGN file
            limit: Maximum games to process
            color_filter: 'white' or 'black' to only learn from that side
        """
        print("=" * 70)
        print("PGN PATTERN IMPORT")
        print("=" * 70)
        print(f"Database: {self.db_path}")
        print(f"PGN File: {pgn_file}")
        if limit:
            print(f"Limit: {limit:,} games")
        print()

        # Parse games
        games = self.parser.parse_file(pgn_file, limit)

        if not games:
            print("No games found in PGN file!")
            return

        # Process games
        print(f"\nProcessing {len(games):,} games...")
        print("-" * 70)

        start_time = time.time()
        processed = 0
        white_wins = 0
        black_wins = 0
        draws = 0
        skipped = 0

        for i, game in enumerate(games, 1):
            result = game.get('result', '*')

            # Skip games without result
            if result == '*':
                skipped += 1
                continue

            # Determine winner
            if result == '1-0':
                white_wins += 1
                white_result = 'win'
                black_result = 'loss'
            elif result == '0-1':
                black_wins += 1
                white_result = 'loss'
                black_result = 'win'
            else:  # Draw
                draws += 1
                white_result = 'draw'
                black_result = 'draw'

            # Process moves
            total_moves = len(game['moves'])

            for move_num, move in enumerate(game['moves'], 1):
                # Determine whose move
                is_white = (move_num % 2) == 1

                # Apply color filter if specified
                if color_filter == 'white' and not is_white:
                    continue
                if color_filter == 'black' and is_white:
                    continue

                # Get move characteristics
                characteristics = self.parser.classify_move(
                    move, move_num, total_moves
                )

                # Determine result for this player
                result_for_player = white_result if is_white else black_result

                # Update database
                self._update_pattern(characteristics, result_for_player)

            processed += 1

            # Show progress
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (len(games) - processed) / rate if rate > 0 else 0
                print(f"  Progress: {processed:,}/{len(games):,} games "
                      f"({processed/len(games)*100:.1f}%) | "
                      f"Rate: {rate:.1f} games/s | "
                      f"ETA: {eta:.0f}s", end='\r')

        # Final commit
        self.conn.commit()

        elapsed = time.time() - start_time

        print(f"\n\n{'=' * 70}")
        print("IMPORT COMPLETE")
        print("=" * 70)
        print(f"Database: {self.db_path}")
        print(f"Time: {elapsed:.1f}s ({elapsed/processed:.2f}s per game)")
        print(f"Processed: {processed:,} games")
        print(f"Results: {white_wins:,}W-{black_wins:,}B-{draws:,}D")
        print(f"Skipped: {skipped:,} (no result)")

        # Show statistics
        self._show_statistics()

        # Verify database was written
        import os
        db_size = os.path.getsize(self.db_path)
        print(f"\nDatabase size: {db_size / 1024 / 1024:.1f} MB")

        if db_size < 1000:
            print("⚠ WARNING: Database is very small! Data may not have been saved properly.")

    def _update_pattern(self, characteristics: Dict, result: str):
        """Update pattern statistics in database"""
        piece_type = characteristics['piece_type']
        move_category = characteristics['move_category']
        distance = characteristics['distance_from_start']
        phase = characteristics['game_phase']

        # Get current stats
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ?
              AND distance_from_start = ? AND game_phase = ?
        ''', (piece_type, move_category, distance, phase))

        row = self.cursor.fetchone()

        if row:
            times_seen, won, lost, drawn = row
        else:
            times_seen, won, lost, drawn = 0, 0, 0, 0

        # Update counters
        times_seen += 1

        if result == 'win':
            won += 1
        elif result == 'loss':
            lost += 1
        else:
            drawn += 1

        # Calculate metrics
        total_games = won + lost + drawn
        win_rate = won / total_games if total_games > 0 else 0.0

        # Confidence increases with more observations (same as LearnableMovePrioritizer)
        confidence = min(1.0, total_games / 50.0)  # Max confidence at 50+ games

        # Calculate differential score (same as LearnableMovePrioritizer)
        # Win = +1050 avg, Loss = -800 avg, Draw = 0
        # This matches the differential scoring system used in training
        total_score = won * 1050 + drawn * 0 - lost * 800
        avg_score = total_score / total_games if total_games > 0 else 0.0

        # Priority score: DIFFERENTIAL SCORE-BASED (same as LearnableMovePrioritizer)
        # Normalize -1500 to +1600 → 0 to 100
        normalized_score = (avg_score + 1500) / 31  # Maps score range to 0-100
        priority_score = normalized_score * confidence  # 0-100, confidence-weighted

        # Insert or update (use ON CONFLICT like LearnableMovePrioritizer)
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase)
            DO UPDATE SET
                times_seen = ?,
                games_won = ?,
                games_lost = ?,
                games_drawn = ?,
                win_rate = ?,
                total_score = ?,
                avg_score = ?,
                confidence = ?,
                priority_score = ?,
                updated_at = datetime('now')
        ''', (piece_type, move_category, distance, phase,
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score,
              # ON CONFLICT DO UPDATE values
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score))

    def _show_statistics(self):
        """Show learned pattern statistics"""
        self.cursor.execute('''
            SELECT COUNT(*), SUM(times_seen), AVG(confidence), AVG(win_rate)
            FROM learned_move_patterns
        ''')

        count, total_seen, avg_conf, avg_wr = self.cursor.fetchone()

        print(f"\nLearning Statistics:")
        print(f"  Unique Patterns: {count}")
        print(f"  Total Observations: {total_seen:,}")
        print(f"  Average Confidence: {avg_conf:.2f}")
        print(f"  Average Win Rate: {avg_wr:.1%}")

        # Show top patterns
        print(f"\nTop 10 Patterns by Priority:")
        print(f"{'Piece':<10} {'Category':<15} {'Phase':<12} {'Seen':>8} {'WR':>6} {'Conf':>6} {'Pri':>6}")
        print("-" * 70)

        self.cursor.execute('''
            SELECT piece_type, move_category, game_phase,
                   times_seen, win_rate, confidence, priority_score
            FROM learned_move_patterns
            ORDER BY priority_score DESC
            LIMIT 10
        ''')

        for row in self.cursor.fetchall():
            piece, cat, phase, seen, wr, conf, pri = row
            print(f"{piece:<10} {cat:<15} {phase:<12} {seen:>8,} {wr:>6.1%} {conf:>6.2f} {pri:>6.1f}")

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Import chess patterns from PGN file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all games from a PGN file
  python chess_pattern_ai/import_pgn_patterns.py games.pgn

  # Import first 10,000 games only
  python chess_pattern_ai/import_pgn_patterns.py --limit 10000 games.pgn

  # Import to specific database
  python chess_pattern_ai/import_pgn_patterns.py --db chess_training.db games.pgn

  # Learn only from white's moves
  python chess_pattern_ai/import_pgn_patterns.py --color white games.pgn
        """
    )

    parser.add_argument('pgn_file', help='Path to PGN file')
    parser.add_argument('--db', default='headless_training.db',
                       help='Database path (default: headless_training.db)')
    parser.add_argument('--limit', type=int, help='Maximum games to process')
    parser.add_argument('--color', choices=['white', 'black'],
                       help='Only learn from one color')

    args = parser.parse_args()

    # Check if PGN file exists
    import os
    if not os.path.exists(args.pgn_file):
        print(f"Error: PGN file not found: {args.pgn_file}")
        print("\nTry:")
        print(f"  ls *.pgn                    # List PGN files in current directory")
        print(f"  find ~ -name '*.pgn' 2>/dev/null  # Find PGN files in home directory")
        sys.exit(1)

    # Import patterns
    importer = PGNPatternImporter(args.db)

    try:
        importer.import_games(args.pgn_file, args.limit, args.color)
    finally:
        importer.close()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nYour AI training database has been updated: {args.db}")
    print("\nNext steps:")
    print("  1. Run training to see improved performance")
    print("  2. The AI will use learned patterns from real games")
    print("  3. Consider importing more games to improve further")


if __name__ == '__main__':
    main()
