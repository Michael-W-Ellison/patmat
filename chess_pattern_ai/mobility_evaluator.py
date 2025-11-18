#!/usr/bin/env python3
"""
Mobility Evaluator - Phase 2.2: Mobility Pattern Discovery
Discovers that positions with more legal moves correlate with winning.
No hardcoded chess knowledge - pure statistical learning.
"""

import sqlite3
import logging
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, '/home/ubuntu/Desktop/chess/chess_pattern_analyzer')
from discovered_chess_engine import DiscoveredChessEngine, BoardState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MobilityDiscovery:
    """Discovered mobility patterns"""
    avg_mobility_winning: float
    avg_mobility_losing: float
    correlation_strength: float
    mobility_weight: float
    observation_count: int


class MobilityEvaluator:
    """Discovers mobility patterns and evaluates positions based on available moves"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.engine = DiscoveredChessEngine(db_path)
        self.mobility_weight = 0.0

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def discover_mobility_correlation(self) -> MobilityDiscovery:
        """
        Discover fine-grained mobility patterns: which pieces on which squares have good mobility.
        FIXED: Changed from coarse-grained (total move counts) to fine-grained patterns.
        Method: Analyze positions from winning vs losing games.
        """
        logger.info("="*80)
        logger.info("DISCOVERING FINE-GRAINED MOBILITY PATTERNS")
        logger.info("="*80)

        # FIXED: Increased sample size from 1000 to 5000 games for better pattern discovery
        self.cursor.execute('''
            SELECT game_id, result
            FROM games
            WHERE result IN ('1-0', '0-1')
            LIMIT 5000
        ''')

        games = self.cursor.fetchall()
        logger.info(f"Analyzing mobility patterns in {len(games)} decisive games...")

        # FIXED: Track fine-grained mobility patterns instead of aggregate statistics
        # Pattern format: "{piece_type}_on_{square}_controls_{num_squares}_squares_{phase}"
        mobility_patterns = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'total_value': 0.0
        })

        import chess

        for game_id, result in games:
            # Sample positions from the game
            self.cursor.execute('''
                SELECT fen_before, move_number
                FROM moves
                WHERE game_id = ? AND move_number BETWEEN 10 AND 40
                ORDER BY RANDOM()
                LIMIT 5
            ''', (game_id,))

            positions = self.cursor.fetchall()

            for fen, move_num in positions:
                try:
                    # Determine game phase
                    if move_num < 15:
                        phase = 'opening'
                    elif move_num < 30:
                        phase = 'middlegame'
                    else:
                        phase = 'endgame'

                    board = chess.Board(fen)

                    # Analyze mobility for each piece
                    for square in chess.SQUARES:
                        piece = board.piece_at(square)
                        if piece is None:
                            continue

                        # Count squares this piece can move to
                        piece_moves = [m for m in board.legal_moves if m.from_square == square]
                        num_squares = len(piece_moves)

                        if num_squares == 0:
                            continue

                        # Create fine-grained pattern signature
                        piece_type = piece.symbol().upper()
                        square_name = chess.square_name(square)

                        # FIXED: Fine-grained pattern instead of just counting total moves
                        pattern_sig = f"{piece_type}_on_{square_name}_controls_{num_squares}_squares_{phase}"

                        # Determine if this is a winning or losing position for the piece's side
                        side_to_move = 'white' if fen.split()[1] == 'w' else 'black'
                        piece_side = 'white' if piece.color == chess.WHITE else 'black'

                        # Track pattern outcome
                        is_winning = (piece_side == 'white' and result == '1-0') or \
                                    (piece_side == 'black' and result == '0-1')

                        if is_winning:
                            mobility_patterns[pattern_sig]['wins'] += 1
                        else:
                            mobility_patterns[pattern_sig]['losses'] += 1

                except Exception as e:
                    logger.debug(f"Error processing position: {e}")
                    continue

        # Analyze discovered patterns
        logger.info(f"\nFINE-GRAINED MOBILITY PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        significant_patterns = []
        total_observations = 0

        for pattern_sig, stats in mobility_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:  # Minimum sample size
                win_rate = stats['wins'] / total
                value_estimate = (win_rate - 0.5) * 2.0  # Convert to centipawn estimate

                significant_patterns.append({
                    'signature': pattern_sig,
                    'frequency': total,
                    'win_rate': win_rate,
                    'value': value_estimate
                })
                total_observations += total

        # Show top patterns
        significant_patterns.sort(key=lambda x: x['frequency'], reverse=True)
        logger.info(f"\nTop 20 most frequent mobility patterns:")
        for pattern in significant_patterns[:20]:
            logger.info(f"  {pattern['signature']}: {pattern['frequency']} occurrences, " +
                       f"win rate: {pattern['win_rate']:.1%}, value: {pattern['value']:+.3f}")

        # Calculate aggregate statistics for backward compatibility
        avg_winning = sum(s['wins'] for s in mobility_patterns.values()) / max(len(mobility_patterns), 1)
        avg_losing = sum(s['losses'] for s in mobility_patterns.values()) / max(len(mobility_patterns), 1)

        if avg_losing > 0:
            correlation = (avg_winning - avg_losing) / avg_losing
        else:
            correlation = 0.0

        # Determine weight based on discovered patterns
        mobility_weight = 0.05  # Base weight per move

        discovery = MobilityDiscovery(
            avg_mobility_winning=avg_winning,
            avg_mobility_losing=avg_losing,
            correlation_strength=correlation,
            mobility_weight=mobility_weight,
            observation_count=total_observations
        )

        logger.info(f"\nDISCOVERY SUMMARY:")
        logger.info(f"  Total unique mobility patterns: {len(mobility_patterns)}")
        logger.info(f"  Patterns with 5+ observations: {len(significant_patterns)}")
        logger.info(f"  Total observations: {total_observations}")
        logger.info(f"  Mobility weight: {mobility_weight:.3f} points per move")

        # Store fine-grained patterns in database
        self._store_mobility_patterns(mobility_patterns)
        self._store_mobility_discovery(discovery)

        self.mobility_weight = mobility_weight

        return discovery

    def analyze_mobility_distribution(self):
        """Analyze distribution of mobility in different game outcomes"""
        logger.info("\n" + "="*80)
        logger.info("MOBILITY DISTRIBUTION ANALYSIS")
        logger.info("="*80)

        # Get sample positions and their mobility
        self.cursor.execute('''
            SELECT g.result, m.fen_before, m.move_number
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE m.move_number BETWEEN 15 AND 25
            ORDER BY RANDOM()
            LIMIT 500
        ''')

        positions = self.cursor.fetchall()

        mobility_by_outcome = defaultdict(list)

        for result, fen, move_num in positions:
            try:
                board_state = BoardState.from_fen(fen)
                legal_moves = self.engine.generate_legal_moves(board_state)
                mobility = len(legal_moves)

                mobility_by_outcome[result].append(mobility)

            except Exception as e:
                continue

        logger.info("\nMOBILITY BY GAME OUTCOME:")
        logger.info("-" * 80)

        for outcome in ['1-0', '0-1', '1/2-1/2']:
            if mobility_by_outcome[outcome]:
                avg = sum(mobility_by_outcome[outcome]) / len(mobility_by_outcome[outcome])
                logger.info(f"{outcome}: {avg:.2f} avg moves ({len(mobility_by_outcome[outcome])} samples)")

    def discover_mobility_thresholds(self):
        """Discover critical mobility thresholds (e.g., <10 moves = losing)"""
        logger.info("\n" + "="*80)
        logger.info("MOBILITY THRESHOLD DISCOVERY")
        logger.info("="*80)

        # Get positions with known outcomes
        self.cursor.execute('''
            SELECT g.result, m.fen_before
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE g.result IN ('1-0', '0-1') AND m.move_number BETWEEN 15 AND 30
            ORDER BY RANDOM()
            LIMIT 1000
        ''')

        positions = self.cursor.fetchall()

        # Group by mobility level
        mobility_buckets = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for result, fen in positions:
            try:
                board_state = BoardState.from_fen(fen)
                legal_moves = self.engine.generate_legal_moves(board_state)
                mobility = len(legal_moves)

                # Bucket mobility
                if mobility < 10:
                    bucket = 'very_low'
                elif mobility < 20:
                    bucket = 'low'
                elif mobility < 30:
                    bucket = 'medium'
                elif mobility < 40:
                    bucket = 'high'
                else:
                    bucket = 'very_high'

                # Track wins/losses
                side_to_move = 'white' if fen.split()[1] == 'w' else 'black'
                is_winning = (side_to_move == 'white' and result == '1-0') or \
                            (side_to_move == 'black' and result == '0-1')

                if is_winning:
                    mobility_buckets[bucket]['wins'] += 1
                else:
                    mobility_buckets[bucket]['losses'] += 1

            except Exception as e:
                continue

        logger.info("\nWIN RATE BY MOBILITY LEVEL:")
        logger.info("-" * 80)

        bucket_order = ['very_low', 'low', 'medium', 'high', 'very_high']
        bucket_ranges = {
            'very_low': '<10 moves',
            'low': '10-20 moves',
            'medium': '20-30 moves',
            'high': '30-40 moves',
            'very_high': '>40 moves'
        }

        for bucket in bucket_order:
            stats = mobility_buckets[bucket]
            total = stats['wins'] + stats['losses']
            if total > 0:
                win_rate = stats['wins'] / total * 100
                logger.info(f"{bucket_ranges[bucket]}: {win_rate:.1f}% win rate ({stats['wins']}W / {stats['losses']}L)")

    def evaluate_mobility(self, fen: str) -> float:
        """
        Evaluate mobility for a position.
        Returns RELATIVE mobility advantage (white_mobility - black_mobility).
        Phase-specific weighting is applied by TemporalEvaluator.
        """
        if not self.mobility_weight:
            # Load from database
            self._load_mobility_weight()

        # Parse FEN to get current side to move
        import chess
        board = chess.Board(fen)

        # Calculate white's mobility
        if board.turn == chess.WHITE:
            white_mobility = len(list(board.legal_moves))
            # Switch turn to calculate black's mobility
            board.turn = chess.BLACK
            black_mobility = len(list(board.legal_moves))
        else:
            black_mobility = len(list(board.legal_moves))
            # Switch turn to calculate white's mobility
            board.turn = chess.WHITE
            white_mobility = len(list(board.legal_moves))

        # CRITICAL FIX: Return RELATIVE mobility (white - black), not absolute!
        # This ensures equal positions score 0, not +20
        mobility_advantage = white_mobility - black_mobility

        return mobility_advantage

    def _store_mobility_patterns(self, mobility_patterns: Dict):
        """Store fine-grained mobility patterns in database"""

        # Create table for fine-grained patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_mobility_patterns_detailed (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_signature TEXT UNIQUE,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                value_estimate REAL,
                frequency INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Clear old patterns
        self.cursor.execute('DELETE FROM discovered_mobility_patterns_detailed')

        # Store each pattern
        patterns_stored = 0
        for pattern_sig, stats in mobility_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:  # Only store significant patterns
                win_rate = stats['wins'] / total
                value_estimate = (win_rate - 0.5) * 2.0

                self.cursor.execute('''
                    INSERT OR REPLACE INTO discovered_mobility_patterns_detailed
                    (pattern_signature, wins, losses, win_rate, value_estimate, frequency)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pattern_sig, stats['wins'], stats['losses'], win_rate, value_estimate, total))

                patterns_stored += 1

        self.conn.commit()
        logger.info(f"✅ Stored {patterns_stored} fine-grained mobility patterns in database")

    def _store_mobility_discovery(self, discovery: MobilityDiscovery):
        """Store discovered mobility summary in database"""

        # Create table if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_mobility_patterns (
                id INTEGER PRIMARY KEY,
                avg_mobility_winning REAL,
                avg_mobility_losing REAL,
                correlation_strength REAL,
                mobility_weight REAL,
                observation_count INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.cursor.execute('''
            INSERT INTO discovered_mobility_patterns
            (id, avg_mobility_winning, avg_mobility_losing, correlation_strength,
             mobility_weight, observation_count)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                avg_mobility_winning = excluded.avg_mobility_winning,
                avg_mobility_losing = excluded.avg_mobility_losing,
                correlation_strength = excluded.correlation_strength,
                mobility_weight = excluded.mobility_weight,
                observation_count = excluded.observation_count,
                discovered_at = CURRENT_TIMESTAMP
        ''', (discovery.avg_mobility_winning, discovery.avg_mobility_losing,
              discovery.correlation_strength, discovery.mobility_weight,
              discovery.observation_count))

        self.conn.commit()
        logger.info("\n✅ Mobility summary stored in database: discovered_mobility_patterns")

    def _load_mobility_weight(self):
        """Load mobility weight from database"""
        try:
            self.cursor.execute('''
                SELECT mobility_weight
                FROM discovered_mobility_patterns
                WHERE id = 1
            ''')

            result = self.cursor.fetchone()
            if result:
                self.mobility_weight = result[0]
                logger.info(f"Loaded mobility weight: {self.mobility_weight}")
        except:
            logger.warning("No mobility patterns found in database")

    def close(self):
        """Close connections"""
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Discover mobility patterns from observation')
    parser.add_argument('--db', type=str, default='rule_discovery.db', help='Database path')

    args = parser.parse_args()

    evaluator = MobilityEvaluator(args.db)

    try:
        # Run discovery process
        discovery = evaluator.discover_mobility_correlation()

        # Additional analyses
        evaluator.analyze_mobility_distribution()
        evaluator.discover_mobility_thresholds()

        print("\n" + "="*80)
        print("✅ MOBILITY PATTERN DISCOVERY COMPLETE")
        print("="*80)
        print(f"\nDiscovered correlation: {discovery.correlation_strength*100:+.1f}%")
        print(f"Mobility weight: {discovery.mobility_weight:.3f} points per move")
        print(f"Observations: {discovery.observation_count}")

        # Test evaluation
        print("\nTesting mobility evaluation on starting position:")
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        mobility_score = evaluator.evaluate_mobility(starting_fen)
        print(f"  Mobility score: {mobility_score:.2f}")

    except Exception as e:
        logger.error(f"Error during mobility discovery: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()
