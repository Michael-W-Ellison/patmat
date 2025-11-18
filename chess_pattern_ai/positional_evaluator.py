#!/usr/bin/env python3
"""
Positional Evaluator - Phase 4.3 Implementation
Discovers positional patterns from pure observation:
- Piece centralization (distance from center)
- Piece coordination (mutual protection)
- Space control (squares controlled)

No hardcoded chess knowledge - all patterns discovered from game data.
"""

import sqlite3
import logging
import chess
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PositionalPattern:
    """Discovered positional pattern"""
    pattern_type: str  # 'centralization', 'coordination', 'space'
    description: str
    value_estimate: float
    frequency: int
    confidence: float


class PositionalEvaluator:
    """Discovers and evaluates positional patterns from game data"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None

        # Discovered weights (will be loaded from database)
        self.centralization_weight = 0.0
        self.coordination_weight = 0.0
        self.space_weight = 0.0

        # Center squares for distance calculation
        self.center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}

    def _connect(self):
        """Connect to database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def discover_positional_patterns(self, num_positions: int = 3000) -> Dict[str, List[PositionalPattern]]:
        """
        Discover fine-grained positional patterns by analyzing positions.
        FIXED: Changed from coarse-grained aggregation to fine-grained piece-specific patterns.

        Returns:
            Dictionary with 'centralization', 'coordination', 'space' patterns
        """
        logger.info("=" * 80)
        logger.info("DISCOVERING FINE-GRAINED POSITIONAL PATTERNS")
        logger.info("=" * 80)

        self._connect()

        # FIXED: Increased sample size from 3000 to 10000 for better pattern discovery
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.fen_after, m.move_number, g.result
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE g.result IN ('1-0', '0-1')
                AND m.move_number >= 10
                AND m.move_number <= 40
            ORDER BY RANDOM()
            LIMIT ?
        """, (10000,))

        positions = cursor.fetchall()
        logger.info(f"Analyzing positional patterns in {len(positions)} positions...")

        # FIXED: Track fine-grained patterns instead of aggregate statistics
        # Pattern format: "{piece_type}_on_{square}_{phase}_{attribute}"
        positional_patterns = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for pos in positions:
            fen = pos['fen_after']
            move_num = pos['move_number']
            result = pos['result']

            try:
                # Determine game phase
                if move_num < 15:
                    phase = 'opening'
                elif move_num < 30:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

                board = chess.Board(fen)

                # FIXED: Analyze each piece individually instead of aggregating
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece is None or piece.piece_type == chess.PAWN:
                        continue

                    piece_type = piece.symbol().upper()
                    square_name = chess.square_name(square)
                    piece_side = 'white' if piece.color == chess.WHITE else 'black'

                    # Calculate piece-specific positional attributes
                    file_idx = chess.square_file(square)
                    rank_idx = chess.square_rank(square)

                    # Centralization: distance to center
                    min_distance = min(
                        max(abs(file_idx - 3), abs(rank_idx - 3)),
                        max(abs(file_idx - 4), abs(rank_idx - 3)),
                        max(abs(file_idx - 3), abs(rank_idx - 4)),
                        max(abs(file_idx - 4), abs(rank_idx - 4)),
                    )
                    centralization = 4 - min_distance  # 0-4, higher is more central

                    # Protection: is piece defended?
                    defenders = board.attackers(piece.color, square)
                    is_protected = len([d for d in defenders if d != square]) > 0

                    # Control: how many squares does it attack?
                    attacks = board.attacks(square)
                    control_count = len(attacks)

                    # Create fine-grained pattern signatures
                    # FIXED: Pattern includes piece type, square, phase, and attributes
                    pattern_sig = f"{piece_type}_on_{square_name}_{phase}_central{centralization}"
                    positional_patterns[pattern_sig]['wins' if (piece_side == 'white' and result == '1-0') or (piece_side == 'black' and result == '0-1') else 'losses'] += 1

                    if is_protected:
                        protection_sig = f"{piece_type}_on_{square_name}_{phase}_protected"
                        positional_patterns[protection_sig]['wins' if (piece_side == 'white' and result == '1-0') or (piece_side == 'black' and result == '0-1') else 'losses'] += 1

                    # Control patterns (binned by ranges for manageability)
                    control_bin = (control_count // 3) * 3  # Bin by 3s
                    control_sig = f"{piece_type}_controls_{control_bin}_squares_{phase}"
                    positional_patterns[control_sig]['wins' if (piece_side == 'white' and result == '1-0') or (piece_side == 'black' and result == '0-1') else 'losses'] += 1

            except Exception as e:
                logger.debug(f"Error analyzing position {fen}: {e}")
                continue

        # Analyze discovered patterns
        logger.info(f"\nFINE-GRAINED POSITIONAL PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        significant_patterns = []
        total_observations = 0

        for pattern_sig, stats in positional_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 10:  # Minimum sample size
                win_rate = stats['wins'] / total
                value_estimate = (win_rate - 0.5) * 2.0

                significant_patterns.append({
                    'signature': pattern_sig,
                    'frequency': total,
                    'win_rate': win_rate,
                    'value': value_estimate
                })
                total_observations += total

        # Show top patterns
        significant_patterns.sort(key=lambda x: x['frequency'], reverse=True)
        logger.info(f"\nTop 30 most frequent positional patterns:")
        for pattern in significant_patterns[:30]:
            logger.info(f"  {pattern['signature']}: {pattern['frequency']} occurrences, " +
                       f"win rate: {pattern['win_rate']:.1%}, value: {pattern['value']:+.3f}")

        # Create pattern objects for backward compatibility
        patterns = {
            'centralization': [],
            'coordination': [],
            'space': []
        }

        # Group patterns by type
        for pattern_dict in significant_patterns:
            sig = pattern_dict['signature']
            if 'central' in sig:
                pattern_type = 'centralization'
            elif 'protected' in sig:
                pattern_type = 'coordination'
            elif 'controls' in sig:
                pattern_type = 'space'
            else:
                continue

            patterns[pattern_type].append(PositionalPattern(
                pattern_type=pattern_type,
                description=sig,
                value_estimate=pattern_dict['value'],
                frequency=pattern_dict['frequency'],
                confidence=min(pattern_dict['frequency'] / 100, 1.0)
            ))

        logger.info(f"\nDISCOVERY SUMMARY:")
        logger.info(f"  Total unique positional patterns: {len(positional_patterns)}")
        logger.info(f"  Patterns with 10+ observations: {len(significant_patterns)}")
        logger.info(f"  - Centralization patterns: {len(patterns['centralization'])}")
        logger.info(f"  - Coordination patterns: {len(patterns['coordination'])}")
        logger.info(f"  - Space control patterns: {len(patterns['space'])}")
        logger.info(f"  Total observations: {total_observations}")

        # Calculate and store weights
        self._calculate_positional_weights(patterns)
        self._store_positional_patterns(patterns)
        self._store_positional_patterns_detailed(positional_patterns)

        return patterns

    def _calculate_centralization(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate piece centralization score.
        Geometric definition: sum of (max_distance - distance_to_center) for all pieces.
        """
        centralization = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece and piece.color == color and piece.piece_type != chess.PAWN:
                # Calculate minimum distance to any center square
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)

                # Distance to center (approximate: use Chebyshev distance to d4/e4/d5/e5)
                min_distance = min(
                    max(abs(file_idx - 3), abs(rank_idx - 3)),  # To d4
                    max(abs(file_idx - 4), abs(rank_idx - 3)),  # To e4
                    max(abs(file_idx - 3), abs(rank_idx - 4)),  # To d5
                    max(abs(file_idx - 4), abs(rank_idx - 4)),  # To e5
                )

                # Max distance is 4 (corner to center)
                # Score: closer to center = higher score
                centralization += (4 - min_distance)

        return centralization

    def _calculate_coordination(self, board: chess.Board, color: chess.Color) -> int:
        """
        Calculate piece coordination score.
        Geometric definition: count of pieces that are protected by friendly pieces.
        """
        protected_count = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece and piece.color == color:
                # Check if this square is attacked by friendly pieces
                defenders = board.attackers(color, square)

                # Don't count self-attack
                if len(defenders) > 0:
                    # Check if any defender is actually a different piece
                    for defender_sq in defenders:
                        if defender_sq != square:
                            protected_count += 1
                            break

        return protected_count

    def _calculate_space_control(self, board: chess.Board, color: chess.Color) -> int:
        """
        Calculate space control.
        Geometric definition: count of squares controlled/attacked by this side.
        """
        controlled_squares = set()

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece and piece.color == color:
                # Add all squares this piece attacks
                attacks = board.attacks(square)
                controlled_squares.update(attacks)

        return len(controlled_squares)

    def _analyze_centralization_correlation(self, stats: Dict) -> List[PositionalPattern]:
        """Analyze correlation between centralization and winning"""
        patterns = []

        logger.info("\nCentralization Analysis:")
        logger.info("-" * 60)

        for central_bin in sorted(stats.keys()):
            wins = stats[central_bin]['wins']
            losses = stats[central_bin]['losses']
            total = wins + losses

            if total > 10:
                win_rate = wins / total
                logger.info(f"  {central_bin:+d} centralization: {wins}/{total} wins ({win_rate:.1%})")

                value_estimate = (win_rate - 0.5) * 20.0
                if central_bin != 0:
                    value_estimate = value_estimate / abs(central_bin)

                patterns.append(PositionalPattern(
                    pattern_type='centralization',
                    description=f'{abs(central_bin)} centralization advantage',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _analyze_coordination_correlation(self, stats: Dict) -> List[PositionalPattern]:
        """Analyze correlation between coordination and winning"""
        patterns = []

        logger.info("\nCoordination Analysis:")
        logger.info("-" * 60)

        for coord_bin in sorted(stats.keys()):
            wins = stats[coord_bin]['wins']
            losses = stats[coord_bin]['losses']
            total = wins + losses

            if total > 10:
                win_rate = wins / total
                logger.info(f"  {coord_bin:+d} coordination: {wins}/{total} wins ({win_rate:.1%})")

                value_estimate = (win_rate - 0.5) * 20.0
                if coord_bin != 0:
                    value_estimate = value_estimate / abs(coord_bin)

                patterns.append(PositionalPattern(
                    pattern_type='coordination',
                    description=f'{abs(coord_bin)} coordination advantage',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _analyze_space_correlation(self, stats: Dict) -> List[PositionalPattern]:
        """Analyze correlation between space control and winning"""
        patterns = []

        logger.info("\nSpace Control Analysis:")
        logger.info("-" * 60)

        for space_bin in sorted(stats.keys()):
            wins = stats[space_bin]['wins']
            losses = stats[space_bin]['losses']
            total = wins + losses

            if total > 10:
                win_rate = wins / total
                logger.info(f"  {space_bin:+d} space control: {wins}/{total} wins ({win_rate:.1%})")

                value_estimate = (win_rate - 0.5) * 20.0
                if space_bin != 0:
                    value_estimate = value_estimate / abs(space_bin)

                patterns.append(PositionalPattern(
                    pattern_type='space',
                    description=f'{abs(space_bin)} space advantage',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _calculate_positional_weights(self, patterns: Dict[str, List[PositionalPattern]]):
        """Calculate evaluation weights from discovered patterns"""

        # Centralization weight
        central_patterns = patterns['centralization']
        if central_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in central_patterns if p.value_estimate > 0)
            total_confidence = sum(p.confidence for p in central_patterns if p.value_estimate > 0)
            self.centralization_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        # Coordination weight
        coord_patterns = patterns['coordination']
        if coord_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in coord_patterns if p.value_estimate > 0)
            total_confidence = sum(p.confidence for p in coord_patterns if p.value_estimate > 0)
            self.coordination_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        # Space weight
        space_patterns = patterns['space']
        if space_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in space_patterns if p.value_estimate > 0)
            total_confidence = sum(p.confidence for p in space_patterns if p.value_estimate > 0)
            self.space_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        logger.info("\n" + "=" * 80)
        logger.info("DISCOVERED POSITIONAL WEIGHTS")
        logger.info("=" * 80)
        logger.info(f"Centralization weight: {self.centralization_weight:+.4f} per unit")
        logger.info(f"Coordination weight:   {self.coordination_weight:+.4f} per protected piece")
        logger.info(f"Space weight:          {self.space_weight:+.4f} per controlled square")
        logger.info("=" * 80)

    def _store_positional_patterns_detailed(self, positional_patterns: Dict):
        """Store fine-grained positional patterns in database"""
        cursor = self.conn.cursor()

        # Create table for fine-grained patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_positional_patterns_detailed (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_signature TEXT UNIQUE,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                value_estimate REAL,
                frequency INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear old patterns
        cursor.execute("DELETE FROM discovered_positional_patterns_detailed")

        # Store each pattern
        patterns_stored = 0
        for pattern_sig, stats in positional_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 10:  # Only store significant patterns
                win_rate = stats['wins'] / total
                value_estimate = (win_rate - 0.5) * 2.0

                cursor.execute("""
                    INSERT OR REPLACE INTO discovered_positional_patterns_detailed
                    (pattern_signature, wins, losses, win_rate, value_estimate, frequency)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pattern_sig, stats['wins'], stats['losses'], win_rate, value_estimate, total))

                patterns_stored += 1

        self.conn.commit()
        logger.info(f"✅ Stored {patterns_stored} fine-grained positional patterns in database")

    def _store_positional_patterns(self, patterns: Dict[str, List[PositionalPattern]]):
        """Store discovered pattern summary in database"""
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_positional_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                value_estimate REAL,
                frequency INTEGER,
                confidence REAL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_positional_weights (
                id INTEGER PRIMARY KEY,
                centralization_weight REAL,
                coordination_weight REAL,
                space_weight REAL,
                observation_count INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear old data
        cursor.execute("DELETE FROM discovered_positional_patterns")
        cursor.execute("DELETE FROM discovered_positional_weights")

        # Store patterns
        total_patterns = 0
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                cursor.execute("""
                    INSERT INTO discovered_positional_patterns
                    (pattern_type, description, value_estimate, frequency, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (pattern.pattern_type, pattern.description, pattern.value_estimate,
                      pattern.frequency, pattern.confidence))
                total_patterns += 1

        # Store weights
        cursor.execute("""
            INSERT INTO discovered_positional_weights
            (id, centralization_weight, coordination_weight, space_weight, observation_count)
            VALUES (1, ?, ?, ?, ?)
        """, (self.centralization_weight, self.coordination_weight, self.space_weight, total_patterns))

        self.conn.commit()
        logger.info(f"✅ Stored {total_patterns} positional pattern summaries in database")

    def _load_positional_weights(self):
        """Load discovered positional weights from database"""
        self._connect()
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT centralization_weight, coordination_weight, space_weight
                FROM discovered_positional_weights
                WHERE id = 1
            """)

            row = cursor.fetchone()
            if row:
                self.centralization_weight = row['centralization_weight']
                self.coordination_weight = row['coordination_weight']
                self.space_weight = row['space_weight']
                logger.info(f"✅ Loaded positional weights: central={self.centralization_weight:.4f}, " +
                           f"coord={self.coordination_weight:.4f}, space={self.space_weight:.4f}")
            else:
                logger.warning("⚠️  No positional weights found in database, run discovery first")
        except sqlite3.OperationalError:
            logger.warning("⚠️  Positional weights table not found, run discovery first")

    def evaluate_position(self, fen: str) -> float:
        """
        Evaluate positional factors.

        Returns:
            Positional score (positive = good, negative = bad)
        """
        try:
            board = chess.Board(fen)
            score = 0.0

            # Calculate factors for both sides
            white_central = self._calculate_centralization(board, chess.WHITE)
            white_coord = self._calculate_coordination(board, chess.WHITE)
            white_space = self._calculate_space_control(board, chess.WHITE)

            black_central = self._calculate_centralization(board, chess.BLACK)
            black_coord = self._calculate_coordination(board, chess.BLACK)
            black_space = self._calculate_space_control(board, chess.BLACK)

            # Apply discovered weights (from white's perspective)
            score += (white_central - black_central) * self.centralization_weight
            score += (white_coord - black_coord) * self.coordination_weight
            score += (white_space - black_space) * self.space_weight

            return score

        except Exception as e:
            logger.debug(f"Error evaluating position: {e}")
            return 0.0


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Discover and evaluate positional patterns')
    parser.add_argument('--db', type=str, default='rule_discovery.db', help='Database path')
    parser.add_argument('--discover', action='store_true', help='Run positional pattern discovery')
    parser.add_argument('--positions', type=int, default=3000, help='Number of positions to analyze')

    args = parser.parse_args()

    evaluator = PositionalEvaluator(args.db)

    try:
        if args.discover:
            # Run discovery
            patterns = evaluator.discover_positional_patterns(num_positions=args.positions)

            print("\n" + "=" * 80)
            print("POSITIONAL PATTERN DISCOVERY COMPLETE")
            print("=" * 80)
            print(f"\nCentralization patterns: {len(patterns['centralization'])}")
            print(f"Coordination patterns: {len(patterns['coordination'])}")
            print(f"Space control patterns: {len(patterns['space'])}")
            print("\n" + "=" * 80)
        else:
            # Load and test
            evaluator._load_positional_weights()

            print("\n" + "=" * 80)
            print("TESTING POSITIONAL EVALUATOR")
            print("=" * 80)

            # Test positions
            test_cases = [
                ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
                ("Centralized knights", "rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1"),
            ]

            for name, fen in test_cases:
                score = evaluator.evaluate_position(fen)
                print(f"\n{name}:")
                print(f"  FEN: {fen}")
                print(f"  Positional score: {score:+.2f}")

            print("\n" + "=" * 80)
            print("✅ POSITIONAL EVALUATOR TESTS COMPLETE")
            print("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()
