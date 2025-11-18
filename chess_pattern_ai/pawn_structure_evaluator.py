#!/usr/bin/env python3
"""
Pawn Structure Evaluator - Phase 4.2 Implementation
Discovers pawn structure patterns from pure observation:
- Passed pawns (no enemy pawns blocking advancement)
- Doubled pawns (multiple pawns on same file)
- Isolated pawns (no friendly pawns on adjacent files)

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
class PawnStructurePattern:
    """Discovered pawn structure pattern"""
    structure_type: str  # 'passed', 'doubled', 'isolated'
    description: str
    value_estimate: float
    frequency: int
    confidence: float


class PawnStructureEvaluator:
    """Discovers and evaluates pawn structure patterns from game data"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None

        # Discovered weights (will be loaded from database)
        self.passed_pawn_weight = 0.0
        self.doubled_pawn_weight = 0.0
        self.isolated_pawn_weight = 0.0

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

    def discover_pawn_structures(self, num_positions: int = 5000) -> Dict[str, List[PawnStructurePattern]]:
        """
        Discover fine-grained pawn structure patterns by analyzing positions.
        FIXED: Changed from coarse-grained count differences to fine-grained file/rank/phase patterns.

        Returns:
            Dictionary with 'passed', 'doubled', 'isolated' patterns
        """
        logger.info("=" * 80)
        logger.info("DISCOVERING FINE-GRAINED PAWN STRUCTURE PATTERNS")
        logger.info("=" * 80)

        self._connect()

        # FIXED: Increased sample size from 5000 to 10000 for better pattern discovery
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.fen_after, m.move_number, g.result
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE g.result IN ('1-0', '0-1')
                AND m.move_number >= 15
                AND m.move_number <= 40
            ORDER BY RANDOM()
            LIMIT ?
        """, (10000,))

        positions = cursor.fetchall()
        logger.info(f"Analyzing pawn structures in {len(positions)} positions...")

        # FIXED: Track fine-grained patterns instead of aggregate count differences
        # Pattern format: "{structure_type}_on_{file}_rank_{rank}_{phase}"
        pawn_patterns = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for pos in positions:
            fen = pos['fen_after']
            move_num = pos['move_number']
            result = pos['result']

            try:
                # Determine game phase
                if move_num < 20:
                    phase = 'opening'
                elif move_num < 35:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

                board = chess.Board(fen)

                # FIXED: Analyze each pawn individually instead of aggregating counts
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece is None or piece.piece_type != chess.PAWN:
                        continue

                    file_idx = chess.square_file(square)
                    rank_idx = chess.square_rank(square)
                    file_name = chess.FILE_NAMES[file_idx]
                    pawn_side = 'white' if piece.color == chess.WHITE else 'black'

                    # Check if passed pawn
                    is_passed = self._is_passed_pawn(board, square, piece.color)
                    if is_passed:
                        # FIXED: Fine-grained pattern including file, rank, and phase
                        pattern_sig = f"passed_pawn_on_{file_name}_rank_{rank_idx}_{phase}"
                        pawn_patterns[pattern_sig]['wins' if (pawn_side == 'white' and result == '1-0') or (pawn_side == 'black' and result == '0-1') else 'losses'] += 1

                    # Check if doubled pawn
                    is_doubled = self._is_doubled_pawn(board, square, piece.color, file_idx)
                    if is_doubled:
                        pattern_sig = f"doubled_pawn_on_{file_name}_{phase}"
                        pawn_patterns[pattern_sig]['wins' if (pawn_side == 'white' and result == '1-0') or (pawn_side == 'black' and result == '0-1') else 'losses'] += 1

                    # Check if isolated pawn
                    is_isolated = self._is_isolated_pawn(board, piece.color, file_idx)
                    if is_isolated:
                        pattern_sig = f"isolated_pawn_on_{file_name}_{phase}"
                        pawn_patterns[pattern_sig]['wins' if (pawn_side == 'white' and result == '1-0') or (pawn_side == 'black' and result == '0-1') else 'losses'] += 1

            except Exception as e:
                logger.debug(f"Error analyzing position {fen}: {e}")
                continue

        # Analyze discovered patterns
        logger.info(f"\nFINE-GRAINED PAWN STRUCTURE PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        significant_patterns = []
        total_observations = 0

        for pattern_sig, stats in pawn_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:  # Minimum sample size
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
        logger.info(f"\nTop 30 most frequent pawn structure patterns:")
        for pattern in significant_patterns[:30]:
            logger.info(f"  {pattern['signature']}: {pattern['frequency']} occurrences, " +
                       f"win rate: {pattern['win_rate']:.1%}, value: {pattern['value']:+.3f}")

        # Create pattern objects for backward compatibility
        patterns = {
            'passed': [],
            'doubled': [],
            'isolated': []
        }

        # Group patterns by type
        for pattern_dict in significant_patterns:
            sig = pattern_dict['signature']
            if 'passed' in sig:
                pattern_type = 'passed'
            elif 'doubled' in sig:
                pattern_type = 'doubled'
            elif 'isolated' in sig:
                pattern_type = 'isolated'
            else:
                continue

            patterns[pattern_type].append(PawnStructurePattern(
                structure_type=pattern_type,
                description=sig,
                value_estimate=pattern_dict['value'],
                frequency=pattern_dict['frequency'],
                confidence=min(pattern_dict['frequency'] / 50, 1.0)
            ))

        logger.info(f"\nDISCOVERY SUMMARY:")
        logger.info(f"  Total unique pawn structure patterns: {len(pawn_patterns)}")
        logger.info(f"  Patterns with 5+ observations: {len(significant_patterns)}")
        logger.info(f"  - Passed pawn patterns: {len(patterns['passed'])}")
        logger.info(f"  - Doubled pawn patterns: {len(patterns['doubled'])}")
        logger.info(f"  - Isolated pawn patterns: {len(patterns['isolated'])}")
        logger.info(f"  Total observations: {total_observations}")

        # Calculate and store weights
        self._calculate_structure_weights(patterns)
        self._store_structure_patterns(patterns)
        self._store_structure_patterns_detailed(pawn_patterns)

        return patterns

    def _is_passed_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        """
        Check if a specific pawn is passed (geometric definition):
        No enemy pawns on same file or adjacent files ahead of this pawn.
        """
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        # Check files: same file and adjacent files
        files_to_check = [file_idx]
        if file_idx > 0:
            files_to_check.append(file_idx - 1)
        if file_idx < 7:
            files_to_check.append(file_idx + 1)

        # Define "ahead" based on color
        if color == chess.WHITE:
            ranks_ahead = range(rank_idx + 1, 8)
        else:
            ranks_ahead = range(0, rank_idx)

        # Check if any enemy pawns block advancement
        for check_file in files_to_check:
            for check_rank in ranks_ahead:
                check_square = chess.square(check_file, check_rank)
                check_piece = board.piece_at(check_square)

                if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color != color:
                    return False

        return True

    def _is_doubled_pawn(self, board: chess.Board, square: int, color: chess.Color, file_idx: int) -> bool:
        """
        Check if a specific pawn is doubled (geometric definition):
        Multiple pawns of same color on the same file.
        """
        pawns_on_file = 0
        for rank_idx in range(8):
            check_square = chess.square(file_idx, rank_idx)
            piece = board.piece_at(check_square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                pawns_on_file += 1

        return pawns_on_file > 1

    def _is_isolated_pawn(self, board: chess.Board, color: chess.Color, file_idx: int) -> bool:
        """
        Check if a pawn on this file is isolated (geometric definition):
        No friendly pawns on adjacent files.
        """
        # Check left file
        if file_idx > 0:
            for rank_idx in range(8):
                check_square = chess.square(file_idx - 1, rank_idx)
                check_piece = board.piece_at(check_square)
                if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == color:
                    return False

        # Check right file
        if file_idx < 7:
            for rank_idx in range(8):
                check_square = chess.square(file_idx + 1, rank_idx)
                check_piece = board.piece_at(check_square)
                if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == color:
                    return False

        return True

    def _count_passed_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """
        Count passed pawns (geometric definition):
        No enemy pawns on same file or adjacent files ahead of this pawn.
        """
        passed_count = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                if self._is_passed_pawn(board, square, color):
                    passed_count += 1

        return passed_count

    def _count_doubled_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """
        Count doubled pawns (geometric definition):
        Multiple pawns of same color on the same file.
        """
        doubled_count = 0

        for file_idx in range(8):
            pawns_on_file = 0

            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)

                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawns_on_file += 1

            # If more than 1 pawn on file, count extras as doubled
            if pawns_on_file > 1:
                doubled_count += (pawns_on_file - 1)

        return doubled_count

    def _count_isolated_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """
        Count isolated pawns (geometric definition):
        Pawn with no friendly pawns on adjacent files.
        """
        isolated_count = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                file_idx = chess.square_file(square)

                # Check adjacent files
                has_neighbor = False

                # Check left file
                if file_idx > 0:
                    for rank_idx in range(8):
                        check_square = chess.square(file_idx - 1, rank_idx)
                        check_piece = board.piece_at(check_square)
                        if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == color:
                            has_neighbor = True
                            break

                # Check right file
                if file_idx < 7 and not has_neighbor:
                    for rank_idx in range(8):
                        check_square = chess.square(file_idx + 1, rank_idx)
                        check_piece = board.piece_at(check_square)
                        if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == color:
                            has_neighbor = True
                            break

                if not has_neighbor:
                    isolated_count += 1

        return isolated_count

    def _analyze_passed_correlation(self, stats: Dict) -> List[PawnStructurePattern]:
        """Analyze correlation between passed pawns and winning"""
        patterns = []

        logger.info("\nPassed Pawn Analysis:")
        logger.info("-" * 60)

        for diff in sorted(stats.keys()):
            wins = stats[diff]['wins']
            losses = stats[diff]['losses']
            total = wins + losses

            if total > 10:  # Minimum sample size
                win_rate = wins / total
                logger.info(f"  {diff:+d} passed pawn advantage: {wins}/{total} wins ({win_rate:.1%})")

                # Convert win rate to value estimate (centered at 50%)
                value_estimate = (win_rate - 0.5) * 20.0  # Scale to reasonable range
                if diff != 0:
                    value_estimate = value_estimate / abs(diff)  # Per-pawn value

                patterns.append(PawnStructurePattern(
                    structure_type='passed',
                    description=f'{abs(diff)} passed pawn advantage',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _analyze_doubled_correlation(self, stats: Dict) -> List[PawnStructurePattern]:
        """Analyze correlation between doubled pawns and winning"""
        patterns = []

        logger.info("\nDoubled Pawn Analysis:")
        logger.info("-" * 60)

        for diff in sorted(stats.keys()):
            wins = stats[diff]['wins']
            losses = stats[diff]['losses']
            total = wins + losses

            if total > 10:
                win_rate = wins / total
                logger.info(f"  {diff:+d} doubled pawn difference: {wins}/{total} wins ({win_rate:.1%})")

                value_estimate = (win_rate - 0.5) * 20.0
                if diff != 0:
                    value_estimate = value_estimate / abs(diff)

                patterns.append(PawnStructurePattern(
                    structure_type='doubled',
                    description=f'{abs(diff)} doubled pawn difference',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _analyze_isolated_correlation(self, stats: Dict) -> List[PawnStructurePattern]:
        """Analyze correlation between isolated pawns and winning"""
        patterns = []

        logger.info("\nIsolated Pawn Analysis:")
        logger.info("-" * 60)

        for diff in sorted(stats.keys()):
            wins = stats[diff]['wins']
            losses = stats[diff]['losses']
            total = wins + losses

            if total > 10:
                win_rate = wins / total
                logger.info(f"  {diff:+d} isolated pawn difference: {wins}/{total} wins ({win_rate:.1%})")

                value_estimate = (win_rate - 0.5) * 20.0
                if diff != 0:
                    value_estimate = value_estimate / abs(diff)

                patterns.append(PawnStructurePattern(
                    structure_type='isolated',
                    description=f'{abs(diff)} isolated pawn difference',
                    value_estimate=value_estimate,
                    frequency=total,
                    confidence=min(total / 100, 1.0)
                ))

        return patterns

    def _calculate_structure_weights(self, patterns: Dict[str, List[PawnStructurePattern]]):
        """Calculate evaluation weights from discovered patterns"""

        # Passed pawn weight
        passed_patterns = patterns['passed']
        if passed_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in passed_patterns if p.value_estimate > 0)
            total_confidence = sum(p.confidence for p in passed_patterns if p.value_estimate > 0)
            self.passed_pawn_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        # Doubled pawn weight (negative - penalty)
        doubled_patterns = patterns['doubled']
        if doubled_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in doubled_patterns if p.value_estimate < 0)
            total_confidence = sum(p.confidence for p in doubled_patterns if p.value_estimate < 0)
            self.doubled_pawn_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        # Isolated pawn weight (negative - penalty)
        isolated_patterns = patterns['isolated']
        if isolated_patterns:
            weighted_sum = sum(p.value_estimate * p.confidence for p in isolated_patterns if p.value_estimate < 0)
            total_confidence = sum(p.confidence for p in isolated_patterns if p.value_estimate < 0)
            self.isolated_pawn_weight = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        logger.info("\n" + "=" * 80)
        logger.info("DISCOVERED PAWN STRUCTURE WEIGHTS")
        logger.info("=" * 80)
        logger.info(f"Passed pawn bonus:     {self.passed_pawn_weight:+.4f} per passed pawn")
        logger.info(f"Doubled pawn penalty:  {self.doubled_pawn_weight:+.4f} per doubled pawn")
        logger.info(f"Isolated pawn penalty: {self.isolated_pawn_weight:+.4f} per isolated pawn")
        logger.info("=" * 80)

    def _store_structure_patterns_detailed(self, pawn_patterns: Dict):
        """Store fine-grained pawn structure patterns in database"""
        cursor = self.conn.cursor()

        # Create table for fine-grained patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_pawn_structure_patterns_detailed (
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
        cursor.execute("DELETE FROM discovered_pawn_structure_patterns_detailed")

        # Store each pattern
        patterns_stored = 0
        for pattern_sig, stats in pawn_patterns.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:  # Only store significant patterns
                win_rate = stats['wins'] / total
                value_estimate = (win_rate - 0.5) * 2.0

                cursor.execute("""
                    INSERT OR REPLACE INTO discovered_pawn_structure_patterns_detailed
                    (pattern_signature, wins, losses, win_rate, value_estimate, frequency)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pattern_sig, stats['wins'], stats['losses'], win_rate, value_estimate, total))

                patterns_stored += 1

        self.conn.commit()
        logger.info(f"✅ Stored {patterns_stored} fine-grained pawn structure patterns in database")

    def _store_structure_patterns(self, patterns: Dict[str, List[PawnStructurePattern]]):
        """Store discovered pattern summary in database"""
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_pawn_structure_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_type TEXT,
                description TEXT,
                value_estimate REAL,
                frequency INTEGER,
                confidence REAL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_pawn_structure_weights (
                id INTEGER PRIMARY KEY,
                passed_pawn_weight REAL,
                doubled_pawn_weight REAL,
                isolated_pawn_weight REAL,
                observation_count INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear old data
        cursor.execute("DELETE FROM discovered_pawn_structure_patterns")
        cursor.execute("DELETE FROM discovered_pawn_structure_weights")

        # Store patterns
        total_patterns = 0
        for structure_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                cursor.execute("""
                    INSERT INTO discovered_pawn_structure_patterns
                    (structure_type, description, value_estimate, frequency, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (pattern.structure_type, pattern.description, pattern.value_estimate,
                      pattern.frequency, pattern.confidence))
                total_patterns += 1

        # Store weights
        cursor.execute("""
            INSERT INTO discovered_pawn_structure_weights
            (id, passed_pawn_weight, doubled_pawn_weight, isolated_pawn_weight, observation_count)
            VALUES (1, ?, ?, ?, ?)
        """, (self.passed_pawn_weight, self.doubled_pawn_weight, self.isolated_pawn_weight, total_patterns))

        self.conn.commit()
        logger.info(f"✅ Stored {total_patterns} pawn structure pattern summaries in database")

    def _load_structure_weights(self):
        """Load discovered pawn structure weights from database"""
        self._connect()
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT passed_pawn_weight, doubled_pawn_weight, isolated_pawn_weight
                FROM discovered_pawn_structure_weights
                WHERE id = 1
            """)

            row = cursor.fetchone()
            if row:
                self.passed_pawn_weight = row['passed_pawn_weight']
                self.doubled_pawn_weight = row['doubled_pawn_weight']
                self.isolated_pawn_weight = row['isolated_pawn_weight']
                logger.info(f"✅ Loaded structure weights: passed={self.passed_pawn_weight:.4f}, " +
                           f"doubled={self.doubled_pawn_weight:.4f}, isolated={self.isolated_pawn_weight:.4f}")
            else:
                logger.warning("⚠️  No structure weights found in database, run discovery first")
        except sqlite3.OperationalError:
            logger.warning("⚠️  Structure weights table not found, run discovery first")

    def evaluate_pawn_structure(self, fen: str) -> float:
        """
        Evaluate pawn structure based on discovered patterns.

        Returns:
            Pawn structure score (positive = good, negative = bad)
        """
        try:
            board = chess.Board(fen)
            score = 0.0

            # Count structures for both sides
            white_passed = self._count_passed_pawns(board, chess.WHITE)
            white_doubled = self._count_doubled_pawns(board, chess.WHITE)
            white_isolated = self._count_isolated_pawns(board, chess.WHITE)

            black_passed = self._count_passed_pawns(board, chess.BLACK)
            black_doubled = self._count_doubled_pawns(board, chess.BLACK)
            black_isolated = self._count_isolated_pawns(board, chess.BLACK)

            # Apply discovered weights (from white's perspective)
            score += (white_passed - black_passed) * self.passed_pawn_weight
            score += (white_doubled - black_doubled) * self.doubled_pawn_weight
            score += (white_isolated - black_isolated) * self.isolated_pawn_weight

            return score

        except Exception as e:
            logger.debug(f"Error evaluating pawn structure: {e}")
            return 0.0


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Discover and evaluate pawn structures')
    parser.add_argument('--db', type=str, default='rule_discovery.db', help='Database path')
    parser.add_argument('--discover', action='store_true', help='Run pawn structure discovery')
    parser.add_argument('--positions', type=int, default=5000, help='Number of positions to analyze')

    args = parser.parse_args()

    evaluator = PawnStructureEvaluator(args.db)

    try:
        if args.discover:
            # Run discovery
            patterns = evaluator.discover_pawn_structures(num_positions=args.positions)

            print("\n" + "=" * 80)
            print("PAWN STRUCTURE DISCOVERY COMPLETE")
            print("=" * 80)
            print(f"\nPassed pawn patterns: {len(patterns['passed'])}")
            print(f"Doubled pawn patterns: {len(patterns['doubled'])}")
            print(f"Isolated pawn patterns: {len(patterns['isolated'])}")
            print("\n" + "=" * 80)
        else:
            # Load and test
            evaluator._load_structure_weights()

            print("\n" + "=" * 80)
            print("TESTING PAWN STRUCTURE EVALUATOR")
            print("=" * 80)

            # Test positions
            test_cases = [
                ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                ("White passed pawn", "8/8/8/3P4/8/8/5p2/8 w - - 0 1"),
                ("Doubled pawns", "8/8/8/3P4/3P4/8/8/8 w - - 0 1"),
                ("Isolated pawn", "8/8/8/8/P7/8/5PPP/8 w - - 0 1"),
            ]

            for name, fen in test_cases:
                score = evaluator.evaluate_pawn_structure(fen)
                print(f"\n{name}:")
                print(f"  FEN: {fen}")
                print(f"  Structure score: {score:+.2f}")

            print("\n" + "=" * 80)
            print("✅ PAWN STRUCTURE EVALUATOR TESTS COMPLETE")
            print("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()
