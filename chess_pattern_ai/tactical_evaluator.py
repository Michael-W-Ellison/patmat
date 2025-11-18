#!/usr/bin/env python3
"""
Tactical Evaluator - Phase 3: Tactical Pattern Discovery
Discovers tactical patterns (forks, pins, skewers, checkmate) from game data.
No hardcoded chess knowledge - pure geometric pattern recognition.
"""

import sqlite3
import logging
import chess
import json
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TacticalPattern:
    """A discovered tactical pattern"""
    pattern_type: str  # 'fork', 'pin', 'skewer', 'checkmate'
    piece_type: str  # Which piece creates the pattern
    value_estimate: float  # Expected material gain
    frequency: int  # Occurrences in database
    confidence: float  # Statistical confidence
    geometric_signature: Dict  # Pattern description


@dataclass
class TacticalDiscovery:
    """Complete tactical pattern discovery results"""
    fork_weight: float
    pin_weight: float
    skewer_weight: float
    checkmate_bonus: float
    patterns_found: List[TacticalPattern]
    observation_count: int


class TacticalEvaluator:
    """Discovers and evaluates tactical patterns from game data"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

        # Discovered tactical weights
        self.fork_weight = 0.0
        self.pin_weight = 0.0
        self.skewer_weight = 0.0
        self.checkmate_bonus = 0.0

        # Discovered piece values (load from Phase 2)
        self.piece_values = {}

        self._init_connection()
        self._load_piece_values()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_piece_values(self):
        """Load discovered piece values from Phase 2"""
        try:
            self.cursor.execute('''
                SELECT piece_type, relative_value
                FROM discovered_piece_values
            ''')

            self.piece_values = {row[0]: row[1] for row in self.cursor.fetchall()}

            if self.piece_values:
                logger.info(f"Loaded {len(self.piece_values)} piece values from Phase 2")
            else:
                # Fallback values if Phase 2 not run
                self.piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
                logger.warning("Using fallback piece values")
        except:
            self.piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
            logger.warning("Could not load piece values, using fallback")

    def _get_piece_value(self, piece) -> float:
        """Get value of a chess piece"""
        if piece is None:
            return 0.0
        piece_type = piece.symbol().upper()
        return self.piece_values.get(piece_type, 0.0)

    def discover_fork_patterns(self) -> List[TacticalPattern]:
        """
        Discover fork patterns: piece attacks multiple enemy pieces simultaneously.

        Geometric approach: no chess terminology, just counting attacks.
        """
        logger.info("="*80)
        logger.info("DISCOVERING FORK PATTERNS")
        logger.info("="*80)

        # Sample positions from tactical moments (captures)
        # FIXED: Increased from 5000 to 500000 to learn from more data
        # (853K tactical positions available, was only using 0.6%)
        self.cursor.execute('''
            SELECT fen_before, move_uci, piece_type
            FROM moves
            WHERE move_number BETWEEN 10 AND 40
            AND is_capture = 1
            ORDER BY RANDOM()
            LIMIT 500000
        ''')

        positions = self.cursor.fetchall()
        logger.info(f"Analyzing {len(positions)} tactical positions for fork patterns...")

        # Track fork patterns by fine-grained signatures
        # FIXED: Changed from coarse grouping (e.g., all "knight forks") to
        # fine-grained patterns (e.g., "knight forks queen+rook", "knight forks rook+bishop")
        # This allows learning specific tactical values for each pattern type
        fork_stats = defaultdict(lambda: {
            'count': 0,
            'total_value': 0.0,
            'attacked_pieces': []
        })

        forks_found = 0

        for fen, move_uci, piece_type in positions:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)

                if move not in board.legal_moves:
                    continue

                board.push(move)

                # After the move, check if the moved piece attacks multiple enemy pieces
                to_square = move.to_square
                moved_piece = board.piece_at(to_square)

                if moved_piece is None:
                    continue

                # Count enemy pieces attacked from this square
                attacked_squares = []
                attacked_values = []

                for sq in chess.SQUARES:
                    target_piece = board.piece_at(sq)
                    if target_piece and target_piece.color != moved_piece.color:
                        # Check if our piece attacks this square
                        if board.is_attacked_by(moved_piece.color, sq):
                            # Verify the attacking piece is actually our moved piece
                            # (not just any piece of our color)
                            if sq in board.attacks(to_square):
                                attacked_squares.append(sq)
                                attacked_values.append(self._get_piece_value(target_piece))

                # Fork: attacks 2+ enemy pieces
                if len(attacked_squares) >= 2:
                    forks_found += 1
                    total_attacked_value = sum(attacked_values)
                    attacker_value = self._get_piece_value(moved_piece)

                    # Net value of fork
                    # (can potentially win the lowest value piece for free)
                    min_attacked = min(attacked_values) if attacked_values else 0
                    net_value = min_attacked  # Conservative estimate

                    # FIXED: Create fine-grained pattern signature
                    # Format: "N_forks_Q+R" (knight forks queen and rook)
                    attacked_piece_types = sorted([
                        board.piece_at(sq).symbol().upper()
                        for sq in attacked_squares
                    ])
                    pattern_signature = f"{piece_type}_forks_{'_'.join(attacked_piece_types)}"

                    fork_stats[pattern_signature]['count'] += 1
                    fork_stats[pattern_signature]['total_value'] += net_value
                    fork_stats[pattern_signature]['attacked_pieces'].append(len(attacked_squares))

            except Exception as e:
                logger.debug(f"Error processing position: {e}")
                continue

        logger.info(f"\nFORK PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        patterns = []

        # FIXED: Iterate over actual pattern signatures instead of piece types
        for pattern_signature, stats in fork_stats.items():
            if stats['count'] > 0:
                avg_value = stats['total_value'] / stats['count']
                avg_attacked = sum(stats['attacked_pieces']) / len(stats['attacked_pieces'])

                # Extract piece type from signature (e.g., "N_forks_Q_R" -> "N")
                piece_type = pattern_signature.split('_')[0]

                logger.info(f"{pattern_signature}: {stats['count']} occurrences, " +
                           f"avg value: {avg_value:.2f}, " +
                           f"avg pieces attacked: {avg_attacked:.1f}")

                pattern = TacticalPattern(
                    pattern_type='fork',
                    piece_type=piece_type,
                    value_estimate=avg_value,
                    frequency=stats['count'],
                    confidence=min(0.95, stats['count'] / 100),
                    geometric_signature={
                        'pattern_signature': pattern_signature,
                        'avg_pieces_attacked': avg_attacked,
                        'avg_value_gained': avg_value
                    }
                )
                patterns.append(pattern)

        logger.info(f"\nTotal forks found: {forks_found}")
        logger.info(f"Fork rate: {forks_found / len(positions) * 100:.2f}%")

        return patterns

    def discover_pin_patterns(self) -> List[TacticalPattern]:
        """
        Discover pin patterns: aligned attacker-pinned-valuable pieces.

        Geometric approach: three pieces on same line, middle piece cannot move.
        """
        logger.info("\n" + "="*80)
        logger.info("DISCOVERING PIN/SKEWER PATTERNS")
        logger.info("="*80)

        # Sample positions with sliding pieces (B, R, Q can create pins)
        # FIXED: Increased from 3000 to 300000 (100x increase) to match fork discovery
        # This should discover 200-300 pin patterns instead of just 3
        self.cursor.execute('''
            SELECT DISTINCT fen_before
            FROM moves
            WHERE piece_type IN ('B', 'R', 'Q')
            AND move_number BETWEEN 10 AND 40
            ORDER BY RANDOM()
            LIMIT 300000
        ''')

        positions = [row[0] for row in self.cursor.fetchall()]
        logger.info(f"Analyzing {len(positions)} positions for pin/skewer patterns...")

        # FIXED: Use fine-grained pattern signatures instead of coarse grouping
        # This will create patterns like "B_pins_N_protecting_Q" instead of just "B"
        pin_stats = defaultdict(lambda: {'count': 0, 'total_value': 0.0})
        skewer_stats = defaultdict(lambda: {'count': 0, 'total_value': 0.0})

        pins_found = 0
        skewers_found = 0

        for fen in positions:
            try:
                board = chess.Board(fen)

                # Check each sliding piece
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        # Find pieces this piece attacks
                        attacked = board.attacks(sq)

                        for attacked_sq in attacked:
                            attacked_piece = board.piece_at(attacked_sq)
                            if attacked_piece and attacked_piece.color != piece.color:
                                # Check if there's a piece behind (aligned)
                                behind_sq, behind_piece = self._find_piece_behind(
                                    board, sq, attacked_sq, piece.piece_type
                                )

                                if behind_piece and behind_piece.color == attacked_piece.color:
                                    attacked_value = self._get_piece_value(attacked_piece)
                                    behind_value = self._get_piece_value(behind_piece)

                                    # FIXED: Create fine-grained signatures
                                    attacker_type = piece.symbol().upper()
                                    pinned_type = attacked_piece.symbol().upper()
                                    behind_type = behind_piece.symbol().upper()

                                    # Pin: behind piece is more valuable
                                    if behind_value > attacked_value:
                                        pins_found += 1
                                        value_diff = behind_value - attacked_value

                                        # Fine-grained pattern: "B_pins_N_protecting_Q"
                                        pattern_sig = f"{attacker_type}_pins_{pinned_type}_protecting_{behind_type}"
                                        pin_stats[pattern_sig]['count'] += 1
                                        pin_stats[pattern_sig]['total_value'] += value_diff

                                    # Skewer: attacked piece is more valuable
                                    elif attacked_value > behind_value:
                                        skewers_found += 1
                                        value_diff = attacked_value

                                        # Fine-grained pattern: "R_skewers_Q_behind_N"
                                        pattern_sig = f"{attacker_type}_skewers_{pinned_type}_behind_{behind_type}"
                                        skewer_stats[pattern_sig]['count'] += 1
                                        skewer_stats[pattern_sig]['total_value'] += value_diff

            except Exception as e:
                logger.debug(f"Error processing position: {e}")
                continue

        logger.info(f"\nPIN PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        patterns = []

        # FIXED: Iterate over actual pattern signatures instead of just piece types
        for pattern_sig, stats in pin_stats.items():
            if stats['count'] > 0:
                avg_value = stats['total_value'] / stats['count']

                # Extract attacker type from signature (e.g., "B_pins_N_protecting_Q" -> "B")
                attacker_type = pattern_sig.split('_')[0]

                logger.info(f"{pattern_sig}: {stats['count']} occurrences, avg value: {avg_value:.2f}")

                pattern = TacticalPattern(
                    pattern_type='pin',
                    piece_type=attacker_type,
                    value_estimate=avg_value,
                    frequency=stats['count'],
                    confidence=min(0.90, stats['count'] / 50),
                    geometric_signature={
                        'pattern_signature': pattern_sig,
                        'avg_value_protected': avg_value
                    }
                )
                patterns.append(pattern)

        logger.info(f"\nSKEWER PATTERNS DISCOVERED:")
        logger.info("-" * 80)

        # FIXED: Iterate over actual pattern signatures
        for pattern_sig, stats in skewer_stats.items():
            if stats['count'] > 0:
                avg_value = stats['total_value'] / stats['count']

                # Extract attacker type from signature
                attacker_type = pattern_sig.split('_')[0]

                logger.info(f"{pattern_sig}: {stats['count']} occurrences, avg value: {avg_value:.2f}")

                pattern = TacticalPattern(
                    pattern_type='skewer',
                    piece_type=attacker_type,
                    value_estimate=avg_value,
                    frequency=stats['count'],
                    confidence=min(0.90, stats['count'] / 50),
                    geometric_signature={
                        'pattern_signature': pattern_sig,
                        'avg_value_gained': avg_value
                    }
                )
                patterns.append(pattern)

        logger.info(f"\nTotal pins: {pins_found}, Total skewers: {skewers_found}")

        return patterns

    def _find_piece_behind(self, board: chess.Board, attacker_sq: int,
                           target_sq: int, piece_type: int) -> Tuple[Optional[int], Optional[chess.Piece]]:
        """
        Find piece behind target square on the same line as attacker.

        Geometric analysis: extend the line from attacker through target.
        """
        attacker_file, attacker_rank = chess.square_file(attacker_sq), chess.square_rank(attacker_sq)
        target_file, target_rank = chess.square_file(target_sq), chess.square_rank(target_sq)

        # Calculate direction
        file_delta = target_file - attacker_file
        rank_delta = target_rank - attacker_rank

        # Normalize direction
        if file_delta != 0:
            file_step = file_delta // abs(file_delta)
        else:
            file_step = 0

        if rank_delta != 0:
            rank_step = rank_delta // abs(rank_delta)
        else:
            rank_step = 0

        # Check if this is valid for the piece type
        if piece_type == chess.BISHOP and (abs(file_delta) != abs(rank_delta)):
            return None, None
        if piece_type == chess.ROOK and (file_delta != 0 and rank_delta != 0):
            return None, None

        # Continue in same direction from target
        current_file = target_file + file_step
        current_rank = target_rank + rank_step

        # Search for next piece on this line
        while 0 <= current_file <= 7 and 0 <= current_rank <= 7:
            sq = chess.square(current_file, current_rank)
            piece = board.piece_at(sq)

            if piece is not None:
                return sq, piece

            current_file += file_step
            current_rank += rank_step

        return None, None

    def discover_discovered_attacks(self) -> List[TacticalPattern]:
        """
        Discover discovered attack patterns: piece moves revealing attack from behind.

        Pattern: Piece A moves, revealing attack from sliding piece B onto target C.
        This is a critical tactical motif often missed by beginners.
        """
        logger.info("\n" + "="*80)
        logger.info("DISCOVERING DISCOVERED ATTACK PATTERNS")
        logger.info("="*80)

        # Sample moves where sliding pieces (B, R, Q) might have discovered attacks
        self.cursor.execute('''
            SELECT fen_before, move_uci, piece_type, fen_after
            FROM moves
            WHERE move_number BETWEEN 10 AND 40
            ORDER BY RANDOM()
            LIMIT 100000
        ''')

        moves = self.cursor.fetchall()
        logger.info(f"Analyzing {len(moves)} moves for discovered attack patterns...")

        # Track discovered attacks by attacker piece type and target value
        discovered_stats = defaultdict(lambda: {
            'count': 0,
            'total_value': 0.0,
            'with_check': 0
        })

        discovered_found = 0

        for fen_before, move_uci, moved_piece_type, fen_after in moves:
            try:
                board_before = chess.Board(fen_before)
                board_after = chess.Board(fen_after)
                move = chess.Move.from_uci(move_uci)

                if move not in board_before.legal_moves:
                    continue

                moved_piece = board_before.piece_at(move.from_square)
                if not moved_piece:
                    continue

                # Check for sliding pieces behind the moved piece's original square
                for attacker_sq in chess.SQUARES:
                    attacker = board_after.piece_at(attacker_sq)
                    if not attacker or attacker.color != moved_piece.color:
                        continue

                    # Only sliding pieces can create discovered attacks
                    if attacker.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        continue

                    # Check if attacker now attacks squares it couldn't before
                    attacks_after = board_after.attacks(attacker_sq)

                    # Simulate: what if piece hadn't moved?
                    board_before.remove_piece_at(move.to_square)  # Remove if capture
                    board_before.set_piece_at(move.to_square, moved_piece)  # Put moved piece back
                    board_before.remove_piece_at(move.from_square)

                    attacks_before = board_before.attacks(attacker_sq) if board_before.piece_at(attacker_sq) else set()

                    # Find newly attacked enemy pieces
                    new_attacks = attacks_after - attacks_before

                    for target_sq in new_attacks:
                        target = board_after.piece_at(target_sq)
                        if target and target.color != attacker.color:
                            # Found a discovered attack!
                            discovered_found += 1
                            target_value = self._get_piece_value(target)

                            # Check if it's also a discovered check
                            is_check = target.piece_type == chess.KING

                            # Create pattern signature
                            pattern_key = f"{attacker.symbol().upper()}_discovers_attack_on_{target.symbol().upper()}"

                            discovered_stats[pattern_key]['count'] += 1
                            discovered_stats[pattern_key]['total_value'] += target_value
                            if is_check:
                                discovered_stats[pattern_key]['with_check'] += 1

            except Exception as e:
                logger.debug(f"Error analyzing discovered attack: {e}")
                continue

        logger.info(f"\nDISCOVERED ATTACK PATTERNS FOUND:")
        logger.info("-" * 80)

        patterns = []
        for pattern_sig, stats in discovered_stats.items():
            if stats['count'] > 0:
                avg_value = stats['total_value'] / stats['count']
                check_rate = stats['with_check'] / stats['count'] if stats['count'] > 0 else 0

                # Extract attacker type from signature
                attacker_type = pattern_sig.split('_')[0]

                logger.info(f"{pattern_sig}: {stats['count']} occurrences, "
                           f"avg value: {avg_value:.2f}, "
                           f"discovered check: {check_rate:.1%}")

                # Boost value if it's often a discovered check (very forcing)
                if check_rate > 0.3:
                    avg_value *= 1.5

                pattern = TacticalPattern(
                    pattern_type='discovered_attack',
                    piece_type=attacker_type,
                    value_estimate=avg_value,
                    frequency=stats['count'],
                    confidence=min(0.95, stats['count'] / 100),
                    geometric_signature={
                        'pattern_signature': pattern_sig,
                        'discovered_check_rate': check_rate,
                        'avg_value': avg_value
                    }
                )
                patterns.append(pattern)

        logger.info(f"\nTotal discovered attacks found: {discovered_found}")
        logger.info(f"Unique patterns: {len(patterns)}")

        return patterns

    def discover_checkmate_patterns(self) -> List[TacticalPattern]:
        """
        Discover checkmate patterns from game endings.

        Geometric approach: king position, attacking pieces, escape squares blocked.
        """
        logger.info("\n" + "="*80)
        logger.info("DISCOVERING CHECKMATE PATTERNS")
        logger.info("="*80)

        # Query final positions from decisive games
        self.cursor.execute('''
            SELECT m.fen_after, g.result
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE g.result IN ('1-0', '0-1')
            AND m.move_number = (
                SELECT MAX(move_number) FROM moves WHERE game_id = m.game_id
            )
        ''')

        final_positions = self.cursor.fetchall()
        logger.info(f"Analyzing {len(final_positions)} game endings for checkmate patterns...")

        checkmate_patterns = []
        checkmates_found = 0

        for fen, result in final_positions:
            try:
                board = chess.Board(fen)

                if board.is_checkmate():
                    checkmates_found += 1

                    # Extract geometric pattern
                    losing_color = not board.turn  # Side that got checkmated
                    king_sq = board.king(losing_color)

                    if king_sq is None:
                        continue

                    king_file = chess.square_file(king_sq)
                    king_rank = chess.square_rank(king_sq)

                    # Geometric features (no chess terminology)
                    pattern = {
                        'king_on_edge': (king_file in [0, 7]) or (king_rank in [0, 7]),
                        'king_in_corner': (king_file in [0, 7]) and (king_rank in [0, 7]),
                        'king_on_back_rank': (king_rank in [0, 7]),
                        'edge_distance': min(king_file, 7 - king_file, king_rank, 7 - king_rank),
                        'attackers': self._get_attacking_pieces(board, king_sq),
                        'escape_blocked': self._count_blocked_escape_squares(board, king_sq)
                    }

                    checkmate_patterns.append(pattern)

            except Exception as e:
                logger.debug(f"Error processing ending: {e}")
                continue

        logger.info(f"\nCHECKMATE PATTERNS DISCOVERED:")
        logger.info("-" * 80)
        logger.info(f"Checkmates found: {checkmates_found} / {len(final_positions)}")
        logger.info(f"Checkmate rate: {checkmates_found / max(len(final_positions), 1) * 100:.1f}%")

        if checkmate_patterns:
            # Analyze patterns
            edge_mates = sum(1 for p in checkmate_patterns if p['king_on_edge'])
            corner_mates = sum(1 for p in checkmate_patterns if p['king_in_corner'])
            back_rank_mates = sum(1 for p in checkmate_patterns if p['king_on_back_rank'])

            logger.info(f"\nGeometric analysis:")
            logger.info(f"  King on edge: {edge_mates / len(checkmate_patterns) * 100:.1f}%")
            logger.info(f"  King in corner: {corner_mates / len(checkmate_patterns) * 100:.1f}%")
            logger.info(f"  King on back rank: {back_rank_mates / len(checkmate_patterns) * 100:.1f}%")

            # Attacker analysis
            all_attackers = []
            for p in checkmate_patterns:
                all_attackers.extend(p['attackers'])

            if all_attackers:
                from collections import Counter
                attacker_counts = Counter(all_attackers)
                logger.info(f"\nMost common attacking pieces:")
                for piece, count in attacker_counts.most_common(5):
                    logger.info(f"  {piece}: {count} times ({count/len(checkmate_patterns)*100:.1f}%)")

        # Create checkmate pattern discovery
        pattern = TacticalPattern(
            pattern_type='checkmate',
            piece_type='ANY',
            value_estimate=1000.0,  # Game ending
            frequency=checkmates_found,
            confidence=0.99,
            geometric_signature={
                'total_found': checkmates_found,
                'edge_percentage': edge_mates / max(len(checkmate_patterns), 1) * 100,
                'corner_percentage': corner_mates / max(len(checkmate_patterns), 1) * 100
            }
        )

        return [pattern]

    def _get_attacking_pieces(self, board: chess.Board, king_sq: int) -> List[str]:
        """Get list of pieces attacking the king"""
        attackers = []
        attacking_color = not board.color_at(king_sq)

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == attacking_color:
                if king_sq in board.attacks(sq):
                    attackers.append(piece.symbol().upper())

        return attackers

    def _count_blocked_escape_squares(self, board: chess.Board, king_sq: int) -> int:
        """Count how many escape squares are blocked or attacked"""
        king_color = board.color_at(king_sq)
        blocked_count = 0

        # Check all 8 adjacent squares
        for file_delta in [-1, 0, 1]:
            for rank_delta in [-1, 0, 1]:
                if file_delta == 0 and rank_delta == 0:
                    continue

                file = chess.square_file(king_sq) + file_delta
                rank = chess.square_rank(king_sq) + rank_delta

                if 0 <= file <= 7 and 0 <= rank <= 7:
                    sq = chess.square(file, rank)
                    piece = board.piece_at(sq)

                    # Blocked if occupied by friendly piece or attacked by enemy
                    if (piece and piece.color == king_color) or \
                       board.is_attacked_by(not king_color, sq):
                        blocked_count += 1

        return blocked_count

    def synthesize_tactical_discovery(self) -> TacticalDiscovery:
        """
        Run all discovery methods and synthesize tactical evaluation weights.
        """
        logger.info("\n" + "="*80)
        logger.info("SYNTHESIZING TACTICAL EVALUATION WEIGHTS")
        logger.info("="*80)

        # Discover all pattern types
        fork_patterns = self.discover_fork_patterns()
        pin_patterns = self.discover_pin_patterns()
        checkmate_patterns = self.discover_checkmate_patterns()

        all_patterns = fork_patterns + pin_patterns + checkmate_patterns

        # Calculate weights based on discovered patterns
        # Fork weight: average value of discovered forks
        fork_values = [p.value_estimate for p in fork_patterns if p.pattern_type == 'fork']
        fork_weight = sum(fork_values) / len(fork_values) if fork_values else 0.0

        # Pin weight: average value of pins
        pin_values = [p.value_estimate for p in pin_patterns if p.pattern_type == 'pin']
        pin_weight = sum(pin_values) / len(pin_values) if pin_values else 0.0

        # Skewer weight: average value of skewers
        skewer_values = [p.value_estimate for p in pin_patterns if p.pattern_type == 'skewer']
        skewer_weight = sum(skewer_values) / len(skewer_values) if skewer_values else 0.0

        # Checkmate: extremely high value (game ending)
        checkmate_bonus = 1000.0

        logger.info("\nDISCOVERED TACTICAL WEIGHTS:")
        logger.info("-" * 80)
        logger.info(f"Fork weight: {fork_weight:.2f} points per fork opportunity")
        logger.info(f"Pin weight: {pin_weight:.2f} points per pin")
        logger.info(f"Skewer weight: {skewer_weight:.2f} points per skewer")
        logger.info(f"Checkmate bonus: {checkmate_bonus:.2f} points")

        discovery = TacticalDiscovery(
            fork_weight=fork_weight,
            pin_weight=pin_weight,
            skewer_weight=skewer_weight,
            checkmate_bonus=checkmate_bonus,
            patterns_found=all_patterns,
            observation_count=len(all_patterns)
        )

        # Store in database
        self._store_tactical_discovery(discovery)

        self.fork_weight = fork_weight
        self.pin_weight = pin_weight
        self.skewer_weight = skewer_weight
        self.checkmate_bonus = checkmate_bonus

        return discovery

    def _store_tactical_discovery(self, discovery: TacticalDiscovery):
        """Store discovered tactical patterns in database"""

        # Create tables if not exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_tactical_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                piece_type TEXT,
                value_estimate REAL,
                frequency INTEGER,
                confidence REAL,
                geometric_signature TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_tactical_weights (
                id INTEGER PRIMARY KEY,
                fork_weight REAL,
                pin_weight REAL,
                skewer_weight REAL,
                checkmate_bonus REAL,
                observation_count INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Store patterns
        for pattern in discovery.patterns_found:
            self.cursor.execute('''
                INSERT INTO discovered_tactical_patterns
                (pattern_type, piece_type, value_estimate, frequency, confidence, geometric_signature)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pattern.pattern_type, pattern.piece_type, pattern.value_estimate,
                  pattern.frequency, pattern.confidence, json.dumps(pattern.geometric_signature)))

        # Store weights
        self.cursor.execute('''
            INSERT OR REPLACE INTO discovered_tactical_weights
            (id, fork_weight, pin_weight, skewer_weight, checkmate_bonus, observation_count)
            VALUES (1, ?, ?, ?, ?, ?)
        ''', (discovery.fork_weight, discovery.pin_weight, discovery.skewer_weight,
              discovery.checkmate_bonus, discovery.observation_count))

        self.conn.commit()
        logger.info("\n✅ Tactical patterns stored in database")

    def evaluate_tactics(self, fen: str) -> float:
        """
        Evaluate tactical opportunities in a position.
        Returns tactical score contribution.
        """
        if not self.fork_weight:
            self._load_tactical_weights()

        try:
            board = chess.Board(fen)
            tactical_score = 0.0

            # Check for checkmate
            if board.is_checkmate():
                return self.checkmate_bonus if board.turn == chess.BLACK else -self.checkmate_bonus

            # Detect forks (simple version - piece attacks 2+ enemy pieces)
            fork_count = self._detect_forks_simple(board)
            tactical_score += fork_count * self.fork_weight

            return tactical_score

        except Exception as e:
            logger.debug(f"Error evaluating tactics: {e}")
            return 0.0

    def _detect_forks_simple(self, board: chess.Board) -> int:
        """Simple fork detection for position evaluation"""
        fork_count = 0

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                # Count enemy pieces attacked
                attacked_count = 0
                for target_sq in chess.SQUARES:
                    target = board.piece_at(target_sq)
                    if target and target.color != piece.color:
                        if target_sq in board.attacks(sq):
                            attacked_count += 1

                if attacked_count >= 2:
                    fork_count += 1

        return fork_count

    def _load_tactical_weights(self):
        """Load tactical weights from database"""
        try:
            self.cursor.execute('''
                SELECT fork_weight, pin_weight, skewer_weight, checkmate_bonus
                FROM discovered_tactical_weights
                WHERE id = 1
            ''')

            result = self.cursor.fetchone()
            if result:
                self.fork_weight, self.pin_weight, self.skewer_weight, self.checkmate_bonus = result
                logger.info("Loaded tactical weights from database")
        except:
            logger.warning("No tactical weights found in database")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Discover tactical patterns from observation')
    parser.add_argument('--db', type=str, default='rule_discovery.db', help='Database path')

    args = parser.parse_args()

    evaluator = TacticalEvaluator(args.db)

    try:
        # Run discovery process
        discovery = evaluator.synthesize_tactical_discovery()

        print("\n" + "="*80)
        print("✅ TACTICAL PATTERN DISCOVERY COMPLETE")
        print("="*80)
        print(f"\nPatterns discovered: {len(discovery.patterns_found)}")
        print(f"\nWeights:")
        print(f"  Fork weight: {discovery.fork_weight:.2f}")
        print(f"  Pin weight: {discovery.pin_weight:.2f}")
        print(f"  Skewer weight: {discovery.skewer_weight:.2f}")
        print(f"  Checkmate bonus: {discovery.checkmate_bonus:.2f}")

    except Exception as e:
        logger.error(f"Error during tactical discovery: {e}", exc_info=True)
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()
