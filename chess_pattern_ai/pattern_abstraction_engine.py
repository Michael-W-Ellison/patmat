#!/usr/bin/env python3
"""
Pattern Abstraction Engine

Instead of memorizing specific positions, extract abstract patterns:
- "Moving queen early loses tempo"
- "Leaving pieces undefended loses material"
- "King safety: castling before move 10"
- "Center control with pawns"

This allows the AI to learn PRINCIPLES, not positions.
"""

import chess
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PatternAbstractionEngine:
    """
    Extracts abstract patterns from concrete positions

    Converts "Queen to h5 in position X lost 3 pawns"
    Into "Early queen moves without development = dangerous"
    """

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self):
        """Create tables for abstract patterns"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS abstract_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT NOT NULL,
                severity REAL DEFAULT 0.0,
                times_seen INTEGER DEFAULT 0,
                times_punished INTEGER DEFAULT 0,
                avg_material_lost REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,

                -- Outcome tracking
                games_with_pattern_won INTEGER DEFAULT 0,
                games_with_pattern_lost INTEGER DEFAULT 0,
                games_with_pattern_draw INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, pattern_description)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_abstract_patterns_type
            ON abstract_patterns(pattern_type)
        ''')

        self.conn.commit()

    def extract_patterns_from_mistake(self, fen: str, move_uci: str,
                                     material_lost: float) -> List[Dict]:
        """
        Extract abstract patterns from a concrete mistake

        Args:
            fen: Position where mistake was made
            move_uci: Move that was a mistake
            material_lost: How much material was lost

        Returns:
            List of abstract patterns detected
        """
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)

        patterns = []

        # Pattern 1: Piece hanging (undefended piece moved/left)
        hanging_pattern = self._check_hanging_piece(board, move)
        if hanging_pattern:
            patterns.append(hanging_pattern)

        # Pattern 2: Early queen development
        queen_pattern = self._check_early_queen(board, move)
        if queen_pattern:
            patterns.append(queen_pattern)

        # Pattern 3: King safety
        king_pattern = self._check_king_safety(board, move)
        if king_pattern:
            patterns.append(king_pattern)

        # Pattern 4: Development vs tempo
        development_pattern = self._check_development(board, move)
        if development_pattern:
            patterns.append(development_pattern)

        # Pattern 5: Center control
        center_pattern = self._check_center_control(board, move)
        if center_pattern:
            patterns.append(center_pattern)

        # Pattern 6: Pawn structure
        pawn_pattern = self._check_pawn_structure(board, move)
        if pawn_pattern:
            patterns.append(pawn_pattern)

        # Record all detected patterns
        for pattern in patterns:
            self._record_pattern(pattern, material_lost)

        return patterns

    def _check_hanging_piece(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check if move leaves a piece hanging"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return None

        # Make the move
        board.push(move)

        # Is the piece now hanging?
        attackers = len(list(board.attackers(not board.turn, move.to_square)))
        defenders = len(list(board.attackers(board.turn, move.to_square)))

        board.pop()

        if attackers > defenders:
            piece_name = chess.piece_name(piece.piece_type)
            piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}[piece.piece_type]

            return {
                'type': 'hanging_piece',
                'description': f'{piece_name}_undefended',
                'severity': piece_value,
                'details': f'Moved {piece_name} to undefended square'
            }

        return None

    def _check_early_queen(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check for premature queen development"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.QUEEN:
            return None

        move_number = len(board.move_stack) + 1

        # Queen moved before move 5?
        if move_number <= 5:
            # Count how many other pieces are developed
            developed = 0
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p and p.color == board.turn:
                    if p.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        # On starting square?
                        if board.turn == chess.WHITE:
                            if sq not in [chess.B1, chess.G1, chess.C1, chess.F1]:
                                developed += 1
                        else:
                            if sq not in [chess.B8, chess.G8, chess.C8, chess.F8]:
                                developed += 1

            if developed < 2:
                return {
                    'type': 'premature_development',
                    'description': 'queen_moved_before_minor_pieces',
                    'severity': 2.0,
                    'details': f'Queen moved on move {move_number} with {developed} pieces developed'
                }

        return None

    def _check_king_safety(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check for king safety violations"""
        piece = board.piece_at(move.from_square)

        # King moved but hasn't castled?
        if piece and piece.piece_type == chess.KING:
            move_number = len(board.move_stack) + 1

            # Still has castling rights?
            if board.turn == chess.WHITE:
                can_castle = board.has_kingside_castling_rights(chess.WHITE) or \
                           board.has_queenside_castling_rights(chess.WHITE)
            else:
                can_castle = board.has_kingside_castling_rights(chess.BLACK) or \
                           board.has_queenside_castling_rights(chess.BLACK)

            if can_castle and move_number < 15:
                return {
                    'type': 'king_safety',
                    'description': 'king_moved_losing_castling_rights',
                    'severity': 1.5,
                    'details': 'Moved king before castling'
                }

        return None

    def _check_development(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check for development principles"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return None

        move_number = len(board.move_stack) + 1

        # Moving same piece twice in opening?
        if move_number <= 10:
            # Check if this piece has moved before
            from_square_file = chess.square_file(move.from_square)
            from_square_rank = chess.square_rank(move.from_square)

            # Is this piece on its starting square?
            is_starting_square = False
            if board.turn == chess.WHITE:
                if from_square_rank == 0 or from_square_rank == 1:
                    is_starting_square = True
            else:
                if from_square_rank == 6 or from_square_rank == 7:
                    is_starting_square = True

            if not is_starting_square and piece.piece_type != chess.PAWN:
                # This piece has moved before - is it the only developed piece?
                developed_count = 0
                for sq in chess.SQUARES:
                    p = board.piece_at(sq)
                    if p and p.color == board.turn and p.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        # Check if developed
                        rank = chess.square_rank(sq)
                        if board.turn == chess.WHITE:
                            if rank > 1:
                                developed_count += 1
                        else:
                            if rank < 6:
                                developed_count += 1

                if developed_count <= 2:
                    return {
                        'type': 'tempo_loss',
                        'description': 'moved_same_piece_twice_in_opening',
                        'severity': 1.0,
                        'details': f'Moved {chess.piece_name(piece.piece_type)} twice with only {developed_count} pieces developed'
                    }

        return None

    def _check_center_control(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check for center control principles"""
        # Did move give up center control?
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]

        piece = board.piece_at(move.from_square)
        if not piece:
            return None

        # Moving a piece away from center?
        if move.from_square in center_squares and move.to_square not in center_squares:
            move_number = len(board.move_stack) + 1

            if move_number <= 15:  # In opening
                return {
                    'type': 'center_control',
                    'description': 'abandoned_center_square',
                    'severity': 0.8,
                    'details': f'Moved {chess.piece_name(piece.piece_type)} away from center'
                }

        return None

    def _check_pawn_structure(self, board: chess.Board, move: chess.Move) -> Optional[Dict]:
        """Check for pawn structure weaknesses"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return None

        board.push(move)

        # Check for isolated pawns
        to_file = chess.square_file(move.to_square)
        has_adjacent_pawns = False

        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == board.turn and p.piece_type == chess.PAWN:
                sq_file = chess.square_file(sq)
                if abs(sq_file - to_file) == 1:
                    has_adjacent_pawns = True
                    break

        board.pop()

        if not has_adjacent_pawns:
            return {
                'type': 'pawn_structure',
                'description': 'created_isolated_pawn',
                'severity': 0.5,
                'details': 'Pawn move created isolated pawn'
            }

        return None

    def _record_pattern(self, pattern: Dict, material_lost: float):
        """Record or update an abstract pattern"""
        self.cursor.execute('''
            INSERT INTO abstract_patterns
            (pattern_type, pattern_description, times_seen, times_punished, avg_material_lost, severity)
            VALUES (?, ?, 1, 1, ?, ?)
            ON CONFLICT(pattern_type, pattern_description) DO UPDATE SET
                times_seen = times_seen + 1,
                times_punished = times_punished + 1,
                avg_material_lost = (avg_material_lost * times_seen + ?) / (times_seen + 1),
                confidence = MIN(1.0, (times_punished + 1) / 10.0)
        ''', (
            pattern['type'],
            pattern['description'],
            material_lost,
            pattern['severity'],
            material_lost
        ))

        # Don't commit here - let the parent tracker handle commits

    def check_for_known_patterns(self, fen: str, move_uci: str) -> List[Tuple[str, float, float, float]]:
        """
        Check if a move violates known abstract patterns

        Returns:
            List of (pattern_description, avg_material_lost, confidence, win_rate) tuples
        """
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)

        # Extract potential patterns from this move
        potential_patterns = self.extract_patterns_from_mistake(fen, move_uci, 0.0)

        # Check which ones we've learned are bad
        violations = []

        for pattern in potential_patterns:
            self.cursor.execute('''
                SELECT pattern_description, avg_material_lost, confidence, win_rate
                FROM abstract_patterns
                WHERE pattern_type = ? AND pattern_description = ?
                AND confidence > 0.3
            ''', (pattern['type'], pattern['description']))

            row = self.cursor.fetchone()
            if row:
                desc, avg_loss, confidence, win_rate = row
                violations.append((desc, avg_loss, confidence, win_rate if win_rate is not None else 0.5))

        return violations

    def get_pattern_statistics(self) -> List[Dict]:
        """Get all learned abstract patterns"""
        self.cursor.execute('''
            SELECT pattern_type, pattern_description, times_seen,
                   times_punished, avg_material_lost, confidence
            FROM abstract_patterns
            WHERE confidence > 0.2
            ORDER BY confidence DESC, avg_material_lost DESC
            LIMIT 50
        ''')

        patterns = []
        for row in self.cursor.fetchall():
            patterns.append({
                'type': row[0],
                'description': row[1],
                'times_seen': row[2],
                'times_punished': row[3],
                'avg_loss': row[4],
                'confidence': row[5]
            })

        return patterns

    def update_patterns_from_game_outcome(self, patterns_in_game: List[Dict], result: str):
        """
        Update pattern statistics based on game outcome

        Args:
            patterns_in_game: List of patterns detected during the game
            result: 'win', 'loss', or 'draw'
        """
        # Track which patterns appeared in this game
        unique_patterns = {}
        for pattern in patterns_in_game:
            key = (pattern['type'], pattern['description'])
            unique_patterns[key] = True

        # Update each pattern's win/loss/draw count
        for (pattern_type, pattern_desc) in unique_patterns.keys():
            if result == 'win':
                self.cursor.execute('''
                    UPDATE abstract_patterns
                    SET games_with_pattern_won = games_with_pattern_won + 1,
                        win_rate = CAST(games_with_pattern_won + 1 AS REAL) /
                                   (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw + 1)
                    WHERE pattern_type = ? AND pattern_description = ?
                ''', (pattern_type, pattern_desc))
            elif result == 'loss':
                self.cursor.execute('''
                    UPDATE abstract_patterns
                    SET games_with_pattern_lost = games_with_pattern_lost + 1,
                        win_rate = CAST(games_with_pattern_won AS REAL) /
                                   (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw + 1)
                    WHERE pattern_type = ? AND pattern_description = ?
                ''', (pattern_type, pattern_desc))
            elif result == 'draw':
                self.cursor.execute('''
                    UPDATE abstract_patterns
                    SET games_with_pattern_draw = games_with_pattern_draw + 1,
                        win_rate = CAST(games_with_pattern_won AS REAL) /
                                   (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw + 1)
                    WHERE pattern_type = ? AND pattern_description = ?
                ''', (pattern_type, pattern_desc))

        # Don't commit here - let parent handle it


def test_pattern_abstraction():
    """Test pattern abstraction"""
    print("=" * 70)
    print("PATTERN ABSTRACTION ENGINE TEST")
    print("=" * 70)

    engine = PatternAbstractionEngine()

    # Test case 1: Early queen move
    fen = "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    move = "d1h5"  # Queen out too early

    patterns = engine.extract_patterns_from_mistake(fen, move, 3.0)

    print("\nTest 1: Early queen development")
    print(f"Position: After e3")
    print(f"Move: Qh5")
    print(f"Patterns detected: {len(patterns)}")
    for p in patterns:
        print(f"  - {p['type']}: {p['description']} (severity: {p['severity']})")

    # Test case 2: Hanging piece
    fen = "rnbqkb1r/pppppppp/5n2/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
    move = "b1c3"  # Develop knight

    patterns = engine.extract_patterns_from_mistake(fen, move, 0.0)
    print("\nTest 2: Normal development")
    print(f"Move: Nc3")
    print(f"Patterns detected: {len(patterns)} (should be 0 or low severity)")

    # Show learned patterns
    print("\n" + "=" * 70)
    print("LEARNED ABSTRACT PATTERNS")
    print("=" * 70)

    stats = engine.get_pattern_statistics()
    for p in stats[:10]:
        print(f"\n{p['type']}: {p['description']}")
        print(f"  Seen: {p['times_seen']} times")
        print(f"  Avg loss: {p['avg_loss']:.1f} material")
        print(f"  Confidence: {p['confidence']:.2f}")


if __name__ == '__main__':
    test_pattern_abstraction()
