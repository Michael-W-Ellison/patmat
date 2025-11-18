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
        Load discovered opening patterns from database
        These patterns were learned from analyzing successful openings
        """
        try:
            # Load opening weights discovered from game analysis
            self.cursor.execute('''
                SELECT pattern_name, weight, confidence, observation_count
                FROM discovered_opening_weights
                ORDER BY ABS(weight) DESC
            ''')

            patterns = self.cursor.fetchall()
            for pattern_name, weight, confidence, obs_count in patterns:
                self.opening_weights[pattern_name] = {
                    'weight': weight,
                    'confidence': confidence,
                    'observations': obs_count
                }

            if self.opening_weights:
                logger.info(f"✓ Loaded {len(self.opening_weights)} discovered opening patterns")
            else:
                logger.warning("No discovered opening patterns found")

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

        # Check discovered opening patterns
        for pattern_name, data in self.opening_weights.items():
            weight = data['weight']
            confidence = data['confidence']

            if self._pattern_matches(pattern_name, board_part, turn, castling, move_num):
                score += weight * confidence

        return score

    def _pattern_matches(self, pattern_name: str, board_part: str,
                        turn: str, castling: str, move_num: int) -> bool:
        """
        Check if a discovered opening pattern matches

        Args:
            pattern_name: Name of the pattern
            board_part: Board position from FEN
            turn: Current turn ('w' or 'b')
            castling: Castling rights
            move_num: Move number

        Returns:
            True if pattern matches
        """
        pattern_lower = pattern_name.lower()

        # Center control patterns
        if "center_control" in pattern_lower:
            return self._has_center_control(board_part)

        # Development patterns
        elif "minor_piece_developed" in pattern_lower:
            return self._has_minor_pieces_developed(board_part, turn)

        # Castling patterns
        elif "castling_available" in pattern_lower:
            if turn == 'w':
                return 'K' in castling or 'Q' in castling
            else:
                return 'k' in castling or 'q' in castling

        # Early queen development (usually bad)
        elif "early_queen" in pattern_lower:
            return self._has_early_queen_development(board_part, turn, move_num)

        # Pawn structure patterns
        elif "doubled_pawns" in pattern_lower:
            return self._has_doubled_pawns(board_part)

        return False

    def _has_center_control(self, board_part: str) -> bool:
        """Check if side controls center squares (e4, d4, e5, d5)"""
        # Parse board
        board = self._parse_board(board_part)

        # Center squares: e4=28, d4=27, e5=36, d5=35
        center_squares = [27, 28, 35, 36]

        white_center = 0
        black_center = 0

        for sq in center_squares:
            piece = board[sq]
            if piece and piece.isupper():
                white_center += 1
            elif piece and piece.islower():
                black_center += 1

        return white_center > black_center

    def _has_minor_pieces_developed(self, board_part: str, turn: str) -> bool:
        """Check if minor pieces (knights, bishops) are developed"""
        board = self._parse_board(board_part)

        if turn == 'w':
            # White's back rank squares for minor pieces: b1=1, c1=2, f1=5, g1=6
            back_rank_squares = [1, 2, 5, 6]
            developed = 0
            for sq in back_rank_squares:
                piece = board[sq]
                if piece is None or piece not in ['N', 'B']:
                    developed += 1
            return developed >= 2
        else:
            # Black's back rank: b8=57, c8=58, f8=61, g8=62
            back_rank_squares = [57, 58, 61, 62]
            developed = 0
            for sq in back_rank_squares:
                piece = board[sq]
                if piece is None or piece not in ['n', 'b']:
                    developed += 1
            return developed >= 2

    def _has_early_queen_development(self, board_part: str, turn: str, move_num: int) -> bool:
        """Check if queen moved early (usually bad)"""
        if move_num > 5:
            return False

        board = self._parse_board(board_part)

        if turn == 'w':
            # White queen starts on d1 (square 3)
            return board[3] != 'Q'
        else:
            # Black queen starts on d8 (square 59)
            return board[59] != 'q'

    def _has_doubled_pawns(self, board_part: str) -> bool:
        """Check for doubled pawns (usually bad)"""
        board = self._parse_board(board_part)

        # Check each file for doubled pawns
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                sq = rank * 8 + file
                piece = board[sq]
                if piece == 'P':
                    white_pawns += 1
                elif piece == 'p':
                    black_pawns += 1

            if white_pawns > 1 or black_pawns > 1:
                return True

        return False

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
