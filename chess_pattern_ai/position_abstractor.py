#!/usr/bin/env python3
"""
Position Abstractor - Helper for Pattern Abstraction
Provides utilities for abstracting chess positions into patterns.
"""

import logging
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionAbstractor:
    """Utilities for abstracting chess positions into patterns"""

    def __init__(self):
        pass

    def abstract_position(self, fen: str) -> Dict[str, any]:
        """
        Abstract a position into key features

        Args:
            fen: FEN string of position

        Returns:
            Dictionary of abstract features
        """
        parts = fen.split()
        board_part = parts[0]
        turn = parts[1]
        castling = parts[2] if len(parts) > 2 else '-'

        features = {
            'material_balance': self._get_material_balance(board_part),
            'piece_count': self._count_pieces(board_part),
            'king_safety': self._abstract_king_safety(board_part, castling),
            'center_control': self._abstract_center_control(board_part),
            'development': self._abstract_development(board_part),
            'pawn_structure': self._abstract_pawn_structure(board_part)
        }

        return features

    def _get_material_balance(self, board_part: str) -> int:
        """Calculate material balance"""
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}

        white_material = 0
        black_material = 0

        for char in board_part:
            if char.isupper():
                white_material += piece_values.get(char, 0)
            elif char.islower():
                black_material += piece_values.get(char.upper(), 0)

        return white_material - black_material

    def _count_pieces(self, board_part: str) -> Dict[str, int]:
        """Count pieces by type"""
        counts = {}

        for char in board_part:
            if char.isalpha():
                piece_type = char.upper()
                if piece_type not in counts:
                    counts[piece_type] = 0
                counts[piece_type] += 1

        return counts

    def _abstract_king_safety(self, board_part: str, castling: str) -> Dict[str, any]:
        """Abstract king safety features"""
        # Find king positions
        white_king_pos = None
        black_king_pos = None
        square = 56

        for rank in board_part.split('/'):
            file = 0
            for char in rank:
                if char.isdigit():
                    file += int(char)
                else:
                    if char == 'K':
                        white_king_pos = square + file
                    elif char == 'k':
                        black_king_pos = square + file
                    file += 1
            square -= 8

        return {
            'white_castled': white_king_pos is not None and white_king_pos not in [4],
            'black_castled': black_king_pos is not None and black_king_pos not in [60],
            'white_can_castle': 'K' in castling or 'Q' in castling,
            'black_can_castle': 'k' in castling or 'q' in castling
        }

    def _abstract_center_control(self, board_part: str) -> Dict[str, int]:
        """Abstract center control"""
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

        return {
            'white_center_pawns': white_center,
            'black_center_pawns': black_center
        }

    def _abstract_development(self, board_part: str) -> Dict[str, int]:
        """Abstract piece development"""
        board = self._parse_board(board_part)

        # Check if minor pieces are off starting squares
        white_developed = 0
        black_developed = 0

        # White minor piece starting squares: b1, c1, f1, g1
        white_starts = [1, 2, 5, 6]
        for sq in white_starts:
            if board[sq] not in ['N', 'B']:
                white_developed += 1

        # Black minor piece starting squares: b8, c8, f8, g8
        black_starts = [57, 58, 61, 62]
        for sq in black_starts:
            if board[sq] not in ['n', 'b']:
                black_developed += 1

        return {
            'white_developed': white_developed,
            'black_developed': black_developed
        }

    def _abstract_pawn_structure(self, board_part: str) -> Dict[str, any]:
        """Abstract pawn structure features"""
        board = self._parse_board(board_part)

        white_doubled = 0
        black_doubled = 0
        white_isolated = 0
        black_isolated = 0

        # Check each file
        for file in range(8):
            white_pawns_in_file = 0
            black_pawns_in_file = 0

            for rank in range(8):
                sq = rank * 8 + file
                piece = board[sq]
                if piece == 'P':
                    white_pawns_in_file += 1
                elif piece == 'p':
                    black_pawns_in_file += 1

            # Doubled pawns
            if white_pawns_in_file > 1:
                white_doubled += white_pawns_in_file - 1
            if black_pawns_in_file > 1:
                black_doubled += black_pawns_in_file - 1

        return {
            'white_doubled_pawns': white_doubled,
            'black_doubled_pawns': black_doubled
        }

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


def test_position_abstractor():
    """Test the position abstractor"""
    print("=" * 70)
    print("POSITION ABSTRACTOR TEST")
    print("=" * 70)

    abstractor = PositionAbstractor()

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "After e4 e5"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game"),
    ]

    for fen, description in test_positions:
        print(f"\n{description}:")
        print(f"  FEN: {fen[:60]}...")

        features = abstractor.abstract_position(fen)

        print(f"\n  Abstract features:")
        for feature_name, feature_value in features.items():
            print(f"    {feature_name}: {feature_value}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_position_abstractor()
