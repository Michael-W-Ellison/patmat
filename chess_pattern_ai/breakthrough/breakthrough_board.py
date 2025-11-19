#!/usr/bin/env python3
"""
Breakthrough Board Representation - Observation-Based

Philosophy: Minimal hardcoded rules
- Board is 8x8, all squares valid
- Only pawns (no piece promotion)
- Goal: reach opponent's back row or capture all pieces
- Movement rules learned from observation
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


@dataclass
class Piece:
    """A breakthrough piece (pawn)"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        color_char = 'W' if self.color == Color.WHITE else 'B'
        return f"{color_char}"

    def __hash__(self):
        """Make piece hashable for use in sets"""
        return hash((self.color, self.position))

    def __eq__(self, other):
        """Equality based on color and position"""
        if not isinstance(other, Piece):
            return False
        return (self.color == other.color and
                self.position == other.position)


@dataclass
class Move:
    """A breakthrough move"""
    piece: Piece
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    is_capture: bool = False
    captured_piece: Optional[Piece] = None

    def notation(self) -> str:
        """Convert to standard move notation"""
        from_sq = self._pos_to_notation(self.from_pos)
        to_sq = self._pos_to_notation(self.to_pos)
        separator = 'x' if self.is_capture else '-'
        return f"{from_sq}{separator}{to_sq}"

    @staticmethod
    def _pos_to_notation(pos: Tuple[int, int]) -> str:
        """Convert (row, col) to chess-like notation (a1-h8)"""
        row, col = pos
        col_letter = chr(ord('a') + col)
        row_number = 8 - row  # Invert row numbering
        return f"{col_letter}{row_number}"


class BreakthroughBoard:
    """
    Breakthrough board - 8x8 with all squares valid

    Coordinates: (row, col) where row 0 is top, row 7 is bottom
    White starts at rows 0-1 (top), Black starts at rows 6-7 (bottom)
    """

    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.pieces: Set[Piece] = set()
        self.turn = Color.WHITE  # White moves first
        self.move_history: List[Move] = []
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Setup standard breakthrough starting position"""
        # White pieces (top 2 rows)
        for row in range(2):
            for col in range(8):
                piece = Piece(Color.WHITE, (row, col))
                self.board[row][col] = piece
                self.pieces.add(piece)

        # Black pieces (bottom 2 rows)
        for row in range(6, 8):
            for col in range(8):
                piece = Piece(Color.BLACK, (row, col))
                self.board[row][col] = piece
                self.pieces.add(piece)

    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[Piece]:
        """Get piece at position"""
        row, col = pos
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_valid_square(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within board bounds"""
        row, col = pos
        return 0 <= row < 8 and 0 <= col < 8

    def get_pieces(self, color: Color) -> List[Piece]:
        """Get all pieces of a color"""
        return [p for p in self.pieces if p.color == color]

    def make_move(self, move: Move) -> bool:
        """
        Execute a move

        Returns: True if move was made, False if invalid
        """
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos

        piece = self.board[from_row][from_col]
        if not piece or piece != move.piece:
            return False

        # Remove piece from set before changing position (hash changes!)
        self.pieces.discard(piece)

        # Move piece
        self.board[from_row][from_col] = None
        self.board[to_row][to_col] = piece
        piece.position = (to_row, to_col)

        # Remove captured piece if applicable
        if move.captured_piece:
            cap_row, cap_col = move.captured_piece.position
            self.board[cap_row][cap_col] = None
            self.pieces.discard(move.captured_piece)

        # Add piece back to set with new position
        self.pieces.add(piece)

        self.move_history.append(move)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE

        return True

    def is_game_over(self) -> bool:
        """Check if game is over"""
        # No pieces left for one side
        white_pieces = len(self.get_pieces(Color.WHITE))
        black_pieces = len(self.get_pieces(Color.BLACK))

        if white_pieces == 0 or black_pieces == 0:
            return True

        # Check if any piece reached opponent's back row
        for piece in self.pieces:
            if piece.color == Color.WHITE and piece.position[0] == 7:
                return True
            if piece.color == Color.BLACK and piece.position[0] == 0:
                return True

        return False

    def get_winner(self) -> Optional[Color]:
        """Get winner if game is over"""
        if not self.is_game_over():
            return None

        white_pieces = len(self.get_pieces(Color.WHITE))
        black_pieces = len(self.get_pieces(Color.BLACK))

        # Check for pieces captured
        if white_pieces == 0:
            return Color.BLACK
        if black_pieces == 0:
            return Color.WHITE

        # Check if piece reached back row
        for piece in self.pieces:
            if piece.color == Color.WHITE and piece.position[0] == 7:
                return Color.WHITE
            if piece.color == Color.BLACK and piece.position[0] == 0:
                return Color.BLACK

        return None  # No winner

    def copy(self) -> 'BreakthroughBoard':
        """Create a copy of the board"""
        new_board = BreakthroughBoard()
        new_board.board = [[None for _ in range(8)] for _ in range(8)]
        new_board.pieces = set()
        new_board.turn = self.turn
        new_board.move_history = self.move_history.copy()

        # Copy pieces
        for piece in self.pieces:
            new_piece = Piece(piece.color, piece.position)
            row, col = piece.position
            new_board.board[row][col] = new_piece
            new_board.pieces.add(new_piece)

        return new_board

    def to_fen(self) -> str:
        """Convert to FEN-like notation for storage"""
        # Simple notation: each rank separated by /
        # W = white piece, B = black piece, . = empty
        fen_parts = []
        for row in range(8):
            rank = []
            for col in range(8):
                piece = self.board[row][col]
                if piece is None:
                    rank.append('.')
                else:
                    symbol = 'W' if piece.color == Color.WHITE else 'B'
                    rank.append(symbol)
            fen_parts.append(''.join(rank))

        turn = 'w' if self.turn == Color.WHITE else 'b'
        return '/'.join(fen_parts) + f" {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(8)))
        for row in range(8):
            line = f"{8-row} "
            for col in range(8):
                piece = self.board[row][col]
                if piece is None:
                    line += ". "
                else:
                    line += f"{piece} "
            lines.append(line)
        lines.append("  a b c d e f g h")
        lines.append(f"\nTurn: {self.turn.value}")
        return '\n'.join(lines)


def test_breakthrough_board():
    """Test the breakthrough board"""
    print("Testing Breakthrough Board")
    print("=" * 70)

    board = BreakthroughBoard()
    print("\nInitial Position:")
    print(board)

    print(f"\nWhite pieces: {len(board.get_pieces(Color.WHITE))}")
    print(f"Black pieces: {len(board.get_pieces(Color.BLACK))}")

    # Test a simple move
    print("\n" + "=" * 70)
    print("Test Move: White pawn from (1,0) to (2,0)")
    piece = board.get_piece_at((1, 0))
    if piece:
        move = Move(piece, (1, 0), (2, 0), is_capture=False)
        board.make_move(move)
        print(board)
        print(f"Move notation: {move.notation()}")

    print("\n" + "=" * 70)
    print("✓ Board representation working!")


if __name__ == '__main__':
    test_breakthrough_board()
