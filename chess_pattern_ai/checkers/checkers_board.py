#!/usr/bin/env python3
"""
Checkers Board Representation - Observation-Based

Philosophy: Minimal hardcoded rules
- Board is 8x8, pieces on dark squares only
- Pieces can be "man" or "king"
- Movement rules learned from observation
- Win conditions learned from observation
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class PieceType(Enum):
    MAN = "man"
    KING = "king"


class Color(Enum):
    RED = "red"
    BLACK = "black"


@dataclass
class Piece:
    """A checkers piece"""
    type: PieceType
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        symbol = 'K' if self.type == PieceType.KING else 'M'
        color_char = 'r' if self.color == Color.RED else 'b'
        return f"{color_char}{symbol}"

    def __hash__(self):
        """Make piece hashable for use in sets"""
        return hash((self.type, self.color, self.position))

    def __eq__(self, other):
        """Equality based on type, color, and position"""
        if not isinstance(other, Piece):
            return False
        return (self.type == other.type and
                self.color == other.color and
                self.position == other.position)


@dataclass
class Move:
    """A checkers move"""
    piece: Piece
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    is_jump: bool = False
    captured_pieces: List[Piece] = None  # For multi-jumps

    def __post_init__(self):
        if self.captured_pieces is None:
            self.captured_pieces = []

    def notation(self) -> str:
        """Convert to standard checkers notation"""
        from_sq = self._pos_to_square(self.from_pos)
        to_sq = self._pos_to_square(self.to_pos)
        separator = 'x' if self.is_jump else '-'
        suffix = 'K' if self.piece.type == PieceType.KING else ''
        return f"{from_sq}{separator}{to_sq}{suffix}"

    @staticmethod
    def _pos_to_square(pos: Tuple[int, int]) -> int:
        """Convert (row, col) to square number (1-32)"""
        row, col = pos
        # Only dark squares are numbered
        square = (row * 4) + (col // 2) + 1
        return square


class CheckersBoard:
    """
    Checkers board - 8x8 with pieces on dark squares only

    Coordinates: (row, col) where row 0 is top, row 7 is bottom
    Dark squares only: (row + col) % 2 == 1
    """

    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.pieces: Set[Piece] = set()
        self.turn = Color.RED  # Red moves first (bottom)
        self.move_history: List[Move] = []
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Setup standard checkers starting position"""
        # Black pieces (top 3 rows)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark square
                    piece = Piece(PieceType.MAN, Color.BLACK, (row, col))
                    self.board[row][col] = piece
                    self.pieces.add(piece)

        # Red pieces (bottom 3 rows)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark square
                    piece = Piece(PieceType.MAN, Color.RED, (row, col))
                    self.board[row][col] = piece
                    self.pieces.add(piece)

    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[Piece]:
        """Get piece at position"""
        row, col = pos
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_valid_square(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a valid dark square"""
        row, col = pos
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        return (row + col) % 2 == 1  # Dark square only

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

        # Remove captured pieces
        for captured in move.captured_pieces:
            cap_row, cap_col = captured.position
            self.board[cap_row][cap_col] = None
            self.pieces.discard(captured)

        # Check for promotion (man reaches opposite end)
        if piece.type == PieceType.MAN:
            if (piece.color == Color.RED and to_row == 0) or \
               (piece.color == Color.BLACK and to_row == 7):
                piece.type = PieceType.KING

        # Add piece back to set with new position/type
        self.pieces.add(piece)

        self.move_history.append(move)
        self.turn = Color.BLACK if self.turn == Color.RED else Color.RED

        return True

    def is_game_over(self) -> bool:
        """Check if game is over"""
        # No pieces left for one side
        red_pieces = len(self.get_pieces(Color.RED))
        black_pieces = len(self.get_pieces(Color.BLACK))

        if red_pieces == 0 or black_pieces == 0:
            return True

        # No legal moves (this will be learned from observation)
        # For now, just check piece count
        return False

    def get_winner(self) -> Optional[Color]:
        """Get winner if game is over"""
        if not self.is_game_over():
            return None

        red_pieces = len(self.get_pieces(Color.RED))
        black_pieces = len(self.get_pieces(Color.BLACK))

        if red_pieces > black_pieces:
            return Color.RED
        elif black_pieces > red_pieces:
            return Color.BLACK
        return None  # Draw

    def copy(self) -> 'CheckersBoard':
        """Create a copy of the board"""
        new_board = CheckersBoard()
        new_board.board = [[None for _ in range(8)] for _ in range(8)]
        new_board.pieces = set()
        new_board.turn = self.turn
        new_board.move_history = self.move_history.copy()

        # Copy pieces
        for piece in self.pieces:
            new_piece = Piece(piece.type, piece.color, piece.position)
            row, col = piece.position
            new_board.board[row][col] = new_piece
            new_board.pieces.add(new_piece)

        return new_board

    def to_fen(self) -> str:
        """Convert to FEN-like notation for storage"""
        # Simple notation: each rank separated by /
        # M = man, K = king, r/b = red/black, . = empty dark square
        fen_parts = []
        for row in range(8):
            rank = []
            for col in range(8):
                if (row + col) % 2 == 0:  # Light square
                    continue
                piece = self.board[row][col]
                if piece is None:
                    rank.append('.')
                else:
                    symbol = 'K' if piece.type == PieceType.KING else 'M'
                    color = 'r' if piece.color == Color.RED else 'b'
                    rank.append(f"{color}{symbol}")
            fen_parts.append(''.join(rank))

        turn = 'r' if self.turn == Color.RED else 'b'
        return '/'.join(fen_parts) + f" {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(8)))
        for row in range(8):
            line = f"{row} "
            for col in range(8):
                if (row + col) % 2 == 0:
                    line += "  "  # Light square
                else:
                    piece = self.board[row][col]
                    if piece is None:
                        line += ". "
                    else:
                        line += f"{piece} "
            lines.append(line)
        lines.append(f"\nTurn: {self.turn.value}")
        return '\n'.join(lines)


def test_checkers_board():
    """Test the checkers board"""
    print("Testing Checkers Board")
    print("=" * 70)

    board = CheckersBoard()
    print("\nInitial Position:")
    print(board)

    print(f"\nRed pieces: {len(board.get_pieces(Color.RED))}")
    print(f"Black pieces: {len(board.get_pieces(Color.BLACK))}")

    # Test a simple move
    print("\n" + "=" * 70)
    print("Test Move: Red man from (5,0) to (4,1)")
    piece = board.get_piece_at((5, 0))
    if piece:
        move = Move(piece, (5, 0), (4, 1), is_jump=False)
        board.make_move(move)
        print(board)
        print(f"Move notation: {move.notation()}")

    print("\n" + "=" * 70)
    print("✓ Board representation working!")


if __name__ == '__main__':
    test_checkers_board()
