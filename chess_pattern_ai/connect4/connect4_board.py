#!/usr/bin/env python3
"""
Connect Four Board Representation - Observation-Based

Philosophy: Minimal hardcoded rules
- Board is 6x7 (6 rows, 7 columns)
- Pieces: Red or Yellow
- Gravity: Pieces drop to lowest empty row in column
- Win: 4 in a row (horizontal, vertical, or diagonal)
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    RED = "red"
    YELLOW = "yellow"


@dataclass
class Piece:
    """A Connect Four piece"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        color_char = 'R' if self.color == Color.RED else 'Y'
        return color_char

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
    """A Connect Four move"""
    column: int  # Column where piece is dropped
    row: int     # Row where piece lands (computed from gravity)
    piece: Piece

    def notation(self) -> str:
        """Convert to standard notation"""
        return f"Col{self.column+1}→Row{self.row}"


class Connect4Board:
    """
    Connect Four board - 6 rows x 7 columns

    Coordinates: (row, col) where row 0 is top, row 5 is bottom
    Gravity: Pieces drop to lowest empty row in column
    """

    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = [[None for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.pieces: Set[Piece] = set()
        self.turn = Color.RED  # Red goes first
        self.move_history: List[Move] = []

    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[Piece]:
        """Get piece at position"""
        row, col = pos
        if 0 <= row < self.ROWS and 0 <= col < self.COLS:
            return self.board[row][col]
        return None

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on board"""
        row, col = pos
        return 0 <= row < self.ROWS and 0 <= col < self.COLS

    def get_pieces(self, color: Color) -> List[Piece]:
        """Get all pieces of a color"""
        return [p for p in self.pieces if p.color == color]

    def can_place_piece(self, column: int) -> bool:
        """Check if column has space (top row must be empty)"""
        if not (0 <= column < self.COLS):
            return False
        return self.board[0][column] is None

    def get_drop_row(self, column: int) -> Optional[int]:
        """
        Get the row where a piece would land if dropped in this column

        Returns: row index (5 = bottom), or None if column is full
        """
        if not (0 <= column < self.COLS):
            return None

        # Find lowest empty row in column (gravity!)
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][column] is None:
                return row

        return None  # Column is full

    def make_move(self, move: Move) -> bool:
        """
        Execute a move (drop piece in column)

        Returns: True if move was made, False if invalid
        """
        column = move.column
        row = move.row

        # Validate
        if not self.is_valid_position((row, column)):
            return False
        if self.board[row][column] is not None:
            return False

        # Remove piece from set before changing position (hash changes!)
        self.pieces.discard(move.piece)

        # Place piece
        self.board[row][column] = move.piece
        move.piece.position = (row, column)

        # Add piece back to set with new position
        self.pieces.add(move.piece)

        self.move_history.append(move)
        self.turn = Color.YELLOW if self.turn == Color.RED else Color.RED

        return True

    def check_win(self, color: Color) -> bool:
        """
        Check if a color has won (4 in a row)

        Returns: True if color has 4 in a row (horizontal, vertical, or diagonal)
        """
        pieces = self.get_pieces(color)
        piece_positions = set(p.position for p in pieces)

        # Check all positions for each piece
        for piece in pieces:
            row, col = piece.position

            # Directions: horizontal, vertical, diagonal-1, diagonal-2
            directions = [
                (0, 1),   # Horizontal (right)
                (1, 0),   # Vertical (down)
                (1, 1),   # Diagonal (down-right)
                (1, -1)   # Diagonal (down-left)
            ]

            for d_row, d_col in directions:
                # Count consecutive pieces in this direction and opposite
                count = 1  # Starting position

                # Forward direction
                for i in range(1, 4):
                    pos = (row + i * d_row, col + i * d_col)
                    if pos not in piece_positions:
                        break
                    count += 1

                # Backward direction
                for i in range(1, 4):
                    pos = (row - i * d_row, col - i * d_col)
                    if pos not in piece_positions:
                        break
                    count += 1

                if count >= 4:
                    return True

        return False

    def is_game_over(self) -> bool:
        """Check if game is over"""
        # Check for winner
        if self.check_win(Color.RED) or self.check_win(Color.YELLOW):
            return True

        # Check if board is full
        return self._is_board_full()

    def _is_board_full(self) -> bool:
        """Check if board is full (no empty top row cells)"""
        for col in range(self.COLS):
            if self.board[0][col] is None:
                return False
        return True

    def get_winner(self) -> Optional[Color]:
        """Get winner if game is over"""
        if self.check_win(Color.RED):
            return Color.RED
        if self.check_win(Color.YELLOW):
            return Color.YELLOW
        return None

    def copy(self) -> 'Connect4Board':
        """Create a copy of the board"""
        new_board = Connect4Board()
        new_board.board = [[None for _ in range(self.COLS)] for _ in range(self.ROWS)]
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
        fen_parts = []
        for row in range(self.ROWS):
            rank = []
            for col in range(self.COLS):
                piece = self.board[row][col]
                if piece is None:
                    rank.append('.')
                else:
                    symbol = 'R' if piece.color == Color.RED else 'Y'
                    rank.append(symbol)
            fen_parts.append(''.join(rank))

        turn = 'R' if self.turn == Color.RED else 'Y'
        return '/'.join(fen_parts) + f" {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.COLS)))
        for row in range(self.ROWS):
            line = f"{row} "
            for col in range(self.COLS):
                piece = self.board[row][col]
                if piece is None:
                    line += ". "
                else:
                    line += f"{piece} "
            lines.append(line)
        lines.append(f"\nTurn: {self.turn.value}")
        return '\n'.join(lines)


def test_connect4_board():
    """Test the Connect Four board"""
    print("Testing Connect Four Board")
    print("=" * 70)

    board = Connect4Board()
    print("\nInitial Position:")
    print(board)

    print(f"\nRed pieces: {len(board.get_pieces(Color.RED))}")
    print(f"Yellow pieces: {len(board.get_pieces(Color.YELLOW))}")

    # Test gravity and moves
    print("\n" + "=" * 70)
    print("Test 1: Drop red piece in column 0")
    row = board.get_drop_row(0)
    if row is not None:
        piece = Piece(Color.RED, (row, 0))
        move = Move(column=0, row=row, piece=piece)
        if board.make_move(move):
            print(f"Piece dropped to row {row}")
            print(board)

    print("\n" + "=" * 70)
    print("Test 2: Drop yellow piece in column 0")
    row = board.get_drop_row(0)
    if row is not None:
        piece = Piece(Color.YELLOW, (row, 0))
        move = Move(column=0, row=row, piece=piece)
        if board.make_move(move):
            print(f"Piece dropped to row {row}")
            print(board)

    print("\n" + "=" * 70)
    print("Test 3: Drop more red pieces in same column")
    for i in range(2):
        row = board.get_drop_row(0)
        if row is not None:
            piece = Piece(Color.RED, (row, 0))
            move = Move(column=0, row=row, piece=piece)
            board.make_move(move)
    print(board)

    print("\n" + "=" * 70)
    print("✓ Board representation working!")


if __name__ == '__main__':
    test_connect4_board()
