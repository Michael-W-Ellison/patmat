#!/usr/bin/env python3
"""
Gomoku Board Representation - Observation-Based

Philosophy: Learn strategy from observation
- Board is 15x15 or 19x19 (standard sizes)
- Black and White stones
- 5-in-a-row wins (horizontal, vertical, both diagonals)
- No captures (unlike Go)
- Hashable Stone class for pattern tracking
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    BLACK = "black"
    WHITE = "white"


@dataclass(frozen=True)
class Stone:
    """A Gomoku stone (immutable for hashing)"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        color_char = 'B' if self.color == Color.BLACK else 'W'
        return f"{color_char}"

    def __hash__(self):
        """Make stone hashable for use in sets and dicts"""
        return hash((self.color, self.position))

    def __eq__(self, other):
        """Equality based on color and position"""
        if not isinstance(other, Stone):
            return False
        return (self.color == other.color and
                self.position == other.position)


@dataclass
class Move:
    """A Gomoku move"""
    position: Tuple[int, int]  # (row, col)
    color: Color
    stone: Stone

    def notation(self) -> str:
        """Convert to standard notation"""
        row, col = self.position
        return f"{chr(ord('a') + col)}{row+1}"


class GomokuBoard:
    """
    Gomoku board - 15x15 or 19x19

    Coordinates: (row, col) where row 0 is top, row N-1 is bottom
    No captures (unlike Go)
    Win condition: 5 in a row (horizontal, vertical, or diagonal)
    """

    def __init__(self, size: int = 15):
        """
        Initialize Gomoku board

        Args:
            size: Board size (15 or 19, standard Gomoku board sizes)

        Raises:
            ValueError: If size is not 15 or 19
        """
        if size not in [15, 19]:
            raise ValueError("Gomoku board size must be 15 or 19")

        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.stones: Set[Stone] = set()
        self.turn = Color.BLACK  # Black plays first
        self.move_history: List[Move] = []

    def get_stone_at(self, pos: Tuple[int, int]) -> Optional[Stone]:
        """Get stone at position"""
        row, col = pos
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.board[row][col]
        return None

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on board"""
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size

    def get_stones(self, color: Color) -> List[Stone]:
        """Get all stones of a color"""
        return [s for s in self.stones if s.color == color]

    def is_empty(self, pos: Tuple[int, int]) -> bool:
        """Check if position is empty"""
        return self.get_stone_at(pos) is None

    def can_place_stone(self, pos: Tuple[int, int]) -> bool:
        """Check if a stone can be placed at position"""
        if not self.is_valid_position(pos):
            return False
        return self.is_empty(pos)

    def make_move(self, move: Move) -> bool:
        """
        Execute a move (place stone on board)

        Returns: True if move was made, False if invalid
        """
        row, col = move.position

        # Validate
        if not self.is_valid_position((row, col)):
            return False
        if not self.is_empty((row, col)):
            return False

        # Place stone
        stone = move.stone
        self.board[row][col] = stone
        self.stones.add(stone)

        self.move_history.append(move)
        self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK

        return True

    def check_win(self, color: Color) -> bool:
        """
        Check if a color has won (5 in a row)

        Returns: True if color has 5 in a row (horizontal, vertical, or diagonal)
        """
        stones = self.get_stones(color)
        stone_positions = set(s.position for s in stones)

        # Check all positions for each stone
        for stone in stones:
            row, col = stone.position

            # Directions: horizontal, vertical, diagonal-1, diagonal-2
            directions = [
                (0, 1),   # Horizontal (right)
                (1, 0),   # Vertical (down)
                (1, 1),   # Diagonal (down-right)
                (1, -1)   # Diagonal (down-left)
            ]

            for d_row, d_col in directions:
                # Count consecutive stones in this direction and opposite
                count = 1  # Starting position

                # Forward direction
                for i in range(1, 5):
                    pos = (row + i * d_row, col + i * d_col)
                    if pos not in stone_positions:
                        break
                    count += 1

                # Backward direction
                for i in range(1, 5):
                    pos = (row - i * d_row, col - i * d_col)
                    if pos not in stone_positions:
                        break
                    count += 1

                if count >= 5:
                    return True

        return False

    def is_game_over(self) -> bool:
        """Check if game is over"""
        # Check for winner
        if self.check_win(Color.BLACK) or self.check_win(Color.WHITE):
            return True

        # Check if board is full
        return self._is_board_full()

    def _is_board_full(self) -> bool:
        """Check if board is full (all intersections occupied)"""
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] is None:
                    return False
        return True

    def get_winner(self) -> Optional[Color]:
        """Get winner if game is over"""
        if self.check_win(Color.BLACK):
            return Color.BLACK
        if self.check_win(Color.WHITE):
            return Color.WHITE
        return None

    def copy(self) -> 'GomokuBoard':
        """Create a copy of the board"""
        new_board = GomokuBoard(self.size)
        new_board.board = [[None for _ in range(self.size)] for _ in range(self.size)]
        new_board.stones = set()
        new_board.turn = self.turn
        new_board.move_history = self.move_history.copy()

        # Copy stones
        for stone in self.stones:
            row, col = stone.position
            new_board.board[row][col] = stone
            new_board.stones.add(stone)

        return new_board

    def to_fen(self) -> str:
        """Convert to FEN-like notation for storage"""
        fen_parts = []
        for row in range(self.size):
            rank = []
            for col in range(self.size):
                stone = self.board[row][col]
                if stone is None:
                    rank.append('.')
                else:
                    symbol = 'B' if stone.color == Color.BLACK else 'W'
                    rank.append(symbol)
            fen_parts.append(''.join(rank))

        turn = 'B' if self.turn == Color.BLACK else 'W'
        return '/'.join(fen_parts) + f" {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []

        # Column labels
        col_labels = []
        for col in range(self.size):
            col_labels.append(chr(ord('a') + col))
        lines.append("   " + " ".join(col_labels))

        # Rows
        for row in range(self.size):
            line = f"{row+1:2d} "
            for col in range(self.size):
                stone = self.board[row][col]
                if stone is None:
                    line += ". "
                else:
                    line += f"{stone} "
            lines.append(line)

        lines.append(f"\nTurn: {self.turn.value}")
        return '\n'.join(lines)


def test_gomoku_board():
    """Test the Gomoku board"""
    print("Testing Gomoku Board")
    print("=" * 70)

    # Test 15x15 board
    board = GomokuBoard(15)
    print("\n15x15 Board - Initial Position:")
    print(board)

    print(f"\nBlack stones: {len(board.get_stones(Color.BLACK))}")
    print(f"White stones: {len(board.get_stones(Color.WHITE))}")

    # Test stone placement
    print("\n" + "=" * 70)
    print("Test 1: Place black stone at d8")
    stone = Stone(Color.BLACK, (7, 3))  # d8 (0-indexed)
    move = Move(position=(7, 3), color=Color.BLACK, stone=stone)
    if board.make_move(move):
        print("Stone placed successfully")
        print(board)

    print("\n" + "=" * 70)
    print("Test 2: Place white stone at e8")
    stone = Stone(Color.WHITE, (7, 4))
    move = Move(position=(7, 4), color=Color.WHITE, stone=stone)
    if board.make_move(move):
        print("Stone placed successfully")
        print(board)

    # Test 5-in-a-row detection
    print("\n" + "=" * 70)
    print("Test 3: Create horizontal 5-in-a-row for black")
    board2 = GomokuBoard(15)
    for col in range(5):
        stone = Stone(Color.BLACK, (7, col))
        move = Move(position=(7, col), color=Color.BLACK, stone=stone)
        board2.make_move(move)

    print(board2)
    print(f"Black has 5-in-a-row: {board2.check_win(Color.BLACK)}")

    # Test stone hashability
    print("\n" + "=" * 70)
    print("Test 4: Stone hashability (for pattern tracking)")
    s1 = Stone(Color.BLACK, (0, 0))
    s2 = Stone(Color.BLACK, (0, 0))
    s3 = Stone(Color.BLACK, (0, 1))
    stone_set = {s1, s2, s3}
    print(f"Created 3 stone objects (2 identical, 1 different)")
    print(f"Set contains {len(stone_set)} unique stones (should be 2)")
    print(f"Stones are hashable and can be used in sets: {len(stone_set) == 2}")

    print("\n" + "=" * 70)
    print("✓ Gomoku board implementation working!")


if __name__ == '__main__':
    test_gomoku_board()
