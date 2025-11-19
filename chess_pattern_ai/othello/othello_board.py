#!/usr/bin/env python3
"""
Othello/Reversi Board Representation - Observation-Based

Philosophy: Minimal hardcoded rules
- Board is 8x8, all squares valid
- Pieces are Black or White discs
- Flipping logic in all 8 directions
- Valid move detection based on flips
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    BLACK = "black"
    WHITE = "white"


@dataclass
class Disc:
    """An Othello disc"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        return 'B' if self.color == Color.BLACK else 'W'

    def __hash__(self):
        """Make disc hashable for use in sets"""
        return hash((self.color, self.position))

    def __eq__(self, other):
        """Equality based on color and position"""
        if not isinstance(other, Disc):
            return False
        return (self.color == other.color and
                self.position == other.position)


@dataclass
class Move:
    """An Othello move"""
    color: Color
    position: Tuple[int, int]  # (row, col) or None for pass
    flipped_discs: Set[Disc] = None  # Discs that will be flipped
    is_pass: bool = False

    def __post_init__(self):
        if self.flipped_discs is None:
            self.flipped_discs = set()

    def notation(self) -> str:
        """Convert to standard notation (e.g., 'a1', 'pass')"""
        if self.is_pass:
            return "PASS"
        row, col = self.position
        col_char = chr(ord('a') + col)
        return f"{col_char}{8 - row}"


class OthelloBoard:
    """
    Othello/Reversi board - 8x8 with Black/White discs

    Coordinates: (row, col) where row 0 is top, row 7 is bottom
    All squares are valid play areas
    """

    # 8 directions: up, down, left, right, and 4 diagonals
    DIRECTIONS = [
        (-1, 0),   # up
        (1, 0),    # down
        (0, -1),   # left
        (0, 1),    # right
        (-1, -1),  # up-left
        (-1, 1),   # up-right
        (1, -1),   # down-left
        (1, 1),    # down-right
    ]

    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.discs: Set[Disc] = set()
        self.turn = Color.BLACK  # Black moves first
        self.move_history: List[Move] = []
        self.passes = 0  # Consecutive passes to detect game over
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Setup standard Othello starting position"""
        # Center 2x2 with alternating colors
        # Row 3: White Black
        # Row 4: Black White
        self._place_disc(Disc(Color.WHITE, (3, 3)))
        self._place_disc(Disc(Color.BLACK, (3, 4)))
        self._place_disc(Disc(Color.BLACK, (4, 3)))
        self._place_disc(Disc(Color.WHITE, (4, 4)))

    def _place_disc(self, disc: Disc):
        """Place a disc on the board"""
        row, col = disc.position
        self.board[row][col] = disc
        self.discs.add(disc)

    def get_disc_at(self, pos: Tuple[int, int]) -> Optional[Disc]:
        """Get disc at position"""
        row, col = pos
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on board"""
        row, col = pos
        return 0 <= row < 8 and 0 <= col < 8

    def get_discs(self, color: Color) -> List[Disc]:
        """Get all discs of a color"""
        return [d for d in self.discs if d.color == color]

    def count_discs(self, color: Color) -> int:
        """Count discs of a color"""
        return len(self.get_discs(color))

    def get_flipped_discs(self, color: Color, pos: Tuple[int, int]) -> Set[Disc]:
        """
        Get all discs that would be flipped if color plays at pos

        Returns empty set if move would be invalid (no discs flipped)
        """
        row, col = pos

        # Position must be empty
        if self.get_disc_at(pos) is not None:
            return set()

        flipped = set()

        # Check all 8 directions
        for d_row, d_col in self.DIRECTIONS:
            # Scan in this direction
            scan_row = row + d_row
            scan_col = col + d_col
            temp_flipped = []

            # Must find at least one opponent disc
            while self.is_valid_position((scan_row, scan_col)):
                target = self.get_disc_at((scan_row, scan_col))

                if target is None:
                    # Empty square - no flips in this direction
                    break

                if target.color == color:
                    # Own disc - flips are valid in this direction
                    flipped.update(temp_flipped)
                    break

                # Opponent disc - might be flipped
                temp_flipped.append(target)

                scan_row += d_row
                scan_col += d_col

        return flipped

    def make_move(self, move: Move) -> bool:
        """
        Execute a move

        Returns: True if move was made, False if invalid
        """
        if move.is_pass:
            self.move_history.append(move)
            self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
            self.passes += 1
            return True

        row, col = move.position

        # Check if position is valid
        if not self.is_valid_position((row, col)):
            return False

        # Check if position is empty
        if self.get_disc_at((row, col)) is not None:
            return False

        # Get flipped discs
        flipped = self.get_flipped_discs(move.color, (row, col))

        # Must flip at least one disc
        if not flipped:
            return False

        # Place new disc
        new_disc = Disc(move.color, (row, col))
        self._place_disc(new_disc)

        # Flip opponent discs
        opponent_color = Color.BLACK if move.color == Color.WHITE else Color.WHITE
        for disc in flipped:
            self.discs.discard(disc)
            flipped_disc = Disc(move.color, disc.position)
            self._place_disc(flipped_disc)
            # Update board
            row_flip, col_flip = disc.position
            self.board[row_flip][col_flip] = flipped_disc

        self.move_history.append(move)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        self.passes = 0  # Reset pass counter on valid move
        return True

    def is_game_over(self) -> bool:
        """Check if game is over"""
        # Game ends when board is full or both players pass
        if self.passes >= 2:
            return True

        # Check if board is full
        empty_count = sum(1 for row in self.board for cell in row if cell is None)
        if empty_count == 0:
            return True

        return False

    def get_winner(self) -> Optional[Color]:
        """Get winner if game is over"""
        if not self.is_game_over():
            return None

        black_count = self.count_discs(Color.BLACK)
        white_count = self.count_discs(Color.WHITE)

        if black_count > white_count:
            return Color.BLACK
        elif white_count > black_count:
            return Color.WHITE
        return None  # Draw

    def copy(self) -> 'OthelloBoard':
        """Create a copy of the board"""
        new_board = OthelloBoard()
        new_board.board = [[None for _ in range(8)] for _ in range(8)]
        new_board.discs = set()
        new_board.turn = self.turn
        new_board.move_history = self.move_history.copy()
        new_board.passes = self.passes

        # Copy discs
        for disc in self.discs:
            new_disc = Disc(disc.color, disc.position)
            row, col = disc.position
            new_board.board[row][col] = new_disc
            new_board.discs.add(new_disc)

        return new_board

    def to_fen(self) -> str:
        """Convert to FEN-like notation for storage"""
        fen_parts = []
        for row in range(8):
            rank = []
            empty_count = 0
            for col in range(8):
                disc = self.board[row][col]
                if disc is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank.append(str(empty_count))
                        empty_count = 0
                    symbol = 'B' if disc.color == Color.BLACK else 'W'
                    rank.append(symbol)
            if empty_count > 0:
                rank.append(str(empty_count))
            fen_parts.append(''.join(rank))

        turn = 'B' if self.turn == Color.BLACK else 'W'
        return '/'.join(fen_parts) + f" {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []
        lines.append("  a b c d e f g h")
        for row in range(8):
            line = f"{8-row} "
            for col in range(8):
                disc = self.board[row][col]
                if disc is None:
                    line += ". "
                else:
                    line += f"{disc} "
            lines.append(line)
        lines.append(f"\nTurn: {self.turn.value}")
        lines.append(f"Black: {self.count_discs(Color.BLACK)}, White: {self.count_discs(Color.WHITE)}")
        return '\n'.join(lines)


def test_othello_board():
    """Test the Othello board"""
    print("Testing Othello Board")
    print("=" * 70)

    board = OthelloBoard()
    print("\nInitial Position:")
    print(board)

    print(f"\nBlack discs: {board.count_discs(Color.BLACK)}")
    print(f"White discs: {board.count_discs(Color.WHITE)}")

    # Test valid moves
    print("\n" + "=" * 70)
    print("Test: Get flipped discs at various positions")

    # Black can play at (3,2), (2,3), (5,4), (4,5) typically
    test_positions = [(3, 2), (2, 3), (5, 4), (4, 5), (0, 0)]

    for pos in test_positions:
        flipped = board.get_flipped_discs(Color.BLACK, pos)
        if flipped:
            print(f"  Position {pos}: Would flip {len(flipped)} disc(s) {flipped}")
        else:
            print(f"  Position {pos}: Invalid move (no flips)")

    # Test a move
    print("\n" + "=" * 70)
    print("Test Move: Black plays at (3,2)")
    move = Move(Color.BLACK, (3, 2))
    move.flipped_discs = board.get_flipped_discs(Color.BLACK, (3, 2))
    if board.make_move(move):
        print(board)
        print(f"Black discs: {board.count_discs(Color.BLACK)}")
        print(f"White discs: {board.count_discs(Color.WHITE)}")

    print("\n" + "=" * 70)
    print("✓ Board representation working!")


if __name__ == '__main__':
    test_othello_board()
