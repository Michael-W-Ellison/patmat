"""
Pentago board representation with quadrant rotation mechanics.

Board layout:
  0 1 2 | 3 4 5
  ----- + -----
0 . . . | . . .    Quadrant 0 | Quadrant 1
1 . . . | . . .    (top-left) | (top-right)
2 . . . | . . .                |
  ----- + -----    -----------+-----------
3 . . . | . . .    Quadrant 2 | Quadrant 3
4 . . . | . . .    (bot-left) | (bot-right)
5 . . . | . . .                |
"""

from typing import Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import copy


class Color(Enum):
    """Marble colors"""
    WHITE = "W"
    BLACK = "B"

    def __str__(self):
        return self.value

    def opposite(self):
        """Get opposite color"""
        return Color.BLACK if self == Color.WHITE else Color.WHITE


@dataclass(frozen=True)
class Marble:
    """Immutable marble representation"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __hash__(self):
        return hash((self.color, self.position))

    def __eq__(self, other):
        if not isinstance(other, Marble):
            return False
        return self.color == other.color and self.position == other.position

    def __str__(self):
        return str(self.color)


class PentagoBoard:
    """
    Pentago board with quadrant rotation.

    Board is 6x6 divided into 4 quadrants (each 3x3).
    Quadrants numbered:
        0 | 1
        -----
        2 | 3
    """

    def __init__(self):
        """Initialize empty 6x6 board"""
        self.board: List[List[Optional[Marble]]] = [[None for _ in range(6)] for _ in range(6)]
        self.marble_count = 0

    def copy(self):
        """Create a deep copy of the board"""
        new_board = PentagoBoard()
        new_board.board = copy.deepcopy(self.board)
        new_board.marble_count = self.marble_count
        return new_board

    def get_marble(self, pos: Tuple[int, int]) -> Optional[Marble]:
        """Get marble at position"""
        row, col = pos
        if not self._is_valid_position(pos):
            return None
        return self.board[row][col]

    def place_marble(self, pos: Tuple[int, int], color: Color) -> bool:
        """
        Place a marble at position.
        Returns True if successful, False if position occupied.
        """
        row, col = pos
        if not self._is_valid_position(pos):
            return False
        if self.board[row][col] is not None:
            return False

        self.board[row][col] = Marble(color, pos)
        self.marble_count += 1
        return True

    def rotate_quadrant(self, quadrant: int, direction: str):
        """
        Rotate a quadrant 90° clockwise (CW) or counter-clockwise (CCW).

        Quadrants:
            0 | 1
            -----
            2 | 3
        """
        if quadrant not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid quadrant: {quadrant}")
        if direction not in ['CW', 'CCW']:
            raise ValueError(f"Invalid direction: {direction}")

        # Get quadrant bounds
        row_start, col_start = self._get_quadrant_origin(quadrant)

        # Extract 3x3 quadrant
        quad = [[self.board[row_start + r][col_start + c] for c in range(3)] for r in range(3)]

        # Rotate 90° clockwise or counter-clockwise
        if direction == 'CW':
            rotated = self._rotate_clockwise(quad)
        else:  # CCW
            rotated = self._rotate_counterclockwise(quad)

        # Write back to board
        for r in range(3):
            for c in range(3):
                self.board[row_start + r][col_start + c] = rotated[r][c]

                # Update marble positions
                if rotated[r][c] is not None:
                    new_pos = (row_start + r, col_start + c)
                    # Create new marble with updated position
                    self.board[row_start + r][col_start + c] = Marble(
                        rotated[r][c].color,
                        new_pos
                    )

    def _rotate_clockwise(self, quad: List[List]) -> List[List]:
        """Rotate 3x3 matrix 90° clockwise"""
        return [[quad[2-c][r] for c in range(3)] for r in range(3)]

    def _rotate_counterclockwise(self, quad: List[List]) -> List[List]:
        """Rotate 3x3 matrix 90° counter-clockwise"""
        return [[quad[c][2-r] for c in range(3)] for r in range(3)]

    def _get_quadrant_origin(self, quadrant: int) -> Tuple[int, int]:
        """Get top-left position of quadrant"""
        origins = {
            0: (0, 0),  # top-left
            1: (0, 3),  # top-right
            2: (3, 0),  # bottom-left
            3: (3, 3),  # bottom-right
        }
        return origins[quadrant]

    def get_quadrant(self, pos: Tuple[int, int]) -> int:
        """Get which quadrant a position belongs to"""
        row, col = pos
        if row < 3 and col < 3:
            return 0
        elif row < 3 and col >= 3:
            return 1
        elif row >= 3 and col < 3:
            return 2
        else:
            return 3

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on the board"""
        row, col = pos
        return 0 <= row < 6 and 0 <= col < 6

    def is_full(self) -> bool:
        """Check if board is full"""
        return self.marble_count >= 36

    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty positions"""
        empty = []
        for row in range(6):
            for col in range(6):
                if self.board[row][col] is None:
                    empty.append((row, col))
        return empty

    def get_marbles(self, color: Color) -> List[Marble]:
        """Get all marbles of a color"""
        marbles = []
        for row in range(6):
            for col in range(6):
                marble = self.board[row][col]
                if marble is not None and marble.color == color:
                    marbles.append(marble)
        return marbles

    def check_five_in_row(self, color: Color) -> bool:
        """Check if color has 5 in a row (horizontal, vertical, or diagonal)"""
        # Horizontal
        for row in range(6):
            for col in range(2):  # Start at col 0-1 (need 5 spaces)
                if all(self.board[row][col + i] is not None and
                       self.board[row][col + i].color == color
                       for i in range(5)):
                    return True

        # Vertical
        for col in range(6):
            for row in range(2):  # Start at row 0-1
                if all(self.board[row + i][col] is not None and
                       self.board[row + i][col].color == color
                       for i in range(5)):
                    return True

        # Diagonal (top-left to bottom-right)
        for row in range(2):
            for col in range(2):
                if all(self.board[row + i][col + i] is not None and
                       self.board[row + i][col + i].color == color
                       for i in range(5)):
                    return True

        # Diagonal (top-right to bottom-left)
        for row in range(2):
            for col in range(4, 6):
                if all(self.board[row + i][col - i] is not None and
                       self.board[row + i][col - i].color == color
                       for i in range(5)):
                    return True

        return False

    def count_threats(self, color: Color, length: int) -> int:
        """
        Count number of potential N-in-a-row threats for a color.
        A threat is N consecutive marbles (possibly with gaps) that could become 5.
        """
        count = 0

        # Check all possible 5-marble windows
        # Horizontal
        for row in range(6):
            for col in range(2):
                marbles_in_window = [self.board[row][col + i] for i in range(5)]
                if self._is_threat(marbles_in_window, color, length):
                    count += 1

        # Vertical
        for col in range(6):
            for row in range(2):
                marbles_in_window = [self.board[row + i][col] for i in range(5)]
                if self._is_threat(marbles_in_window, color, length):
                    count += 1

        # Diagonal (TL to BR)
        for row in range(2):
            for col in range(2):
                marbles_in_window = [self.board[row + i][col + i] for i in range(5)]
                if self._is_threat(marbles_in_window, color, length):
                    count += 1

        # Diagonal (TR to BL)
        for row in range(2):
            for col in range(4, 6):
                marbles_in_window = [self.board[row + i][col - i] for i in range(5)]
                if self._is_threat(marbles_in_window, color, length):
                    count += 1

        return count

    def _is_threat(self, window: List[Optional[Marble]], color: Color, length: int) -> bool:
        """Check if a 5-marble window contains a threat of given length"""
        my_marbles = sum(1 for m in window if m is not None and m.color == color)
        opponent_marbles = sum(1 for m in window if m is not None and m.color == color.opposite())

        # Threat if we have 'length' marbles and opponent has none
        return my_marbles >= length and opponent_marbles == 0

    def to_string(self) -> str:
        """Convert board to string representation"""
        lines = []
        lines.append("  0 1 2   3 4 5")
        lines.append("  -----   -----")
        for row in range(6):
            row_str = f"{row} "
            for col in range(6):
                marble = self.board[row][col]
                if marble is None:
                    row_str += ". "
                else:
                    row_str += f"{marble.color.value} "
                if col == 2:
                    row_str += "  "
            lines.append(row_str)
            if row == 2:
                lines.append("  -----   -----")
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()
