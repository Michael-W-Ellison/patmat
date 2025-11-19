"""
Nine Men's Morris board representation with mill detection.

Board Layout (24 positions):

    0 -------- 1 -------- 2
    |          |          |
    |  8 ----- 9 ----- 10 |
    |  |       |       |  |
    |  | 16 - 17 - 18  |  |
    |  |  |         |  |  |
    7-15-23         19-11- 3
    |  |  |         |  |  |
    |  | 22 - 21 - 20  |  |
    |  |       |       |  |
    |  14 --- 13 --- 12  |
    |          |          |
    6 -------- 5 -------- 4

Positions 0-23 arranged in 3 concentric squares:
- Outer ring: 0, 1, 2, 3, 4, 5, 6, 7
- Middle ring: 8, 9, 10, 11, 12, 13, 14, 15
- Inner ring: 16, 17, 18, 19, 20, 21, 22, 23
"""

from typing import Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """Piece colors"""
    WHITE = "W"
    BLACK = "B"

    def __str__(self):
        return self.value

    def opposite(self):
        """Get opposite color"""
        return Color.BLACK if self == Color.WHITE else Color.WHITE


@dataclass(frozen=True)
class Piece:
    """Immutable piece representation"""
    color: Color
    position: int  # 0-23

    def __hash__(self):
        return hash((self.color, self.position))

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.color == other.color and self.position == other.position

    def __str__(self):
        return str(self.color)


class MorrisBoard:
    """
    Nine Men's Morris board with 24 positions.

    Board has 3 concentric squares connected by lines.
    Mills (3 in a row) allow removing opponent pieces.
    """

    # Define adjacency graph (which positions connect to which)
    ADJACENCIES = {
        0: [1, 7], 1: [0, 2, 9], 2: [1, 3],
        3: [2, 11], 4: [5, 11], 5: [4, 6, 13],
        6: [5, 7], 7: [0, 6, 15], 8: [9, 15],
        9: [1, 8, 10, 17], 10: [9, 11], 11: [3, 4, 10, 19],
        12: [13, 19], 13: [5, 12, 14, 21], 14: [13, 15],
        15: [7, 8, 14, 23], 16: [17, 23], 17: [9, 16, 18],
        18: [17, 19], 19: [11, 12, 18], 20: [21, 19],
        21: [13, 20, 22], 22: [21, 23], 23: [15, 16, 22]
    }

    # Define all possible mills (3 in a row)
    MILLS = [
        # Outer square
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],
        # Middle square
        [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8],
        # Inner square
        [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16],
        # Connecting lines (vertical)
        [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23]
    ]

    def __init__(self):
        """Initialize empty board"""
        self.positions = [None] * 24  # 24 positions, None = empty
        self.pieces_white = []
        self.pieces_black = []

    def copy(self):
        """Create a deep copy of the board"""
        new_board = MorrisBoard()
        new_board.positions = self.positions.copy()
        new_board.pieces_white = self.pieces_white.copy()
        new_board.pieces_black = self.pieces_black.copy()
        return new_board

    def get_piece(self, pos: int) -> Optional[Piece]:
        """Get piece at position (0-23)"""
        if not (0 <= pos < 24):
            return None
        return self.positions[pos]

    def place_piece(self, pos: int, color: Color) -> bool:
        """
        Place a piece at position.
        Returns True if successful, False if position occupied.
        """
        if not (0 <= pos < 24):
            return False
        if self.positions[pos] is not None:
            return False

        piece = Piece(color, pos)
        self.positions[pos] = piece

        if color == Color.WHITE:
            self.pieces_white.append(piece)
        else:
            self.pieces_black.append(piece)

        return True

    def move_piece(self, from_pos: int, to_pos: int) -> bool:
        """
        Move a piece from one position to another.
        Returns True if successful.
        """
        if not (0 <= from_pos < 24 and 0 <= to_pos < 24):
            return False

        piece = self.positions[from_pos]
        if piece is None:
            return False
        if self.positions[to_pos] is not None:
            return False

        # Update board
        self.positions[from_pos] = None
        new_piece = Piece(piece.color, to_pos)
        self.positions[to_pos] = new_piece

        # Update piece lists
        if piece.color == Color.WHITE:
            self.pieces_white.remove(piece)
            self.pieces_white.append(new_piece)
        else:
            self.pieces_black.remove(piece)
            self.pieces_black.append(new_piece)

        return True

    def remove_piece(self, pos: int) -> bool:
        """
        Remove a piece from position.
        Returns True if successful.
        """
        if not (0 <= pos < 24):
            return False

        piece = self.positions[pos]
        if piece is None:
            return False

        self.positions[pos] = None

        if piece.color == Color.WHITE:
            self.pieces_white.remove(piece)
        else:
            self.pieces_black.remove(piece)

        return True

    def get_adjacent_positions(self, pos: int) -> List[int]:
        """Get list of positions adjacent to pos"""
        return self.ADJACENCIES.get(pos, [])

    def is_mill(self, pos: int, color: Color) -> bool:
        """
        Check if placing/moving to position completes a mill for color.
        A mill is 3 of the same color in a row.
        """
        for mill in self.MILLS:
            if pos in mill:
                # Check if all 3 positions in mill have pieces of this color
                if all(self.positions[p] is not None and
                       self.positions[p].color == color
                       for p in mill):
                    return True
        return False

    def count_mills(self, color: Color) -> int:
        """Count number of complete mills for a color"""
        count = 0
        for mill in self.MILLS:
            if all(self.positions[p] is not None and
                   self.positions[p].color == color
                   for p in mill):
                count += 1
        return count

    def count_potential_mills(self, color: Color, pieces_needed: int) -> int:
        """
        Count potential mills where color has (3 - pieces_needed) pieces
        and the remaining positions are empty.

        pieces_needed=1: Has 2 pieces, needs 1 more (strong threat)
        pieces_needed=2: Has 1 piece, needs 2 more (weak threat)
        """
        count = 0
        for mill in self.MILLS:
            my_pieces = sum(1 for p in mill
                          if self.positions[p] is not None and
                          self.positions[p].color == color)
            opp_pieces = sum(1 for p in mill
                           if self.positions[p] is not None and
                           self.positions[p].color == color.opposite())

            # Potential mill if we have (3-pieces_needed) pieces and no opponent pieces
            if my_pieces == (3 - pieces_needed) and opp_pieces == 0:
                count += 1

        return count

    def get_pieces(self, color: Color) -> List[Piece]:
        """Get all pieces of a color"""
        return self.pieces_white if color == Color.WHITE else self.pieces_black

    def get_empty_positions(self) -> List[int]:
        """Get all empty positions"""
        return [i for i in range(24) if self.positions[i] is None]

    def count_pieces(self, color: Color) -> int:
        """Count pieces of a color on board"""
        return len(self.get_pieces(color))

    def get_removable_pieces(self, color: Color) -> List[int]:
        """
        Get opponent pieces that can be removed.
        Pieces in mills can only be removed if all opponent pieces are in mills.
        """
        opponent_pieces = self.get_pieces(color.opposite())
        opponent_positions = [p.position for p in opponent_pieces]

        # Find pieces not in mills
        not_in_mill = []
        in_mill = []

        for pos in opponent_positions:
            if self.is_mill(pos, color.opposite()):
                in_mill.append(pos)
            else:
                not_in_mill.append(pos)

        # If any pieces are not in mills, only those can be removed
        if not_in_mill:
            return not_in_mill
        else:
            # All pieces are in mills, can remove any
            return in_mill

    def to_string(self) -> str:
        """Convert board to string representation"""
        def piece_str(pos):
            p = self.positions[pos]
            return p.color.value if p else str(pos) if pos < 10 else chr(ord('A') + pos - 10)

        lines = []
        lines.append(f"{piece_str(0)} -------- {piece_str(1)} -------- {piece_str(2)}")
        lines.append(f"|          |          |")
        lines.append(f"|  {piece_str(8)} ----- {piece_str(9)} ----- {piece_str(10)}  |")
        lines.append(f"|  |       |       |  |")
        lines.append(f"|  | {piece_str(16)} -- {piece_str(17)} -- {piece_str(18)}  |  |")
        lines.append(f"|  |  |         |  |  |")
        lines.append(f"{piece_str(7)}-{piece_str(15)}-{piece_str(23)}         {piece_str(19)}-{piece_str(11)}- {piece_str(3)}")
        lines.append(f"|  |  |         |  |  |")
        lines.append(f"|  | {piece_str(22)} -- {piece_str(21)} -- {piece_str(20)}  |  |")
        lines.append(f"|  |       |       |  |")
        lines.append(f"|  {piece_str(14)} ---- {piece_str(13)} ---- {piece_str(12)}  |")
        lines.append(f"|          |          |")
        lines.append(f"{piece_str(6)} -------- {piece_str(5)} -------- {piece_str(4)}")

        return "\n".join(lines)

    def __str__(self):
        return self.to_string()
