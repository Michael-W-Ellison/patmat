"""
Lines of Action board representation with connectivity checking.

Starting Position (8x8 board):
    0 1 2 3 4 5 6 7
  +-----------------+
0 | . B B B B B B . |
1 | W . . . . . . W |
2 | W . . . . . . W |
3 | W . . . . . . W |
4 | W . . . . . . W |
5 | W . . . . . . W |
6 | W . . . . . . W |
7 | . B B B B B B . |
  +-----------------+

Each player starts with 12 pieces arranged on opposite edges.
Movement is unique: distance = count of all pieces in that line.
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
    position: Tuple[int, int]  # (row, col)

    def __hash__(self):
        return hash((self.color, self.position))

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.color == other.color and self.position == other.position

    def __str__(self):
        return str(self.color)


class LOABoard:
    """
    Lines of Action board with connectivity checking.

    Movement rule: Move along a line (H/V/D) by exactly the number
    of pieces (yours + opponent's) in that line.
    """

    def __init__(self):
        """Initialize board with standard starting position"""
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.pieces_white: Set[Piece] = set()
        self.pieces_black: Set[Piece] = set()
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Set up standard LOA starting position"""
        # Black on top/bottom rows (columns 1-6)
        for col in range(1, 7):
            self._add_piece(0, col, Color.BLACK)
            self._add_piece(7, col, Color.BLACK)

        # White on left/right columns (rows 1-6)
        for row in range(1, 7):
            self._add_piece(row, 0, Color.WHITE)
            self._add_piece(row, 7, Color.WHITE)

    def _add_piece(self, row: int, col: int, color: Color):
        """Add a piece to the board (internal use)"""
        piece = Piece(color, (row, col))
        self.board[row][col] = piece
        if color == Color.WHITE:
            self.pieces_white.add(piece)
        else:
            self.pieces_black.add(piece)

    def copy(self):
        """Create a deep copy of the board"""
        new_board = LOABoard.__new__(LOABoard)
        new_board.board = [row[:] for row in self.board]
        new_board.pieces_white = self.pieces_white.copy()
        new_board.pieces_black = self.pieces_black.copy()
        return new_board

    def get_piece(self, pos: Tuple[int, int]) -> Optional[Piece]:
        """Get piece at position"""
        row, col = pos
        if not self._is_valid_position(pos):
            return None
        return self.board[row][col]

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Move a piece from one position to another.
        Returns True if successful (may capture).
        """
        if not (self._is_valid_position(from_pos) and self._is_valid_position(to_pos)):
            return False

        piece = self.board[from_pos[0]][from_pos[1]]
        if piece is None:
            return False

        target = self.board[to_pos[0]][to_pos[1]]

        # Remove piece from old position
        self.board[from_pos[0]][from_pos[1]] = None
        if piece.color == Color.WHITE:
            self.pieces_white.discard(piece)
        else:
            self.pieces_black.discard(piece)

        # Capture opponent piece if present
        if target is not None:
            if target.color == Color.WHITE:
                self.pieces_white.discard(target)
            else:
                self.pieces_black.discard(target)

        # Place piece at new position
        new_piece = Piece(piece.color, to_pos)
        self.board[to_pos[0]][to_pos[1]] = new_piece
        if piece.color == Color.WHITE:
            self.pieces_white.add(new_piece)
        else:
            self.pieces_black.add(new_piece)

        return True

    def count_pieces_in_line(self, pos: Tuple[int, int], direction: Tuple[int, int]) -> int:
        """
        Count total pieces (both colors) in a line through pos in given direction.

        Direction: (dr, dc) e.g., (0, 1) for horizontal right
        """
        row, col = pos
        dr, dc = direction
        count = 0

        # Count in positive direction
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] is not None:
                count += 1
            r += dr
            c += dc

        # Count in negative direction
        r, c = row - dr, col - dc
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] is not None:
                count += 1
            r -= dr
            c -= dc

        # Count piece at starting position
        if self.board[row][col] is not None:
            count += 1

        return count

    def can_jump_to(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Check if piece can jump from from_pos to to_pos.
        Can jump over own pieces, NOT opponent pieces.
        """
        piece = self.board[from_pos[0]][from_pos[1]]
        if piece is None:
            return False

        # Calculate direction
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        # Normalize to unit direction
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return False

        step_dr = dr // steps
        step_dc = dc // steps

        # Check each position along the path (excluding start and end)
        r, c = from_pos[0] + step_dr, from_pos[1] + step_dc
        for _ in range(steps - 1):
            obstacle = self.board[r][c]
            # Can't jump over opponent pieces
            if obstacle is not None and obstacle.color != piece.color:
                return False
            r += step_dr
            c += step_dc

        # Check landing position
        target = self.board[to_pos[0]][to_pos[1]]
        # Can land on empty or opponent piece (capture)
        return target is None or target.color != piece.color

    def is_connected(self, color: Color) -> bool:
        """
        Check if all pieces of a color are connected in a single group.
        Uses flood fill to check connectivity.
        """
        pieces = self.get_pieces(color)
        if not pieces:
            return True  # No pieces = trivially connected
        if len(pieces) == 1:
            return True  # Single piece = connected

        # Start flood fill from first piece
        start_piece = next(iter(pieces))
        visited = set()
        to_visit = [start_piece.position]

        while to_visit:
            pos = to_visit.pop()
            if pos in visited:
                continue
            visited.add(pos)

            # Check all 8 adjacent positions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    adj_pos = (pos[0] + dr, pos[1] + dc)
                    if not self._is_valid_position(adj_pos):
                        continue
                    if adj_pos in visited:
                        continue

                    adj_piece = self.board[adj_pos[0]][adj_pos[1]]
                    if adj_piece is not None and adj_piece.color == color:
                        to_visit.append(adj_pos)

        # All pieces connected if visited count equals piece count
        return len(visited) == len(pieces)

    def count_groups(self, color: Color) -> int:
        """Count number of separate groups for a color"""
        pieces = self.get_pieces(color)
        if not pieces:
            return 0

        unvisited = {p.position for p in pieces}
        groups = 0

        while unvisited:
            # Start new group
            groups += 1
            start = unvisited.pop()
            to_visit = [start]

            # Flood fill this group
            while to_visit:
                pos = to_visit.pop()

                # Check all 8 adjacent positions
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        adj_pos = (pos[0] + dr, pos[1] + dc)
                        if adj_pos not in unvisited:
                            continue

                        adj_piece = self.board[adj_pos[0]][adj_pos[1]]
                        if adj_piece is not None and adj_piece.color == color:
                            unvisited.discard(adj_pos)
                            to_visit.append(adj_pos)

        return groups

    def get_pieces(self, color: Color) -> Set[Piece]:
        """Get all pieces of a color"""
        return self.pieces_white if color == Color.WHITE else self.pieces_black

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on the board"""
        row, col = pos
        return 0 <= row < 8 and 0 <= col < 8

    def to_string(self) -> str:
        """Convert board to string representation"""
        lines = []
        lines.append("  0 1 2 3 4 5 6 7")
        lines.append("  ---------------")
        for row in range(8):
            row_str = f"{row}|"
            for col in range(8):
                piece = self.board[row][col]
                if piece is None:
                    row_str += ". "
                else:
                    row_str += f"{piece.color.value} "
            lines.append(row_str + "|")
        lines.append("  ---------------")
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()
