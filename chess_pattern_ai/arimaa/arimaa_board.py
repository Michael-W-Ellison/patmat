"""
Arimaa board representation with push/pull mechanics and traps.

Starting Position (8x8 board):
    0 1 2 3 4 5 6 7
  +-----------------+
0 | r r r r r r r r | ← Gold rabbits
1 | d h c e m c h d | ← Gold stronger pieces
2 | . . X . . X . . |   X = trap
3 | . . . . . . . . |
4 | . . . . . . . . |
5 | . . X . . X . . |   X = trap
6 | D H C M E C H D | ← Silver stronger pieces
7 | R R R R R R R R | ← Silver rabbits
  +-----------------+

Piece Strength: Elephant (E) > Camel (M) > Horse (H) > Dog (D) > Cat (C) > Rabbit (R)
Traps at: (2,2), (2,5), (5,2), (5,5)
"""

from typing import Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """Piece colors"""
    GOLD = "gold"    # Lowercase pieces
    SILVER = "silver"  # Uppercase pieces

    def __str__(self):
        return self.value

    def opposite(self):
        """Get opposite color"""
        return Color.SILVER if self == Color.GOLD else Color.GOLD


class PieceType(Enum):
    """Piece types with strength values"""
    RABBIT = ("R", 1)
    CAT = ("C", 2)
    DOG = ("D", 3)
    HORSE = ("H", 4)
    CAMEL = ("M", 5)
    ELEPHANT = ("E", 6)

    def __init__(self, symbol: str, strength: int):
        self.symbol = symbol
        self.strength = strength

    def __str__(self):
        return self.symbol

    def __lt__(self, other):
        """Compare piece strengths"""
        if not isinstance(other, PieceType):
            return NotImplemented
        return self.strength < other.strength


@dataclass(frozen=True)
class Piece:
    """Immutable piece representation"""
    type: PieceType
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __hash__(self):
        return hash((self.type, self.color, self.position))

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return (self.type == other.type and
                self.color == other.color and
                self.position == other.position)

    def __str__(self):
        """Display piece (lowercase=gold, uppercase=silver)"""
        symbol = self.type.symbol.lower() if self.color == Color.GOLD else self.type.symbol.upper()
        return symbol

    def is_stronger_than(self, other: 'Piece') -> bool:
        """Check if this piece is stronger than another"""
        return self.type.strength > other.type.strength


class ArimaaBoard:
    """
    Arimaa board with push/pull mechanics and traps.

    Trap squares: (2,2), (2,5), (5,2), (5,5)
    Pieces die on traps unless adjacent to friendly piece.
    """

    TRAP_SQUARES = [(2, 2), (2, 5), (5, 2), (5, 5)]

    def __init__(self):
        """Initialize board with standard starting position"""
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.pieces_gold: Set[Piece] = set()
        self.pieces_silver: Set[Piece] = set()
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Set up standard Arimaa starting position"""
        # Gold (row 0-1, lowercase)
        gold_setup = [
            # Row 0: 8 rabbits
            [(0, i, PieceType.RABBIT) for i in range(8)],
            # Row 1: d h c e m c h d (standard setup)
            [(1, 0, PieceType.DOG), (1, 1, PieceType.HORSE), (1, 2, PieceType.CAT),
             (1, 3, PieceType.ELEPHANT), (1, 4, PieceType.CAMEL), (1, 5, PieceType.CAT),
             (1, 6, PieceType.HORSE), (1, 7, PieceType.DOG)]
        ]

        for row_pieces in gold_setup:
            for row, col, piece_type in row_pieces:
                self._add_piece(row, col, piece_type, Color.GOLD)

        # Silver (row 6-7, uppercase)
        silver_setup = [
            # Row 6: D H C M E C H D (standard setup)
            [(6, 0, PieceType.DOG), (6, 1, PieceType.HORSE), (6, 2, PieceType.CAT),
             (6, 3, PieceType.CAMEL), (6, 4, PieceType.ELEPHANT), (6, 5, PieceType.CAT),
             (6, 6, PieceType.HORSE), (6, 7, PieceType.DOG)],
            # Row 7: 8 rabbits
            [(7, i, PieceType.RABBIT) for i in range(8)]
        ]

        for row_pieces in silver_setup:
            for row, col, piece_type in row_pieces:
                self._add_piece(row, col, piece_type, Color.SILVER)

    def _add_piece(self, row: int, col: int, piece_type: PieceType, color: Color):
        """Add a piece to the board (internal use)"""
        piece = Piece(piece_type, color, (row, col))
        self.board[row][col] = piece
        if color == Color.GOLD:
            self.pieces_gold.add(piece)
        else:
            self.pieces_silver.add(piece)

    def copy(self):
        """Create a deep copy of the board"""
        new_board = ArimaaBoard.__new__(ArimaaBoard)
        new_board.board = [row[:] for row in self.board]
        new_board.pieces_gold = self.pieces_gold.copy()
        new_board.pieces_silver = self.pieces_silver.copy()
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
        Returns True if successful.
        """
        if not (self._is_valid_position(from_pos) and self._is_valid_position(to_pos)):
            return False

        piece = self.board[from_pos[0]][from_pos[1]]
        if piece is None:
            return False
        if self.board[to_pos[0]][to_pos[1]] is not None:
            return False  # Destination must be empty

        # Remove piece from old position
        self.board[from_pos[0]][from_pos[1]] = None
        if piece.color == Color.GOLD:
            self.pieces_gold.discard(piece)
        else:
            self.pieces_silver.discard(piece)

        # Place piece at new position
        new_piece = Piece(piece.type, piece.color, to_pos)
        self.board[to_pos[0]][to_pos[1]] = new_piece
        if piece.color == Color.GOLD:
            self.pieces_gold.add(new_piece)
        else:
            self.pieces_silver.add(new_piece)

        return True

    def remove_piece(self, pos: Tuple[int, int]) -> bool:
        """Remove a piece (e.g., from trap)"""
        piece = self.board[pos[0]][pos[1]]
        if piece is None:
            return False

        self.board[pos[0]][pos[1]] = None
        if piece.color == Color.GOLD:
            self.pieces_gold.discard(piece)
        else:
            self.pieces_silver.discard(piece)

        return True

    def check_and_remove_trapped(self):
        """Check trap squares and remove unsupported pieces"""
        for trap_pos in self.TRAP_SQUARES:
            piece = self.board[trap_pos[0]][trap_pos[1]]
            if piece is None:
                continue

            # Check if piece has friendly adjacent support
            has_support = False
            for adj_pos in self.get_adjacent_positions(trap_pos):
                adj_piece = self.board[adj_pos[0]][adj_pos[1]]
                if adj_piece is not None and adj_piece.color == piece.color:
                    has_support = True
                    break

            # Remove if no support
            if not has_support:
                self.remove_piece(trap_pos)

    def is_frozen(self, pos: Tuple[int, int]) -> bool:
        """
        Check if piece is frozen (adjacent to stronger enemy, no friendly support).
        Frozen pieces cannot move.
        """
        piece = self.board[pos[0]][pos[1]]
        if piece is None:
            return False

        has_stronger_enemy = False
        has_friendly_support = False

        for adj_pos in self.get_adjacent_positions(pos):
            adj_piece = self.board[adj_pos[0]][adj_pos[1]]
            if adj_piece is None:
                continue

            if adj_piece.color == piece.color:
                has_friendly_support = True
            elif adj_piece.is_stronger_than(piece):
                has_stronger_enemy = True

        return has_stronger_enemy and not has_friendly_support

    def get_adjacent_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent positions (4 directions: N, S, E, W)"""
        row, col = pos
        adjacent = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self._is_valid_position(new_pos):
                adjacent.append(new_pos)

        return adjacent

    def get_pieces(self, color: Color) -> Set[Piece]:
        """Get all pieces of a color"""
        return self.pieces_gold if color == Color.GOLD else self.pieces_silver

    def count_rabbits(self, color: Color) -> int:
        """Count rabbits of a color"""
        pieces = self.get_pieces(color)
        return sum(1 for p in pieces if p.type == PieceType.RABBIT)

    def has_rabbit_on_goal(self, color: Color) -> bool:
        """Check if color has rabbit on goal row"""
        goal_row = 7 if color == Color.GOLD else 0
        for col in range(8):
            piece = self.board[goal_row][col]
            if (piece is not None and piece.color == color and
                piece.type == PieceType.RABBIT):
                return True
        return False

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
                if (row, col) in self.TRAP_SQUARES and piece is None:
                    row_str += "X "
                elif piece is None:
                    row_str += ". "
                else:
                    row_str += f"{piece} "
            lines.append(row_str + "|")
        lines.append("  ---------------")
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()
