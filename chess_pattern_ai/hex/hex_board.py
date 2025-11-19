#!/usr/bin/env python3
"""
Hex Board Representation - Connection-Based Learning

Philosophy: Learn strategy from connection strength
- 11x11 hexagonal board (standard)
- Red stones (top-bottom goal)
- Blue stones (left-right goal)
- No captures - only placement matters
- Hashable Stone class for game state tracking
- Connection detection via flood fill
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True)
class Stone:
    """A Hex stone (immutable for hashing)"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        color_char = 'R' if self.color == Color.RED else 'B'
        return f"{color_char}"

    def __hash__(self):
        return hash((self.color, self.position))

    def __eq__(self, other):
        if not isinstance(other, Stone):
            return False
        return self.color == other.color and self.position == other.position


@dataclass
class Move:
    """A Hex move"""
    color: Color
    position: Tuple[int, int]

    def notation(self) -> str:
        """Return move in coordinate notation"""
        row, col = self.position
        col_letter = chr(ord('A') + col)
        return f"{col_letter}{row + 1}"


class HexBoard:
    """
    Hex board representation

    Standard 11x11 board
    - Red goal: top-bottom connection
    - Blue goal: left-right connection
    """

    def __init__(self, size: int = 11):
        """
        Initialize Hex board

        Args:
            size: Board size (standard is 11)
        """
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.stones: Set[Stone] = set()
        self.turn = Color.RED
        self.move_history: List[Move] = []

    def __str__(self) -> str:
        """Visual representation of board"""
        result = []

        # Top margin
        result.append("    " + " ".join(chr(ord('A') + i) for i in range(self.size)))

        for row in range(self.size):
            # Offset for visual hex appearance
            offset = " " * row
            row_str = f"{row + 1:2d} {offset}"

            for col in range(self.size):
                stone = self.board[row][col]
                if stone is None:
                    row_str += ". "
                else:
                    row_str += f"{stone} "

            result.append(row_str.rstrip())

        result.append(f"\nTurn: {self.turn.value}")
        result.append(f"Stones placed: {len(self.stones)}")

        return "\n".join(result)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on board"""
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size

    def get_stone_at(self, pos: Tuple[int, int]) -> Optional[Stone]:
        """Get stone at position"""
        if not self.is_valid_position(pos):
            return None
        row, col = pos
        return self.board[row][col]

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all neighboring hexagons of a position

        In a hex grid, each position has up to 6 neighbors.
        Using axial coordinate system:
        - For even rows: NW, NE, E, SE, SW, W
        - For odd rows: same relative directions
        """
        row, col = pos
        neighbors = []

        # Hexagonal neighbor offsets (varies by row parity)
        if row % 2 == 0:
            # Even row
            offsets = [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)]
        else:
            # Odd row
            offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]

        for dr, dc in offsets:
            neighbor = (row + dr, col + dc)
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def make_move(self, move: Move) -> bool:
        """
        Execute a move

        Returns:
            True if move was legal and executed
        """
        row, col = move.position

        # Check if position is empty
        if self.board[row][col] is not None:
            return False

        # Place stone
        stone = Stone(move.color, move.position)
        self.board[row][col] = stone
        self.stones.add(stone)

        # Update move history
        self.move_history.append(move)

        # Switch turns
        self.turn = Color.BLUE if self.turn == Color.RED else Color.RED

        return True

    def get_connected_group(self, pos: Tuple[int, int]) -> Set[Stone]:
        """Get all connected stones of same color via flood fill"""
        stone = self.get_stone_at(pos)
        if stone is None:
            return set()

        group = set()
        to_check = [pos]
        checked = set()

        while to_check:
            current_pos = to_check.pop()
            if current_pos in checked:
                continue

            checked.add(current_pos)
            current_stone = self.get_stone_at(current_pos)

            if current_stone and current_stone.color == stone.color:
                group.add(current_stone)
                for neighbor_pos in self.get_neighbors(current_pos):
                    if neighbor_pos not in checked:
                        to_check.append(neighbor_pos)

        return group

    def is_connected_to_start(self, color: Color) -> Set[Stone]:
        """
        Get all stones connected to starting edge for a color

        Red: stones touching top edge (row 0)
        Blue: stones touching left edge (col 0)
        """
        connected = set()

        if color == Color.RED:
            # Find all stones in top row or connected to top row
            start_positions = [(0, col) for col in range(self.size)]
        else:  # BLUE
            # Find all stones in left column or connected to left column
            start_positions = [(row, 0) for row in range(self.size)]

        for pos in start_positions:
            group = self.get_connected_group(pos)
            connected.update(group)

        return connected

    def is_connected_to_end(self, color: Color, start_group: Set[Stone]) -> bool:
        """
        Check if a group connected to start reaches the goal

        Red goal: bottom edge (row = size-1)
        Blue goal: right edge (col = size-1)
        """
        if color == Color.RED:
            # Check if any stone in group is in bottom row
            return any(stone.position[0] == self.size - 1 for stone in start_group)
        else:  # BLUE
            # Check if any stone in group is in rightmost column
            return any(stone.position[1] == self.size - 1 for stone in start_group)

    def has_won(self, color: Color) -> bool:
        """Check if a color has won (connected start to goal)"""
        start_group = self.is_connected_to_start(color)
        if not start_group:
            return False
        return self.is_connected_to_end(color, start_group)

    def get_connection_strength(self, color: Color) -> Tuple[int, int]:
        """
        Estimate connection strength for a color

        Returns: (group_size, distance_to_goal)
        where distance_to_goal is estimated via shortest path
        """
        connected = self.is_connected_to_start(color)

        if not connected:
            return (0, float('inf'))

        group_size = len(connected)

        # Estimate distance to goal via BFS from connected group
        if color == Color.RED:
            goal_positions = {(self.size - 1, col) for col in range(self.size)}
        else:  # BLUE
            goal_positions = {(row, self.size - 1) for row in range(self.size)}

        # BFS distance from connected group to goal
        visited = {stone.position for stone in connected}
        to_check = [(pos, 0) for stone in connected for pos in self.get_neighbors(stone.position)]
        min_distance = float('inf')

        while to_check:
            pos, dist = to_check.pop(0)

            if pos in visited:
                continue

            visited.add(pos)

            if pos in goal_positions:
                min_distance = min(min_distance, dist)
                continue

            # Continue BFS
            for neighbor_pos in self.get_neighbors(pos):
                if neighbor_pos not in visited:
                    to_check.append((neighbor_pos, dist + 1))

        return (group_size, min_distance)

    def copy(self) -> 'HexBoard':
        """Create a copy of the board"""
        new_board = HexBoard(self.size)
        new_board.turn = self.turn

        for stone in self.stones:
            row, col = stone.position
            new_stone = Stone(stone.color, stone.position)
            new_board.board[row][col] = new_stone
            new_board.stones.add(new_stone)

        return new_board


def test_hex_board():
    """Test Hex board"""
    print("=" * 70)
    print("TESTING HEX BOARD (11x11)")
    print("=" * 70)

    board = HexBoard(11)
    print("\nInitial board:")
    print(board)

    # Place some stones
    print("\nPlacing stones...")
    move1 = Move(Color.RED, (5, 5))
    board.make_move(move1)

    move2 = Move(Color.BLUE, (3, 3))
    board.make_move(move2)

    print(board)

    # Test winning path
    print("\nBuilding Red winning path (top to bottom)...")
    board2 = HexBoard(11)

    # Red builds a path from top to bottom
    red_path = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5)]
    for i, pos in enumerate(red_path):
        move = Move(Color.RED if i % 2 == 0 else Color.BLUE, pos)
        if board2.make_move(move):
            pass
        else:
            print(f"Failed to place at {pos}")

    print(board2)
    print(f"Red connected to start: {len(board2.is_connected_to_start(Color.RED))} stones")
    print(f"Red has won: {board2.has_won(Color.RED)}")

    print("\n" + "=" * 70)
    print("✓ Hex board working!")


if __name__ == '__main__':
    test_hex_board()
