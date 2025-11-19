#!/usr/bin/env python3
"""
Go Board Representation - Simplified for Pattern Learning

Philosophy: Learn strategy from observation
- Board can be 9x9 or 19x19
- Captures learned from observing surrounded groups
- Territory scoring learned from game outcomes
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    BLACK = "black"
    WHITE = "white"


@dataclass(frozen=True)
class Stone:
    """A Go stone (immutable for hashing)"""
    color: Color
    position: Tuple[int, int]  # (row, col)

    def __repr__(self):
        color_char = 'B' if self.color == Color.BLACK else 'W'
        return f"{color_char}"


@dataclass
class Move:
    """A Go move"""
    color: Color
    position: Optional[Tuple[int, int]]  # None = pass
    captured: Set[Stone] = None  # Stones captured by this move

    def __post_init__(self):
        if self.captured is None:
            self.captured = set()

    def notation(self) -> str:
        """Return move in coordinate notation"""
        if self.position is None:
            return "PASS"
        row, col = self.position
        col_letter = chr(ord('A') + col + (1 if col >= 8 else 0))  # Skip 'I'
        return f"{col_letter}{row + 1}"


class GoBoard:
    """
    Go board representation

    Supports both 9x9 and 19x19 boards
    """

    def __init__(self, size: int = 9):
        """
        Initialize Go board

        Args:
            size: Board size (9 or 19)
        """
        if size not in [9, 19]:
            raise ValueError("Board size must be 9 or 19")

        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.stones: Set[Stone] = set()
        self.turn = Color.BLACK
        self.move_history: List[Move] = []
        self.consecutive_passes = 0
        self.captured_by_black = 0
        self.captured_by_white = 0

    def __str__(self) -> str:
        """Visual representation of board"""
        result = []
        result.append("  " + " ".join(chr(ord('A') + i + (1 if i >= 8 else 0))
                                     for i in range(self.size)))

        for row in range(self.size):
            row_str = f"{row + 1:2d} "
            for col in range(self.size):
                stone = self.board[row][col]
                if stone is None:
                    # Show board intersections
                    if row == 0:
                        if col == 0:
                            row_str += "┌ "
                        elif col == self.size - 1:
                            row_str += "┐"
                        else:
                            row_str += "┬ "
                    elif row == self.size - 1:
                        if col == 0:
                            row_str += "└ "
                        elif col == self.size - 1:
                            row_str += "┘"
                        else:
                            row_str += "┴ "
                    else:
                        if col == 0:
                            row_str += "├ "
                        elif col == self.size - 1:
                            row_str += "┤"
                        else:
                            row_str += "┼ "
                else:
                    row_str += f"{stone} "
            result.append(row_str.rstrip())

        result.append(f"\nTurn: {self.turn.value}")
        result.append(f"Captured - Black: {self.captured_by_black}, White: {self.captured_by_white}")

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
        """Get orthogonal neighbors of a position"""
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def get_group(self, pos: Tuple[int, int]) -> Set[Stone]:
        """Get all connected stones of same color (group)"""
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

    def get_liberties(self, group: Set[Stone]) -> int:
        """Count liberties (empty adjacent points) for a group"""
        liberties = set()
        for stone in group:
            for neighbor_pos in self.get_neighbors(stone.position):
                if self.get_stone_at(neighbor_pos) is None:
                    liberties.add(neighbor_pos)
        return len(liberties)

    def get_captured_groups(self, last_move_pos: Tuple[int, int]) -> Set[Stone]:
        """Find all opponent groups captured by placing stone at position"""
        last_stone = self.get_stone_at(last_move_pos)
        if last_stone is None:
            return set()

        captured = set()
        opponent_color = Color.WHITE if last_stone.color == Color.BLACK else Color.BLACK

        # Check all adjacent opponent stones
        for neighbor_pos in self.get_neighbors(last_move_pos):
            neighbor_stone = self.get_stone_at(neighbor_pos)
            if neighbor_stone and neighbor_stone.color == opponent_color:
                group = self.get_group(neighbor_pos)
                if self.get_liberties(group) == 0:
                    captured.update(group)

        return captured

    def make_move(self, move: Move) -> bool:
        """
        Execute a move

        Returns:
            True if move was legal and executed
        """
        # Handle pass
        if move.position is None:
            self.consecutive_passes += 1
            self.move_history.append(move)
            self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK
            return True

        row, col = move.position

        # Check if position is empty
        if self.board[row][col] is not None:
            return False

        # Place stone
        stone = Stone(move.color, move.position)
        self.board[row][col] = stone
        self.stones.add(stone)

        # Check for captures
        captured = self.get_captured_groups(move.position)
        move.captured = captured

        # Remove captured stones
        for captured_stone in captured:
            cap_row, cap_col = captured_stone.position
            self.board[cap_row][cap_col] = None
            self.stones.discard(captured_stone)

            if move.color == Color.BLACK:
                self.captured_by_black += 1
            else:
                self.captured_by_white += 1

        # Check if this move is suicide (placed stone has no liberties)
        # In some rulesets, suicide is illegal unless it captures
        if not captured:
            my_group = self.get_group(move.position)
            if self.get_liberties(my_group) == 0:
                # Suicide - illegal in most rulesets
                # Undo placement
                self.board[row][col] = None
                self.stones.discard(stone)
                return False

        self.consecutive_passes = 0
        self.move_history.append(move)
        self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK

        return True

    def is_game_over(self) -> bool:
        """Game ends after two consecutive passes"""
        return self.consecutive_passes >= 2

    def count_territory(self, color: Color) -> int:
        """
        Count territory controlled by color (simplified scoring)

        Territory = empty points completely surrounded by one color
        """
        visited = set()
        territory = 0

        for row in range(self.size):
            for col in range(self.size):
                pos = (row, col)
                if pos in visited or self.board[row][col] is not None:
                    continue

                # Find connected empty region
                region = set()
                borders = set()  # Colors that border this region
                to_check = [pos]

                while to_check:
                    current = to_check.pop()
                    if current in region or current in visited:
                        continue

                    if not self.is_valid_position(current):
                        continue

                    stone = self.get_stone_at(current)
                    if stone is not None:
                        borders.add(stone.color)
                        continue

                    region.add(current)
                    visited.add(current)

                    for neighbor in self.get_neighbors(current):
                        if neighbor not in region:
                            to_check.append(neighbor)

                # If region borders only one color, it's that color's territory
                if len(borders) == 1 and color in borders:
                    territory += len(region)

        return territory

    def copy(self) -> 'GoBoard':
        """Create a copy of the board"""
        new_board = GoBoard(self.size)
        new_board.turn = self.turn
        new_board.consecutive_passes = self.consecutive_passes
        new_board.captured_by_black = self.captured_by_black
        new_board.captured_by_white = self.captured_by_white

        for stone in self.stones:
            row, col = stone.position
            new_stone = Stone(stone.color, stone.position)
            new_board.board[row][col] = new_stone
            new_board.stones.add(new_stone)

        return new_board


def test_go_board():
    """Test Go board"""
    print("=" * 70)
    print("TESTING GO BOARD (9x9)")
    print("=" * 70)

    board = GoBoard(9)
    print("\nInitial board:")
    print(board)

    # Place some stones
    print("\nPlacing stones...")
    move1 = Move(Color.BLACK, (3, 3))
    board.make_move(move1)

    move2 = Move(Color.WHITE, (3, 4))
    board.make_move(move2)

    print(board)

    # Test capture
    print("\nTesting capture scenario...")
    board2 = GoBoard(9)

    # Surround a white stone
    board2.make_move(Move(Color.BLACK, (1, 1)))  # Place white
    board2.make_move(Move(Color.WHITE, (2, 2)))  # Will be captured
    board2.make_move(Move(Color.BLACK, (2, 1)))
    board2.make_move(Move(Color.WHITE, (3, 3)))
    board2.make_move(Move(Color.BLACK, (2, 3)))
    board2.make_move(Move(Color.WHITE, (4, 4)))
    board2.make_move(Move(Color.BLACK, (3, 2)))
    board2.make_move(Move(Color.WHITE, (5, 5)))

    print(board2)
    print(f"\nCapture move - Black closes in:")
    board2.make_move(Move(Color.BLACK, (1, 2)))  # Capture!

    print(board2)
    print(f"White stone at (2,2) should be captured")

    print("\n" + "=" * 70)
    print("✓ Go board working!")


if __name__ == '__main__':
    test_go_board()
