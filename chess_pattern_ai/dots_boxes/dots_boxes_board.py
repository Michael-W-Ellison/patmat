#!/usr/bin/env python3
"""
Dots and Boxes Board Representation - Observation-Based

Philosophy: Learn strategy from observation
- Board is 5x5 dots (4x4 grid of boxes)
- Players draw edges between dots
- When 4th edge of a box is drawn, that player claims the box
- Double-move rule: Get another turn if you complete a box
- Hashable Edge class for pattern tracking
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True)
class Edge:
    """
    An edge in Dots and Boxes (immutable for hashing)

    Connects two dots. Stored in canonical form: dot1 < dot2
    """
    dot1: Tuple[int, int]  # (row, col)
    dot2: Tuple[int, int]  # (row, col)

    def __init__(self, dot1: Tuple[int, int], dot2: Tuple[int, int]):
        # Ensure canonical form: smaller coordinate first
        if dot1 <= dot2:
            object.__setattr__(self, 'dot1', dot1)
            object.__setattr__(self, 'dot2', dot2)
        else:
            object.__setattr__(self, 'dot1', dot2)
            object.__setattr__(self, 'dot2', dot1)

    def __hash__(self):
        """Make edge hashable for use in sets and dicts"""
        return hash((self.dot1, self.dot2))

    def __eq__(self, other):
        """Equality based on dots"""
        if not isinstance(other, Edge):
            return False
        return self.dot1 == other.dot1 and self.dot2 == other.dot2

    def __repr__(self):
        return f"Edge({self.dot1}-{self.dot2})"


class DotsBoxesBoard:
    """
    Dots and Boxes board - 5x5 dots (4x4 grid of boxes)

    Coordinates: dots at (row, col) where row,col in 0..4
    Boxes at grid positions (row, col) where row,col in 0..3

    Each box needs 4 edges to be completed.
    When a player draws the 4th edge of a box, they claim it.
    """

    DOTS = 5  # 5x5 dot grid
    BOXES = 4  # 4x4 box grid

    def __init__(self):
        """Initialize Dots and Boxes board"""
        self.board_size = self.DOTS
        self.grid_size = self.BOXES

        # Track drawn edges
        self.edges: Set[Edge] = set()

        # Track box ownership: (box_row, box_col) -> Color or None
        self.boxes = {}
        for r in range(self.BOXES):
            for c in range(self.BOXES):
                self.boxes[(r, c)] = None

        # Track whose turn it is
        self.turn = Color.RED

        # Move history: list of (edge, color) tuples
        self.move_history: List[Tuple[Edge, Color]] = []

    def is_valid_dot(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a valid dot"""
        row, col = pos
        return 0 <= row < self.DOTS and 0 <= col < self.DOTS

    def can_draw_edge(self, edge: Edge) -> bool:
        """Check if an edge can be drawn (not already drawn)"""
        return edge not in self.edges

    def get_legal_edges(self) -> List[Edge]:
        """Get all undrawn edges (legal moves)"""
        legal = []

        # Horizontal edges: between (r,c) and (r,c+1)
        for r in range(self.DOTS):
            for c in range(self.DOTS - 1):
                edge = Edge((r, c), (r, c + 1))
                if self.can_draw_edge(edge):
                    legal.append(edge)

        # Vertical edges: between (r,c) and (r+1,c)
        for r in range(self.DOTS - 1):
            for c in range(self.DOTS):
                edge = Edge((r, c), (r + 1, c))
                if self.can_draw_edge(edge):
                    legal.append(edge)

        return legal

    def get_box_edges(self, box_pos: Tuple[int, int]) -> List[Edge]:
        """
        Get the 4 edges that form a box

        Args:
            box_pos: (row, col) position of box in 4x4 grid

        Returns: List of 4 edges that form this box
        """
        r, c = box_pos

        # Top-left corner is at (r, c)
        top_left = (r, c)
        top_right = (r, c + 1)
        bottom_left = (r + 1, c)
        bottom_right = (r + 1, c + 1)

        # 4 edges of the box
        edges = [
            Edge(top_left, top_right),      # Top
            Edge(bottom_left, bottom_right),  # Bottom
            Edge(top_left, bottom_left),     # Left
            Edge(top_right, bottom_right),   # Right
        ]

        return edges

    def draw_edge(self, edge: Edge, color: Color) -> Optional[Tuple[int, int]]:
        """
        Draw an edge

        Args:
            edge: Edge to draw
            color: Color of player drawing

        Returns: Box position (row, col) if a box was completed, None otherwise
        """
        if not self.can_draw_edge(edge):
            return None

        # Add edge
        self.edges.add(edge)
        self.move_history.append((edge, color))

        # Check which boxes are affected by this edge
        # An edge affects at most 2 boxes
        completed_box = None

        for box_pos in self._get_affected_boxes(edge):
            # Check if this box is now complete
            if self._is_box_complete(box_pos) and self.boxes[box_pos] is None:
                # Mark box as owned
                self.boxes[box_pos] = color
                completed_box = box_pos

        return completed_box

    def _get_affected_boxes(self, edge: Edge) -> List[Tuple[int, int]]:
        """Get boxes that are adjacent to an edge"""
        affected = []

        d1, d2 = edge.dot1, edge.dot2

        # Check if edge is horizontal or vertical
        if d1[0] == d2[0]:  # Horizontal edge
            row = d1[0]
            col = min(d1[1], d2[1])

            # Two boxes: above and below (if they exist)
            if row > 0:  # Box above
                affected.append((row - 1, col))
            if row < self.BOXES:  # Box below
                affected.append((row, col))

        else:  # Vertical edge
            row = min(d1[0], d2[0])
            col = d1[1]

            # Two boxes: left and right (if they exist)
            if col > 0:  # Box to the left
                affected.append((row, col - 1))
            if col < self.BOXES:  # Box to the right
                affected.append((row, col))

        return [b for b in affected if 0 <= b[0] < self.BOXES and 0 <= b[1] < self.BOXES]

    def _is_box_complete(self, box_pos: Tuple[int, int]) -> bool:
        """Check if a box has all 4 edges drawn"""
        box_edges = self.get_box_edges(box_pos)
        return all(edge in self.edges for edge in box_edges)

    def count_drawn_edges(self, box_pos: Tuple[int, int]) -> int:
        """Count how many edges of a box have been drawn"""
        box_edges = self.get_box_edges(box_pos)
        return sum(1 for edge in box_edges if edge in self.edges)

    def is_game_over(self) -> bool:
        """Check if all boxes are claimed"""
        return all(self.boxes[pos] is not None for pos in self.boxes)

    def get_winner(self) -> Optional[Color]:
        """
        Get winner if game is over

        Returns: Color of winner, or None if draw
        """
        if not self.is_game_over():
            return None

        red_boxes = sum(1 for c in self.boxes.values() if c == Color.RED)
        blue_boxes = sum(1 for c in self.boxes.values() if c == Color.BLUE)

        if red_boxes > blue_boxes:
            return Color.RED
        elif blue_boxes > red_boxes:
            return Color.BLUE
        else:
            return None  # Draw

    def get_box_counts(self) -> Tuple[int, int]:
        """Get (red_boxes, blue_boxes) count"""
        red = sum(1 for c in self.boxes.values() if c == Color.RED)
        blue = sum(1 for c in self.boxes.values() if c == Color.BLUE)
        return (red, blue)

    def get_potential_boxes(self, color: Color) -> int:
        """Count boxes with 3 edges drawn (potential for next move)"""
        count = 0
        for box_pos in self.boxes:
            if self.boxes[box_pos] is None:  # Unclaimed
                if self.count_drawn_edges(box_pos) == 3:
                    count += 1
        return count

    def copy(self) -> 'DotsBoxesBoard':
        """Create a copy of the board"""
        new_board = DotsBoxesBoard()
        new_board.edges = set(self.edges)
        new_board.boxes = dict(self.boxes)
        new_board.turn = self.turn
        new_board.move_history = list(self.move_history)
        return new_board

    def to_fen(self) -> str:
        """Convert to FEN-like notation for storage"""
        # Encode edges as a bitmask
        edge_list = []

        # Horizontal edges
        for r in range(self.DOTS):
            for c in range(self.DOTS - 1):
                edge = Edge((r, c), (r, c + 1))
                edge_list.append('1' if edge in self.edges else '0')

        # Vertical edges
        for r in range(self.DOTS - 1):
            for c in range(self.DOTS):
                edge = Edge((r, c), (r + 1, c))
                edge_list.append('1' if edge in self.edges else '0')

        edges_str = ''.join(edge_list)

        # Encode boxes
        boxes_str = []
        for r in range(self.BOXES):
            for c in range(self.BOXES):
                owner = self.boxes[(r, c)]
                if owner is None:
                    boxes_str.append('.')
                elif owner == Color.RED:
                    boxes_str.append('R')
                else:
                    boxes_str.append('B')

        turn = 'R' if self.turn == Color.RED else 'B'
        return f"{edges_str}/{'/'.join(boxes_str)} {turn}"

    def __str__(self) -> str:
        """Print board"""
        lines = []

        # Print board with dots and edges
        for r in range(self.DOTS):
            # Dots and horizontal edges row
            dot_line = ""
            for c in range(self.DOTS):
                dot_line += "● "
                if c < self.DOTS - 1:
                    edge = Edge((r, c), (r, c + 1))
                    if edge in self.edges:
                        dot_line += "─ "
                    else:
                        dot_line += "  "
            lines.append(dot_line)

            # Vertical edges and boxes row
            if r < self.BOXES:
                edge_line = ""
                for c in range(self.DOTS):
                    edge = Edge((r, c), (r + 1, c))
                    if edge in self.edges:
                        edge_line += "│ "
                    else:
                        edge_line += "  "

                    if c < self.BOXES:
                        owner = self.boxes[(r, c)]
                        if owner == Color.RED:
                            edge_line += "R "
                        elif owner == Color.BLUE:
                            edge_line += "B "
                        else:
                            edge_line += ". "

                lines.append(edge_line)

        # Score line
        red, blue = self.get_box_counts()
        lines.append(f"\nScore: Red={red}, Blue={blue}")
        lines.append(f"Turn: {self.turn.value}")

        return '\n'.join(lines)


def test_dots_boxes_board():
    """Test the Dots and Boxes board"""
    print("Testing Dots and Boxes Board")
    print("=" * 70)

    # Test 1: Initial board
    board = DotsBoxesBoard()
    print("\nInitial 5x5 Dots (4x4 Boxes):")
    print(board)

    print(f"\nTotal dots: {board.DOTS}x{board.DOTS}")
    print(f"Total boxes: {board.BOXES}x{board.BOXES}")
    print(f"Total possible edges: {2 * board.DOTS * (board.DOTS - 1)}")

    # Test 2: Draw some edges
    print("\n" + "=" * 70)
    print("Test: Drawing edges")

    legal = board.get_legal_edges()
    print(f"Legal edges available: {len(legal)}")

    # Draw a few edges
    test_edges = [legal[0], legal[1], legal[2]]
    for edge in test_edges:
        completed = board.draw_edge(edge, Color.RED)
        print(f"Drew edge {edge}, completed box: {completed}")

    print(board)

    # Test 3: Complete a box
    print("\n" + "=" * 70)
    print("Test: Complete a 1x1 box")

    board2 = DotsBoxesBoard()

    # Get all edges for box (0, 0)
    box_edges = board2.get_box_edges((0, 0))
    print(f"\nEdges of box (0,0): {box_edges}")

    # Draw all 4 edges
    for i, edge in enumerate(box_edges):
        print(f"Drawing edge {i+1}/4: {edge}")
        completed = board2.draw_edge(edge, Color.RED)
        print(f"  Completed box: {completed}")

    print(board2)
    red, blue = board2.get_box_counts()
    print(f"Boxes: Red={red}, Blue={blue}")

    # Test 4: Edge hashability
    print("\n" + "=" * 70)
    print("Test: Edge hashability")
    e1 = Edge((0, 0), (0, 1))
    e2 = Edge((0, 1), (0, 0))  # Reversed, should be same
    e3 = Edge((0, 0), (1, 0))

    edge_set = {e1, e2, e3}
    print(f"Created 3 edge objects (2 identical, 1 different)")
    print(f"Set contains {len(edge_set)} unique edges (should be 2)")
    print(f"Edges are hashable: {len(edge_set) == 2}")

    print("\n" + "=" * 70)
    print("✓ Dots and Boxes board implementation working!")


if __name__ == '__main__':
    test_dots_boxes_board()
