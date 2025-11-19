#!/usr/bin/env python3
"""
Dots and Boxes Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation (any undrawn edge)
- Future: Replace with learned move patterns
"""

from typing import List, Optional, Tuple
from .dots_boxes_board import DotsBoxesBoard, Edge, Color


class DotsBoxesGame:
    """
    Dots and Boxes game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[DotsBoxesBoard] = None):
        """
        Initialize Dots and Boxes game

        Args:
            board: Existing board or None to create new
        """
        self.board = board or DotsBoxesBoard()
        self.current_player = Color.RED
        self.last_move_completed_box = False

    def get_legal_moves(self, color: Optional[Color] = None) -> List[Edge]:
        """
        Get legal moves (all undrawn edges)

        Args:
            color: Color to get moves for (ignored, included for consistency)

        Returns: List of legal edges that can be drawn
        """
        return self.board.get_legal_edges()

    def make_move(self, edge: Edge) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Execute a move (draw an edge)

        Args:
            edge: Edge to draw

        Returns: (success, completed_box)
            success: True if move was made
            completed_box: Box position if a box was completed, None otherwise
        """
        if not self.board.can_draw_edge(edge):
            return (False, None)

        # Draw the edge
        completed_box = self.board.draw_edge(edge, self.current_player)

        if completed_box is not None:
            # Box was completed - player gets another turn
            self.last_move_completed_box = True
            return (True, completed_box)
        else:
            # No box completed - switch turns
            self.last_move_completed_box = False
            self._switch_turn()
            return (True, None)

    def _switch_turn(self):
        """Switch to the other player"""
        self.current_player = (
            Color.BLUE if self.current_player == Color.RED else Color.RED
        )

    def get_another_turn(self) -> bool:
        """Check if current player should get another turn"""
        return self.last_move_completed_box

    def is_game_over(self) -> bool:
        """Check if game is over (all boxes claimed)"""
        return self.board.is_game_over()

    def get_winner(self) -> Optional[Color]:
        """
        Get winner

        Returns: Color of winner, or None if game not over or draw
        """
        if not self.is_game_over():
            return None

        return self.board.get_winner()

    def get_box_counts(self) -> Tuple[int, int]:
        """Get (red_boxes, blue_boxes)"""
        return self.board.get_box_counts()

    def get_game_status(self) -> dict:
        """Get current game status"""
        red, blue = self.board.get_box_counts()
        legal_moves = self.get_legal_moves()

        return {
            'red_boxes': red,
            'blue_boxes': blue,
            'current_player': self.current_player,
            'game_over': self.is_game_over(),
            'legal_moves_count': len(legal_moves),
            'last_move_completed_box': self.last_move_completed_box,
            'winner': self.get_winner(),
        }


def test_dots_boxes_game():
    """Test Dots and Boxes game engine"""
    print("=" * 70)
    print("TESTING DOTS AND BOXES GAME ENGINE")
    print("=" * 70)

    game = DotsBoxesGame()
    print("\nInitial board:")
    print(game.board)

    # Get legal moves
    legal_moves = game.get_legal_moves()
    print(f"\nLegal moves available: {len(legal_moves)}")
    print(f"Total edges possible: {2 * game.board.DOTS * (game.board.DOTS - 1)}")

    # Test move sequence
    print("\n" + "=" * 70)
    print("Making moves...")

    # Draw 3 edges of box (0, 0) to make it nearly complete
    box_edges = game.board.get_box_edges((0, 0))
    print(f"\nEdges for box (0,0): {box_edges}")

    for i in range(3):
        edge = box_edges[i]
        success, completed = game.make_move(edge)
        red, blue = game.board.get_box_counts()
        print(f"\nMove {i+1}: {edge}")
        print(f"  Success: {success}")
        print(f"  Completed box: {completed}")
        print(f"  Score: Red={red}, Blue={blue}")
        print(f"  Current player: {game.current_player.value}")
        print(f"  Get another turn: {game.get_another_turn()}")

    # Complete the box with 4th edge
    print("\n" + "=" * 70)
    print("Completing box with 4th edge...")
    edge = box_edges[3]
    success, completed = game.make_move(edge)
    red, blue = game.board.get_box_counts()

    print(f"\nMove 4: {edge}")
    print(f"  Success: {success}")
    print(f"  Completed box: {completed}")
    print(f"  Score: Red={red}, Blue={blue}")
    print(f"  Current player: {game.current_player.value}")
    print(f"  Get another turn: {game.get_another_turn()}")

    print(game.board)

    # Test game status
    print("\n" + "=" * 70)
    print("Game Status:")
    status = game.get_game_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_dots_boxes_game()
