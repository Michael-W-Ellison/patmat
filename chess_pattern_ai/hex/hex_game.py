#!/usr/bin/env python3
"""
Hex Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Move generation for any empty cell
- Smart filtering: Prefer moves near existing stones (bridge patterns)
- Future: Replace with learned move patterns
"""

from typing import List, Optional, Tuple
import random
from .hex_board import HexBoard, Move, Color


class HexGame:
    """
    Hex game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[HexBoard] = None, board_size: int = 11):
        """
        Initialize Hex game

        Args:
            board: Optional HexBoard instance
            board_size: Size of board (standard 11) if creating new board
        """
        self.board = board or HexBoard(board_size)

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get all legal moves for a color

        Legal moves in Hex:
        1. Any empty cell on the board

        Returns:
            List of legal Move objects
        """
        moves = []

        # Add all empty points as valid moves
        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.get_stone_at((row, col)) is None:
                    moves.append(Move(color, (row, col)))

        return moves

    def select_ai_move(self, color: Color, strategy: str = "random") -> Optional[Move]:
        """
        Simple AI move selection

        Strategies:
        - "random": Random legal move
        - "aggressive": Prefer moves that extend existing stones (bridges)
        - "center": Prefer moves closer to center
        - "defensive": Prefer moves blocking opponent's path
        """
        legal_moves = self.get_legal_moves(color)

        if not legal_moves:
            return None

        if strategy == "random":
            return random.choice(legal_moves)

        if strategy == "aggressive":
            # Prefer moves near existing stones (bridge patterns)
            own_positions = {s.position for s in self.board.stones if s.color == color}

            if own_positions:
                # Score moves by proximity to own stones
                def score_move(move):
                    row, col = move.position
                    min_dist = min(
                        abs(row - pr) + abs(col - pc)
                        for pr, pc in own_positions
                    )
                    return -min_dist  # Prefer closer moves

                best_moves = sorted(legal_moves, key=score_move, reverse=True)
                # Return one of the best few moves (add randomness)
                return best_moves[0] if best_moves else random.choice(legal_moves)

            return random.choice(legal_moves)

        if strategy == "center":
            # Prefer moves closer to center
            center = self.board.size // 2
            def score_move(move):
                row, col = move.position
                dist_to_center = abs(row - center) + abs(col - center)
                return -dist_to_center  # Prefer closer to center

            best_moves = sorted(legal_moves, key=score_move, reverse=True)
            return best_moves[0] if best_moves else random.choice(legal_moves)

        if strategy == "defensive":
            # Prefer moves that block opponent's connection
            opponent_color = Color.BLUE if color == Color.RED else Color.RED

            # Get opponent's connected group
            opp_connected = self.board.is_connected_to_start(opponent_color)

            if opp_connected:
                # Prefer moves that are neighbors of opponent stones
                opp_neighbors = set()
                for stone in opp_connected:
                    for neighbor in self.board.get_neighbors(stone.position):
                        if self.board.get_stone_at(neighbor) is None:
                            opp_neighbors.add(neighbor)

                # Filter legal moves to those near opponent
                defensive_moves = [m for m in legal_moves if m.position in opp_neighbors]
                if defensive_moves:
                    return random.choice(defensive_moves)

            return random.choice(legal_moves)

        # Default to random
        return random.choice(legal_moves)

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over (someone has won)"""
        return self.board.has_won(Color.RED) or self.board.has_won(Color.BLUE)

    def get_winner(self) -> Optional[Color]:
        """
        Determine winner

        Returns:
            Color of winner, or None if game not over
        """
        if self.board.has_won(Color.RED):
            return Color.RED
        elif self.board.has_won(Color.BLUE):
            return Color.BLUE
        else:
            return None

    def get_material_count(self, color: Color) -> int:
        """
        Get connection strength for a color

        In Hex, "material" is the number of stones placed (connection strength)
        Higher is better for player pursuing that goal
        """
        return sum(1 for stone in self.board.stones if stone.color == color)

    def get_connection_strength(self, color: Color) -> Tuple[int, int]:
        """
        Get connection strength metrics

        Returns:
            (group_size, distance_to_goal) for the color's main connected group
        """
        return self.board.get_connection_strength(color)


def test_hex_game():
    """Test Hex game engine"""
    print("=" * 70)
    print("TESTING HEX GAME ENGINE")
    print("=" * 70)

    game = HexGame(board_size=11)
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for red
    legal_moves = game.get_legal_moves(Color.RED)
    print(f"\nRed has {len(legal_moves)} legal moves")
    print(f"Sample moves: {[m.notation() for m in legal_moves[:5]]}")

    # Make some moves
    print("\n" + "=" * 70)
    print("Playing some moves...")
    print("=" * 70)

    # Red plays in center
    move1 = Move(Color.RED, (5, 5))
    if game.make_move(move1):
        print(f"Red plays {move1.notation()}")
        print(game.board)

    # Blue responds
    move2 = Move(Color.BLUE, (5, 3))
    if game.make_move(move2):
        print(f"Blue plays {move2.notation()}")
        print(game.board)

    # Red extends
    move3 = Move(Color.RED, (5, 6))
    if game.make_move(move3):
        print(f"Red plays {move3.notation()}")
        print(game.board)

    # Test AI move selection
    print("\n" + "=" * 70)
    print("Testing AI move selection...")
    print("=" * 70)

    for strategy in ["random", "aggressive", "center", "defensive"]:
        game2 = HexGame(board_size=11)
        ai_move = game2.select_ai_move(Color.RED, strategy)
        print(f"\n{strategy.capitalize()} strategy: {ai_move.notation() if ai_move else 'None'}")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_hex_game()
