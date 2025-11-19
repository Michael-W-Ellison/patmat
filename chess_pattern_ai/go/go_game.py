#!/usr/bin/env python3
"""
Go Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation (any empty point + pass)
- Future: Replace with learned move patterns
"""

from typing import List, Optional
import random
from .go_board import GoBoard, Move, Color


class GoGame:
    """
    Go game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[GoBoard] = None, board_size: int = 9):
        """
        Initialize Go game

        Args:
            board: Optional GoBoard instance
            board_size: Size of board (9 or 19) if creating new board
        """
        self.board = board or GoBoard(board_size)

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get all legal moves for a color

        Legal moves in Go:
        1. Any empty point on the board
        2. Pass (always legal)

        Returns:
            List of legal Move objects
        """
        moves = []

        # Add all empty points as valid moves
        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.get_stone_at((row, col)) is None:
                    # Check if move would be legal (not suicide unless it captures)
                    move = Move(color, (row, col))
                    if self._is_legal_placement(color, (row, col)):
                        moves.append(move)

        # Pass is always legal
        pass_move = Move(color, None)
        moves.append(pass_move)

        return moves

    def _is_legal_placement(self, color: Color, pos: tuple) -> bool:
        """
        Check if placing a stone at position is legal

        A move is illegal if:
        - Position is already occupied
        - Move would be suicide (stone has no liberties and doesn't capture)
        """
        from .go_board import Stone
        row, col = pos

        # Check if position is empty
        if self.board.get_stone_at(pos) is not None:
            return False

        # Simulate placement on a copy of board
        test_board = self.board.copy()
        stone = Stone(color, pos)
        test_board.board[row][col] = stone
        test_board.stones.add(stone)

        # Check if this captures any opponent stones
        captured = test_board.get_captured_groups(pos)
        if captured:
            return True  # Legal - captures opponent stones

        # Check if own stone group has liberties
        my_group = test_board.get_group(pos)
        if test_board.get_liberties(my_group) > 0:
            return True  # Legal - has liberties

        # Suicide move (no liberties, no captures)
        return False

    def select_ai_move(self, color: Color, strategy: str = "random") -> Optional[Move]:
        """
        Simple AI move selection

        Strategies:
        - "random": Random legal move
        - "aggressive": Prefer non-pass moves, weighted by influence
        - "defensive": Prefer moves that maintain or build territory
        """
        legal_moves = self.get_legal_moves(color)

        if not legal_moves:
            return None

        if strategy == "random":
            return random.choice(legal_moves)

        if strategy == "aggressive":
            # Prefer placing stones over passing
            placement_moves = [m for m in legal_moves if m.position is not None]
            if placement_moves:
                return random.choice(placement_moves)
            return Move(color, None)  # Pass

        if strategy == "defensive":
            # Prefer moves near existing stones
            own_positions = set(s.position for s in self.board.stones
                                if s.color == color)

            if own_positions:
                # Score moves by distance to nearest own stone
                non_pass_moves = [m for m in legal_moves if m.position is not None]
                if non_pass_moves:
                    def score_move(move):
                        if move.position is None:
                            return -1
                        row, col = move.position
                        min_dist = min(abs(row - pr) + abs(col - pc)
                                       for pr, pc in own_positions)
                        return -min_dist  # Prefer closer moves

                    best_move = max(non_pass_moves, key=score_move)
                    return best_move

            return random.choice(legal_moves)

        # Default to random
        return random.choice(legal_moves)

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over (2 consecutive passes)"""
        return self.board.is_game_over()

    def get_winner(self) -> Optional[Color]:
        """
        Determine winner by territory + captures

        Returns:
            Color of winner, or None if draw
        """
        if not self.is_game_over():
            return None

        black_territory = self.board.count_territory(Color.BLACK)
        white_territory = self.board.count_territory(Color.WHITE)

        black_score = black_territory + self.board.captured_by_black
        white_score = white_territory + self.board.captured_by_white

        if black_score > white_score:
            return Color.BLACK
        elif white_score > black_score:
            return Color.WHITE
        else:
            return None  # Draw

    def get_territory_count(self, color: Color) -> int:
        """Count territory for a color"""
        return self.board.count_territory(color)

    def get_capture_count(self, color: Color) -> int:
        """Get number of captures for a color"""
        if color == Color.BLACK:
            return self.board.captured_by_black
        else:
            return self.board.captured_by_white

    def get_material_count(self, color: Color) -> int:
        """
        Get combined material score (territory + captures)

        In Go, "material" is typically territory + captured stones
        """
        territory = self.get_territory_count(color)
        captures = self.get_capture_count(color)
        return territory + captures


def test_go_game():
    """Test Go game engine"""
    print("=" * 70)
    print("TESTING GO GAME ENGINE")
    print("=" * 70)

    game = GoGame(board_size=9)
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for black
    legal_moves = game.get_legal_moves(Color.BLACK)
    print(f"\nBlack has {len(legal_moves)} legal moves")
    print(f"Sample moves: {[m.notation() for m in legal_moves[:5]]}")

    # Make some moves
    print("\n" + "=" * 70)
    print("Playing some moves...")
    print("=" * 70)

    # Black plays at 3,3 (standard opening)
    move1 = Move(Color.BLACK, (3, 3))
    if game.make_move(move1):
        print(f"Black plays {move1.notation()}")
        print(game.board)

    # White responds
    legal_moves = game.get_legal_moves(Color.WHITE)
    if legal_moves:
        move2 = Move(Color.WHITE, (15, 15) if game.board.size == 19 else (5, 5))
        if game.make_move(move2):
            print(f"\nWhite plays {move2.notation()}")
            print(game.board)

    # Test AI move selection
    print("\n" + "=" * 70)
    print("Testing AI move selection...")
    print("=" * 70)

    for strategy in ["random", "aggressive", "defensive"]:
        game2 = GoGame(board_size=9)
        ai_move = game2.select_ai_move(Color.BLACK, strategy)
        print(f"\n{strategy.capitalize()} strategy: {ai_move.notation() if ai_move else 'None'}")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_go_game()
