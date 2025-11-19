#!/usr/bin/env python3
"""
Gomoku Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation (any empty intersection)
- Future: Replace with learned move patterns
"""

from typing import List, Optional, Set, Tuple
from .gomoku_board import GomokuBoard, Move, Stone, Color


class GomokuGame:
    """
    Gomoku game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[GomokuBoard] = None, board_size: int = 15):
        """
        Initialize Gomoku game

        Args:
            board: Existing board or None to create new
            board_size: Board size (15 or 19) if creating new board
        """
        self.board = board or GomokuBoard(board_size)

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get legal moves for a color

        Legal moves are placements on any empty intersection.
        In Gomoku, any empty position is a valid move.
        """
        moves = []

        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.is_empty((row, col)):
                    stone = Stone(color, (row, col))
                    move = Move(position=(row, col), color=color, stone=stone)
                    moves.append(move)

        return moves

    def get_reasonable_moves(self, color: Color, depth: int = 2) -> List[Move]:
        """
        Get reasonable moves (not all empty intersections)

        For efficiency, return moves near existing stones plus corner/center openings.
        This is a heuristic to reduce search space.

        Args:
            color: Color to get moves for
            depth: Distance from existing stones to consider

        Returns: List of reasonable moves
        """
        moves = []
        considered = set()

        if not self.board.stones:
            # Opening moves: center area
            center = self.board.size // 2
            for row in range(max(0, center - 2), min(self.board.size, center + 3)):
                for col in range(max(0, center - 2), min(self.board.size, center + 3)):
                    if self.board.is_empty((row, col)):
                        stone = Stone(color, (row, col))
                        move = Move(position=(row, col), color=color, stone=stone)
                        moves.append(move)
                        considered.add((row, col))
        else:
            # Add moves near existing stones
            for stone in self.board.stones:
                row, col = stone.position
                for dr in range(-depth, depth + 1):
                    for dc in range(-depth, depth + 1):
                        new_row, new_col = row + dr, col + dc
                        if (self.board.is_valid_position((new_row, new_col)) and
                            self.board.is_empty((new_row, new_col)) and
                            (new_row, new_col) not in considered):
                            stone = Stone(color, (new_row, new_col))
                            move = Move(position=(new_row, new_col), color=color, stone=stone)
                            moves.append(move)
                            considered.add((new_row, new_col))

        return moves

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over"""
        if self.board.is_game_over():
            return True

        # Check if current player has legal moves (shouldn't happen in Gomoku unless board full)
        legal_moves = self.get_legal_moves(self.board.turn)
        return len(legal_moves) == 0

    def get_winner(self) -> Optional[Color]:
        """Get winner"""
        if not self.board.is_game_over():
            return None

        # Check for winner
        winner = self.board.get_winner()
        if winner:
            return winner

        # If no winner and board is full, it's a draw
        if self.board._is_board_full():
            return None  # Draw

        return None

    def get_stone_count(self, color: Color) -> int:
        """Get number of stones for a color"""
        return len(self.board.get_stones(color))

    def detect_threats(self, color: Color) -> dict:
        """
        Detect threats for a color

        Returns dict with threat counts:
        - open_4: 4 in a row with empty space on both ends
        - open_3: 3 in a row with empty space on both ends
        - open_2: 2 in a row with empty space on both ends
        - blocked_4: 4 in a row with stone blocked on one end
        - blocked_3: 3 in a row with stone blocked on one end
        - blocked_2: 2 in a row with stone blocked on one end
        """
        threats = {
            'open_4': 0,
            'open_3': 0,
            'open_2': 0,
            'blocked_4': 0,
            'blocked_3': 0,
            'blocked_2': 0,
        }

        stones = self.board.get_stones(color)
        stone_positions = set(s.position for s in stones)
        opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
        opp_positions = set(s.position for s in self.board.get_stones(opp_color))

        # Check each stone
        for stone in stones:
            row, col = stone.position

            # Directions: horizontal, vertical, diagonal-1, diagonal-2
            directions = [
                (0, 1),   # Horizontal (right)
                (1, 0),   # Vertical (down)
                (1, 1),   # Diagonal (down-right)
                (1, -1)   # Diagonal (down-left)
            ]

            for d_row, d_col in directions:
                # Count consecutive stones in both directions
                forward_count = 1
                forward_blocked = False
                backward_count = 1
                backward_blocked = False

                # Forward direction
                for i in range(1, 5):
                    pos = (row + i * d_row, col + i * d_col)
                    if not self.board.is_valid_position(pos):
                        forward_blocked = True
                        break
                    if pos in stone_positions:
                        forward_count += 1
                    elif pos in opp_positions:
                        forward_blocked = True
                        break
                    else:
                        break

                # Backward direction
                for i in range(1, 5):
                    pos = (row - i * d_row, col - i * d_col)
                    if not self.board.is_valid_position(pos):
                        backward_blocked = True
                        break
                    if pos in stone_positions:
                        backward_count += 1
                    elif pos in opp_positions:
                        backward_blocked = True
                        break
                    else:
                        break

                total = forward_count + backward_count - 1

                # Categorize threat
                if total == 4:
                    if forward_blocked or backward_blocked:
                        threats['blocked_4'] += 1
                    else:
                        threats['open_4'] += 1
                elif total == 3:
                    if forward_blocked or backward_blocked:
                        threats['blocked_3'] += 1
                    else:
                        threats['open_3'] += 1
                elif total == 2:
                    if forward_blocked or backward_blocked:
                        threats['blocked_2'] += 1
                    else:
                        threats['open_2'] += 1

        # Divide by number of directions to avoid overcounting
        for key in threats:
            threats[key] = threats[key] // 4

        return threats


def test_gomoku_game():
    """Test Gomoku game engine"""
    print("=" * 70)
    print("TESTING GOMOKU GAME ENGINE")
    print("=" * 70)

    game = GomokuGame(board_size=15)
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for black
    legal_moves = game.get_legal_moves(Color.BLACK)
    print(f"\nBlack has {len(legal_moves)} legal moves")

    # Get reasonable moves
    reasonable_moves = game.get_reasonable_moves(Color.BLACK)
    print(f"Black has {len(reasonable_moves)} reasonable moves (opening)")

    # Make a series of moves
    print("\n" + "=" * 70)
    print("Making moves in center...")

    move_sequence = [
        (Color.BLACK, (7, 7)),   # Black center
        (Color.WHITE, (7, 8)),   # White right of black
        (Color.BLACK, (8, 7)),   # Black below
        (Color.WHITE, (6, 7)),   # White above
    ]

    for color, pos in move_sequence:
        legal_moves = game.get_legal_moves(color)
        move = next((m for m in legal_moves if m.position == pos), None)
        if move:
            print(f"\n{color.value.upper()} plays at {move.notation()}")
            game.make_move(move)
            print(game.board)
        else:
            print(f"Move {pos} not available")
            break

    # Test threat detection
    print("\n" + "=" * 70)
    print("Testing threat detection:")
    game2 = GomokuGame(board_size=15)

    # Create black line: 3 in a row in middle
    for col in range(7, 10):
        stone = Stone(Color.BLACK, (7, col))
        move = Move(position=(7, col), color=Color.BLACK, stone=stone)
        game2.make_move(move)

    print("\nPosition with 3 black stones in a row (d8-f8):")
    print(game2.board)

    threats = game2.detect_threats(Color.BLACK)
    print("\nBlack threats:")
    for threat_type, count in threats.items():
        if count > 0:
            print(f"  {threat_type}: {count}")

    # Test 5-in-a-row detection
    print("\n" + "=" * 70)
    print("Testing 5-in-a-row detection:")
    game3 = GomokuGame(board_size=15)

    # Create black line: 5 in a row
    for col in range(7, 12):
        stone = Stone(Color.BLACK, (7, col))
        move = Move(position=(7, col), color=Color.BLACK, stone=stone)
        game3.make_move(move)

    print("\nPosition with 5 black stones in a row:")
    print(game3.board)
    print(f"Black has won: {game3.board.check_win(Color.BLACK)}")
    print(f"Game is over: {game3.is_game_over()}")
    print(f"Winner: {game3.get_winner()}")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_gomoku_game()
