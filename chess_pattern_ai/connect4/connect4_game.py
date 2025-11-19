#!/usr/bin/env python3
"""
Connect Four Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation (legal columns)
- Future: Replace with learned move patterns
"""

from typing import List, Optional
from .connect4_board import Connect4Board, Move, Piece, Color


class Connect4Game:
    """
    Connect Four game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[Connect4Board] = None):
        self.board = board or Connect4Board()

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get legal moves for a color

        Legal moves are drops into columns that have space (top row empty)
        """
        moves = []

        for column in range(self.board.COLS):
            if self.board.can_place_piece(column):
                row = self.board.get_drop_row(column)
                if row is not None:
                    piece = Piece(color, (row, column))
                    move = Move(column=column, row=row, piece=piece)
                    moves.append(move)

        return moves

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over"""
        if self.board.is_game_over():
            return True

        # Also check if current player has no legal moves (shouldn't happen in Connect 4)
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

    def get_piece_count(self, color: Color) -> int:
        """Get number of pieces for a color"""
        return len(self.board.get_pieces(color))


def test_connect4_game():
    """Test Connect Four game engine"""
    print("=" * 70)
    print("TESTING CONNECT FOUR GAME ENGINE")
    print("=" * 70)

    game = Connect4Game()
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for red
    legal_moves = game.get_legal_moves(Color.RED)
    print(f"\nRed has {len(legal_moves)} legal moves")

    # Show all legal moves
    print("\nAll legal moves:")
    for i, move in enumerate(legal_moves):
        print(f"  {i+1}. Column {move.column} → Row {move.row}")

    # Make a series of moves
    print("\n" + "=" * 70)
    print("Making moves...")

    move_sequence = [
        (Color.RED, 3),      # Red column 3
        (Color.YELLOW, 3),   # Yellow column 3
        (Color.RED, 3),      # Red column 3
        (Color.YELLOW, 3),   # Yellow column 3
    ]

    for color, column in move_sequence:
        legal_moves = game.get_legal_moves(color)
        move = next((m for m in legal_moves if m.column == column), None)
        if move:
            print(f"\n{color.value.upper()} plays column {column}")
            game.make_move(move)
            print(game.board)
            if game.board.check_win(color):
                print(f"✓ {color.value.upper()} WINS!")
        else:
            print(f"No legal move available for column {column}")
            break

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_connect4_game()
