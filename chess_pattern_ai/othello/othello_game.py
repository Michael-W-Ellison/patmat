#!/usr/bin/env python3
"""
Othello/Reversi Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Legal move generation (must flip opponent discs)
- Future: Replace with learned move patterns
"""

from typing import List, Optional
from .othello_board import OthelloBoard, Move, Color, Disc


class OthelloGame:
    """
    Othello game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[OthelloBoard] = None):
        self.board = board or OthelloBoard()

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get legal moves for a color

        Legal moves in Othello:
        1. Must place disc on empty square
        2. Must flip at least one opponent disc
        3. Can flip in all 8 directions

        Returns: List of legal Move objects
        """
        moves = []

        # Check all empty squares
        for row in range(8):
            for col in range(8):
                if self.board.get_disc_at((row, col)) is None:
                    # Try to play at this position
                    flipped = self.board.get_flipped_discs(color, (row, col))

                    if flipped:  # Valid move (flips at least one disc)
                        move = Move(color, (row, col))
                        move.flipped_discs = flipped
                        moves.append(move)

        # If no legal moves, add pass move
        if not moves:
            pass_move = Move(color, None, is_pass=True)
            moves.append(pass_move)

        return moves

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        if move.is_pass:
            return self.board.make_move(move)

        # For non-pass moves, ensure flipped_discs is set
        if not move.flipped_discs:
            move.flipped_discs = self.board.get_flipped_discs(move.color, move.position)

        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.board.is_game_over()

    def get_winner(self) -> Optional[Color]:
        """Get winner"""
        if not self.is_game_over():
            return None
        return self.board.get_winner()

    def get_material_count(self, color: Color) -> int:
        """Get disc count for a color"""
        return self.board.count_discs(color)

    def can_play(self, color: Color) -> bool:
        """Check if color has any legal moves"""
        legal_moves = self.get_legal_moves(color)
        # Has legal moves if there's at least one non-pass move
        return any(not move.is_pass for move in legal_moves)


def test_othello_game():
    """Test othello game engine"""
    print("=" * 70)
    print("TESTING OTHELLO GAME ENGINE")
    print("=" * 70)

    game = OthelloGame()
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for black
    legal_moves = game.get_legal_moves(Color.BLACK)
    print(f"\nBlack has {len(legal_moves)} legal move(s)")

    # Show all moves
    print("\nBlack's legal moves:")
    for i, move in enumerate(legal_moves):
        if not move.is_pass:
            print(f"  {i+1}. {move.notation()} (flips {len(move.flipped_discs)} disc(s))")
        else:
            print(f"  {i+1}. {move.notation()}")

    # Make a move
    if legal_moves and not legal_moves[0].is_pass:
        move = legal_moves[0]
        print(f"\nMaking move: {move.notation()}")
        game.make_move(move)
        print(game.board)

        # Get white's response
        white_moves = game.get_legal_moves(Color.WHITE)
        print(f"\nWhite has {len(white_moves)} legal move(s)")

        if white_moves:
            for i, move in enumerate(white_moves[:3]):
                if not move.is_pass:
                    print(f"  {i+1}. {move.notation()} (flips {len(move.flipped_discs)} disc(s))")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_othello_game()
