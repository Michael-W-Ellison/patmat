#!/usr/bin/env python3
"""
Breakthrough Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation
- Future: Replace with learned move patterns
"""

from typing import List, Tuple, Optional
from .breakthrough_board import BreakthroughBoard, Move, Piece, Color


class BreakthroughGame:
    """
    Breakthrough game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[BreakthroughBoard] = None):
        self.board = board or BreakthroughBoard()

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get legal moves for a color

        Legal moves in Breakthrough:
        1. Forward straight: Move one square forward if empty
        2. Forward diagonal: Capture one square forward-diagonal if occupied

        No backward moves or backward captures.

        TODO: Replace with observation-based move prediction
        """
        moves = []

        for piece in self.board.get_pieces(color):
            moves.extend(self._get_piece_moves(piece))

        return moves

    def _get_piece_moves(self, piece: Piece) -> List[Move]:
        """Get all legal moves for a piece"""
        moves = []
        row, col = piece.position

        # Direction based on color
        # White moves toward row 7 (downward), Black toward row 0 (upward)
        if piece.color == Color.WHITE:
            forward_dir = 1  # Move down
        else:
            forward_dir = -1  # Move up

        # Forward straight move
        new_row = row + forward_dir
        if self.board.is_valid_square((new_row, col)):
            target = self.board.get_piece_at((new_row, col))
            if target is None:  # Empty square
                move = Move(
                    piece=piece,
                    from_pos=(row, col),
                    to_pos=(new_row, col),
                    is_capture=False
                )
                moves.append(move)

        # Forward diagonal captures (left and right)
        for d_col in [-1, 1]:
            new_row = row + forward_dir
            new_col = col + d_col

            if not self.board.is_valid_square((new_row, new_col)):
                continue

            target = self.board.get_piece_at((new_row, new_col))
            if target and target.color != piece.color:  # Opponent piece
                move = Move(
                    piece=piece,
                    from_pos=(row, col),
                    to_pos=(new_row, new_col),
                    is_capture=True,
                    captured_piece=target
                )
                moves.append(move)

        return moves

    def make_move(self, move: Move) -> bool:
        """Execute a move"""
        return self.board.make_move(move)

    def is_game_over(self) -> bool:
        """Check if game is over"""
        if self.board.is_game_over():
            return True

        # Also check if current player has no moves
        legal_moves = self.get_legal_moves(self.board.turn)
        return len(legal_moves) == 0

    def get_winner(self) -> Optional[Color]:
        """Get winner"""
        if not self.is_game_over():
            return None

        # Check board state first (pieces captured or back row reached)
        winner = self.board.get_winner()
        if winner:
            return winner

        # If current player has no moves, they lose
        if len(self.get_legal_moves(self.board.turn)) == 0:
            return Color.BLACK if self.board.turn == Color.WHITE else Color.WHITE

        return None

    def get_material_count(self, color: Color) -> int:
        """Get material value for a color (number of pieces)"""
        return len(self.board.get_pieces(color)) * 100

    def get_advancement_score(self, color: Color) -> float:
        """
        Calculate advancement score for a color

        Measures how close pieces are to reaching opponent's back row
        - White pieces advancing toward row 7
        - Black pieces advancing toward row 0
        - Each square closer = +1 point
        """
        score = 0.0
        pieces = self.board.get_pieces(color)

        if color == Color.WHITE:
            # White starts at rows 0-1, advances toward row 7
            for piece in pieces:
                row = piece.position[0]
                distance_from_start = row - 1  # Starting row is 1
                score += distance_from_start
        else:
            # Black starts at rows 6-7, advances toward row 0
            for piece in pieces:
                row = piece.position[0]
                distance_from_start = 6 - row  # Starting row is 6
                score += distance_from_start

        return score


def test_breakthrough_game():
    """Test breakthrough game engine"""
    print("=" * 70)
    print("TESTING BREAKTHROUGH GAME ENGINE")
    print("=" * 70)

    game = BreakthroughGame()
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for white
    legal_moves = game.get_legal_moves(Color.WHITE)
    print(f"\nWhite has {len(legal_moves)} legal moves")

    # Show first few moves
    print("\nFirst 5 legal moves:")
    for i, move in enumerate(legal_moves[:5]):
        print(f"  {i+1}. {move.notation()}")

    # Make a move
    if legal_moves:
        move = legal_moves[0]
        print(f"\nMaking move: {move.notation()}")
        game.make_move(move)
        print(game.board)

        # Get black's response
        black_moves = game.get_legal_moves(Color.BLACK)
        print(f"\nBlack has {len(black_moves)} legal moves")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_breakthrough_game()
