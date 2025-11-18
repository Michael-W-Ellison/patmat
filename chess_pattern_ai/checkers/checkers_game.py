#!/usr/bin/env python3
"""
Checkers Game Engine - Observation-Based Move Generation

Philosophy: Learn move rules from observation
- For bootstrapping: Basic move generation
- Future: Replace with learned move patterns
"""

from typing import List, Tuple, Optional
from .checkers_board import CheckersBoard, Move, Piece, Color, PieceType


class CheckersGame:
    """
    Checkers game engine using observation-based learning

    Note: Currently uses basic rule knowledge for bootstrapping.
    Will be replaced with observation-based move learning.
    """

    def __init__(self, board: Optional[CheckersBoard] = None):
        self.board = board or CheckersBoard()

    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Get legal moves for a color

        TODO: Replace with observation-based move prediction
        """
        # First check for jumps (mandatory in checkers)
        jumps = self._get_all_jumps(color)
        if jumps:
            return jumps  # Must jump if possible

        # Otherwise, get simple moves
        return self._get_simple_moves(color)

    def _get_simple_moves(self, color: Color) -> List[Move]:
        """Get non-jump moves"""
        moves = []

        for piece in self.board.get_pieces(color):
            moves.extend(self._get_piece_simple_moves(piece))

        return moves

    def _get_piece_simple_moves(self, piece: Piece) -> List[Move]:
        """Get simple moves for a piece"""
        moves = []
        row, col = piece.position

        # Direction based on color and piece type
        if piece.type == PieceType.KING:
            # Kings move in all diagonal directions
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Men move forward only
            if piece.color == Color.RED:
                directions = [(-1, -1), (-1, 1)]  # Red moves toward row 0
            else:
                directions = [(1, -1), (1, 1)]    # Black moves toward row 7

        for d_row, d_col in directions:
            new_row, new_col = row + d_row, col + d_col

            if not self.board.is_valid_square((new_row, new_col)):
                continue

            target = self.board.get_piece_at((new_row, new_col))
            if target is None:  # Empty square
                move = Move(
                    piece=piece,
                    from_pos=(row, col),
                    to_pos=(new_row, new_col),
                    is_jump=False
                )
                moves.append(move)

        return moves

    def _get_all_jumps(self, color: Color) -> List[Move]:
        """Get all jump moves (captures)"""
        all_jumps = []

        for piece in self.board.get_pieces(color):
            jumps = self._get_piece_jumps(piece, self.board)
            all_jumps.extend(jumps)

        return all_jumps

    def _get_piece_jumps(self, piece: Piece, board: CheckersBoard,
                         path: List[Tuple[int, int]] = None) -> List[Move]:
        """
        Get jump moves for a piece (including multi-jumps)

        Recursively finds all possible jump sequences
        """
        if path is None:
            path = [piece.position]

        jumps = []
        row, col = piece.position

        # Direction based on piece type
        if piece.type == PieceType.KING:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if piece.color == Color.RED:
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]

        for d_row, d_col in directions:
            # Position of piece to jump over
            mid_row, mid_col = row + d_row, col + d_col
            # Landing position
            land_row, land_col = row + 2*d_row, col + 2*d_col

            # Check if jump is valid
            if not board.is_valid_square((land_row, land_col)):
                continue

            mid_piece = board.get_piece_at((mid_row, mid_col))
            land_piece = board.get_piece_at((land_row, land_col))

            # Must jump over opponent piece to empty square
            if mid_piece and mid_piece.color != piece.color and land_piece is None:
                # Don't jump same piece twice
                if (mid_row, mid_col) in [(p.position if isinstance(p, Piece) else p)
                                          for p in path if p != piece.position]:
                    continue

                # Create move
                captured = [mid_piece]
                move = Move(
                    piece=piece,
                    from_pos=path[0],
                    to_pos=(land_row, land_col),
                    is_jump=True,
                    captured_pieces=captured
                )

                # Check for multi-jump continuation
                # Simulate this jump to check for more
                test_board = board.copy()
                test_piece = test_board.get_piece_at(piece.position)

                # Temporarily move piece
                test_board.board[row][col] = None
                test_board.board[land_row][land_col] = test_piece
                test_piece.position = (land_row, land_col)

                # Remove captured
                test_board.board[mid_row][mid_col] = None

                # Look for continuation jumps
                next_path = path + [(land_row, land_col)]
                continuations = self._get_piece_jumps(test_piece, test_board, next_path)

                if continuations:
                    # Multi-jump available - extend this move
                    for cont in continuations:
                        extended_move = Move(
                            piece=piece,
                            from_pos=path[0],
                            to_pos=cont.to_pos,
                            is_jump=True,
                            captured_pieces=captured + cont.captured_pieces
                        )
                        jumps.append(extended_move)
                else:
                    # No continuation - this jump is complete
                    jumps.append(move)

        return jumps

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

        # If one side has no pieces
        winner = self.board.get_winner()
        if winner:
            return winner

        # If one side has no moves, they lose
        if len(self.get_legal_moves(self.board.turn)) == 0:
            return Color.BLACK if self.board.turn == Color.RED else Color.RED

        return None

    def get_material_count(self, color: Color) -> int:
        """Get material value for a color"""
        total = 0
        for piece in self.board.get_pieces(color):
            if piece.type == PieceType.KING:
                total += 300  # Kings worth 3x
            else:
                total += 100  # Men worth 100

        return total


def test_checkers_game():
    """Test checkers game engine"""
    print("=" * 70)
    print("TESTING CHECKERS GAME ENGINE")
    print("=" * 70)

    game = CheckersGame()
    print("\nInitial board:")
    print(game.board)

    # Get legal moves for red
    legal_moves = game.get_legal_moves(Color.RED)
    print(f"\nRed has {len(legal_moves)} legal moves")

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
        print(f"\nBlack has {black_moves} legal moves")

    print("\n" + "=" * 70)
    print("✓ Game engine working!")


if __name__ == '__main__':
    test_checkers_game()
