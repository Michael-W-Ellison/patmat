"""
Lines of Action game logic with unique movement rules.

Movement Rule:
- Move along any line (horizontal, vertical, diagonal)
- Move distance = number of pieces (both colors) in that line
- Can jump over own pieces
- Cannot jump over opponent pieces
- Can capture by landing on opponent piece

Win Condition:
- Connect all your pieces into a single group
"""

from typing import Tuple, List, Optional
from .loa_board import LOABoard, Color, Piece


class LOAMove:
    """Represents a Lines of Action move"""

    def __init__(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        self.from_pos = from_pos
        self.to_pos = to_pos

    def __repr__(self):
        return f"Move({self.from_pos}→{self.to_pos})"

    def __eq__(self, other):
        if not isinstance(other, LOAMove):
            return False
        return self.from_pos == other.from_pos and self.to_pos == other.to_pos

    def __hash__(self):
        return hash((self.from_pos, self.to_pos))


class LOAGame:
    """Lines of Action game engine"""

    # Eight directions: horizontal, vertical, diagonal
    DIRECTIONS = [
        (0, 1),   # Right
        (0, -1),  # Left
        (1, 0),   # Down
        (-1, 0),  # Up
        (1, 1),   # Down-right
        (1, -1),  # Down-left
        (-1, 1),  # Up-right
        (-1, -1)  # Up-left
    ]

    def __init__(self, board: Optional[LOABoard] = None,
                 current_player: Color = Color.BLACK):
        """Initialize game"""
        self.board = board if board is not None else LOABoard()
        self.current_player = current_player
        self.winner: Optional[Color] = None
        self.is_draw = False

    def copy(self):
        """Create a copy of the game state"""
        new_game = LOAGame(
            board=self.board.copy(),
            current_player=self.current_player
        )
        new_game.winner = self.winner
        new_game.is_draw = self.is_draw
        return new_game

    def get_legal_moves(self) -> List[LOAMove]:
        """
        Generate all legal moves for current player.

        For each piece, try moving in each of 8 directions
        by exactly the number of pieces in that line.
        """
        if self.is_game_over():
            return []

        moves = []
        my_pieces = self.board.get_pieces(self.current_player)

        for piece in my_pieces:
            from_pos = piece.position

            # Try each direction
            for direction in self.DIRECTIONS:
                # Count pieces in this line
                count = self.board.count_pieces_in_line(from_pos, direction)

                # Calculate destination
                dr, dc = direction
                to_pos = (from_pos[0] + dr * count, from_pos[1] + dc * count)

                # Check if destination is valid
                if not self.board._is_valid_position(to_pos):
                    continue

                # Check if can jump to destination
                if self.board.can_jump_to(from_pos, to_pos):
                    moves.append(LOAMove(from_pos, to_pos))

        return moves

    def make_move(self, move: LOAMove) -> 'LOAGame':
        """
        Execute a move and return new game state.

        Returns new LOAGame (immutable-style).
        """
        new_game = self.copy()

        # Make the move
        success = new_game.board.move_piece(move.from_pos, move.to_pos)
        if not success:
            raise ValueError(f"Invalid move: {move}")

        # Check for win (current player wins if all pieces connected)
        if new_game.board.is_connected(new_game.current_player):
            new_game.winner = new_game.current_player
            return new_game

        # Check if opponent won (can happen if captured their last isolating piece)
        if new_game.board.is_connected(new_game.current_player.opposite()):
            new_game.winner = new_game.current_player.opposite()
            return new_game

        # Switch turns
        new_game.current_player = new_game.current_player.opposite()

        # Check if opponent has no legal moves → current player wins
        opponent_moves = new_game.get_legal_moves()
        if not opponent_moves:
            new_game.winner = new_game.current_player.opposite()

        return new_game

    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.winner is not None or self.is_draw

    def get_result(self, perspective: Color) -> str:
        """
        Get game result from perspective of a color.

        Returns: 'win', 'loss', 'draw', or 'ongoing'
        """
        if not self.is_game_over():
            return 'ongoing'

        if self.is_draw:
            return 'draw'

        if self.winner == perspective:
            return 'win'
        else:
            return 'loss'

    def get_winner(self) -> Optional[Color]:
        """Get the winner, if any"""
        return self.winner

    def __str__(self):
        """String representation"""
        white_groups = self.board.count_groups(Color.WHITE)
        black_groups = self.board.count_groups(Color.BLACK)
        return (f"Lines of Action - {self.current_player} to move\n"
                f"White: {len(self.board.pieces_white)} pieces, {white_groups} groups\n"
                f"Black: {len(self.board.pieces_black)} pieces, {black_groups} groups\n"
                f"{self.board}")
