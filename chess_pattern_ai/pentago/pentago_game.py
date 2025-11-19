"""
Pentago game logic and move generation.

A move in Pentago consists of two parts:
1. Place a marble on an empty position
2. Rotate one quadrant 90° CW or CCW
"""

from typing import Tuple, List, Optional
from .pentago_board import PentagoBoard, Color, Marble


class PentagoMove:
    """
    Represents a Pentago move: placement + rotation.

    Format: (placement_pos, quadrant, direction)
    - placement_pos: (row, col) where to place marble
    - quadrant: 0-3 (which quadrant to rotate)
    - direction: 'CW' or 'CCW'
    """

    def __init__(self, placement: Tuple[int, int], quadrant: int, direction: str):
        self.placement = placement
        self.quadrant = quadrant
        self.direction = direction

    def __repr__(self):
        return f"Move({self.placement}, Q{self.quadrant} {self.direction})"

    def __eq__(self, other):
        if not isinstance(other, PentagoMove):
            return False
        return (self.placement == other.placement and
                self.quadrant == other.quadrant and
                self.direction == other.direction)

    def __hash__(self):
        return hash((self.placement, self.quadrant, self.direction))


class PentagoGame:
    """Pentago game engine"""

    def __init__(self, board: Optional[PentagoBoard] = None,
                 current_player: Color = Color.WHITE):
        """Initialize game with optional board state"""
        self.board = board if board is not None else PentagoBoard()
        self.current_player = current_player
        self.winner: Optional[Color] = None
        self.is_draw = False

    def copy(self):
        """Create a copy of the game state"""
        new_game = PentagoGame(
            board=self.board.copy(),
            current_player=self.current_player
        )
        new_game.winner = self.winner
        new_game.is_draw = self.is_draw
        return new_game

    def get_legal_moves(self) -> List[PentagoMove]:
        """
        Generate all legal moves.

        Each move = placement + rotation.
        Number of moves = empty_positions × 4_quadrants × 2_directions
        Max: 36 × 4 × 2 = 288 moves (first move)
        Typical: 20-30 × 8 = 160-240 moves
        """
        if self.is_game_over():
            return []

        moves = []
        empty_positions = self.board.get_empty_positions()

        # For each empty position
        for pos in empty_positions:
            # For each quadrant
            for quadrant in range(4):
                # For each rotation direction
                for direction in ['CW', 'CCW']:
                    moves.append(PentagoMove(pos, quadrant, direction))

        return moves

    def make_move(self, move: PentagoMove) -> 'PentagoGame':
        """
        Execute a move and return new game state.

        Returns new PentagoGame (immutable-style).
        """
        new_game = self.copy()

        # Place marble
        success = new_game.board.place_marble(move.placement, new_game.current_player)
        if not success:
            raise ValueError(f"Invalid placement: {move.placement}")

        # Rotate quadrant
        new_game.board.rotate_quadrant(move.quadrant, move.direction)

        # Check for win
        if new_game.board.check_five_in_row(Color.WHITE) and \
           new_game.board.check_five_in_row(Color.BLACK):
            # Both players have 5 in a row (simultaneous win) → draw
            new_game.is_draw = True
            new_game.winner = None
        elif new_game.board.check_five_in_row(new_game.current_player):
            # Current player wins
            new_game.winner = new_game.current_player
        elif new_game.board.is_full():
            # Board full, no winner → draw
            new_game.is_draw = True
            new_game.winner = None
        else:
            # Continue playing, switch turns
            new_game.current_player = new_game.current_player.opposite()

        return new_game

    def is_game_over(self) -> bool:
        """Check if game is over"""
        return (self.winner is not None or
                self.is_draw or
                self.board.is_full())

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
        return f"Pentago - {self.current_player} to move\n{self.board}"
