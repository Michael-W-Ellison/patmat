"""
Nine Men's Morris game logic with three distinct phases.

Game Phases:
1. Placement: Players alternate placing 9 pieces each
2. Movement: Players move pieces to adjacent positions
3. Flying: When down to 3 pieces, can move to any empty position

Mill Formation: Getting 3 in a row allows removing an opponent piece
Win Conditions: Reduce opponent to 2 pieces OR block all their moves
"""

from typing import Tuple, List, Optional, Union
from enum import Enum
from .morris_board import MorrisBoard, Color, Piece


class GamePhase(Enum):
    """Game phases"""
    PLACEMENT = "placement"
    MOVEMENT = "movement"
    FLYING = "flying"


class MorrisMove:
    """
    Represents a Morris move.

    Placement phase: (position,)
    Movement phase: (from_pos, to_pos)
    Remove piece: (remove_pos,) - happens after forming a mill
    """

    def __init__(self, *args):
        if len(args) == 1:
            # Placement or removal
            self.is_placement = True
            self.is_removal = False
            self.position = args[0]
            self.from_pos = None
            self.to_pos = None
        elif len(args) == 2:
            # Movement
            self.is_placement = False
            self.is_removal = False
            self.from_pos = args[0]
            self.to_pos = args[1]
            self.position = None
        else:
            raise ValueError("Invalid move format")

    def __repr__(self):
        if self.is_placement:
            return f"Place({self.position})"
        else:
            return f"Move({self.from_pos}→{self.to_pos})"

    def __eq__(self, other):
        if not isinstance(other, MorrisMove):
            return False
        return (self.is_placement == other.is_placement and
                self.position == other.position and
                self.from_pos == other.from_pos and
                self.to_pos == other.to_pos)

    def __hash__(self):
        return hash((self.is_placement, self.position, self.from_pos, self.to_pos))


class MorrisGame:
    """Nine Men's Morris game engine"""

    def __init__(self, board: Optional[MorrisBoard] = None,
                 current_player: Color = Color.WHITE):
        """Initialize game"""
        self.board = board if board is not None else MorrisBoard()
        self.current_player = current_player
        self.winner: Optional[Color] = None
        self.is_draw = False

        # Track placements
        self.white_pieces_placed = 0
        self.black_pieces_placed = 0

        # Phase tracking
        self.phase = GamePhase.PLACEMENT
        self.pending_removal = False  # True if mill formed, need to remove piece

    def copy(self):
        """Create a copy of the game state"""
        new_game = MorrisGame(
            board=self.board.copy(),
            current_player=self.current_player
        )
        new_game.winner = self.winner
        new_game.is_draw = self.is_draw
        new_game.white_pieces_placed = self.white_pieces_placed
        new_game.black_pieces_placed = self.black_pieces_placed
        new_game.phase = self.phase
        new_game.pending_removal = self.pending_removal
        return new_game

    def get_game_phase(self) -> GamePhase:
        """Determine current game phase"""
        if self.white_pieces_placed < 9 or self.black_pieces_placed < 9:
            return GamePhase.PLACEMENT

        # Check if current player has 3 pieces → flying phase
        if self.board.count_pieces(self.current_player) == 3:
            return GamePhase.FLYING

        return GamePhase.MOVEMENT

    def get_legal_moves(self) -> List[Union[MorrisMove, int]]:
        """
        Generate all legal moves.

        Returns list of MorrisMove objects or integers (for removal).
        """
        if self.is_game_over():
            return []

        # If mill was formed, must remove opponent piece
        if self.pending_removal:
            removable = self.board.get_removable_pieces(self.current_player)
            return removable  # Return positions that can be removed

        self.phase = self.get_game_phase()

        if self.phase == GamePhase.PLACEMENT:
            return self._get_placement_moves()
        elif self.phase == GamePhase.MOVEMENT:
            return self._get_movement_moves()
        else:  # FLYING
            return self._get_flying_moves()

    def _get_placement_moves(self) -> List[MorrisMove]:
        """Get legal placement moves"""
        empty = self.board.get_empty_positions()
        return [MorrisMove(pos) for pos in empty]

    def _get_movement_moves(self) -> List[MorrisMove]:
        """Get legal movement moves (to adjacent positions)"""
        moves = []
        my_pieces = self.board.get_pieces(self.current_player)

        for piece in my_pieces:
            from_pos = piece.position
            # Can move to adjacent empty positions
            for to_pos in self.board.get_adjacent_positions(from_pos):
                if self.board.get_piece(to_pos) is None:
                    moves.append(MorrisMove(from_pos, to_pos))

        return moves

    def _get_flying_moves(self) -> List[MorrisMove]:
        """Get legal flying moves (to any empty position)"""
        moves = []
        my_pieces = self.board.get_pieces(self.current_player)
        empty = self.board.get_empty_positions()

        for piece in my_pieces:
            from_pos = piece.position
            for to_pos in empty:
                moves.append(MorrisMove(from_pos, to_pos))

        return moves

    def make_move(self, move: Union[MorrisMove, int]) -> 'MorrisGame':
        """
        Execute a move and return new game state.

        Returns new MorrisGame (immutable-style).
        """
        new_game = self.copy()

        # Handle removal (if move is just an integer)
        if isinstance(move, int):
            if not new_game.pending_removal:
                raise ValueError("Not in removal state")
            new_game.board.remove_piece(move)
            new_game.pending_removal = False
            # Switch turns after removal
            new_game.current_player = new_game.current_player.opposite()
            new_game._check_game_over()
            return new_game

        # Handle placement
        if move.is_placement:
            success = new_game.board.place_piece(move.position, new_game.current_player)
            if not success:
                raise ValueError(f"Invalid placement: {move.position}")

            # Track placements
            if new_game.current_player == Color.WHITE:
                new_game.white_pieces_placed += 1
            else:
                new_game.black_pieces_placed += 1

            # Check if mill formed
            if new_game.board.is_mill(move.position, new_game.current_player):
                new_game.pending_removal = True
                # Don't switch turns yet - need to remove piece first
            else:
                # Switch turns
                new_game.current_player = new_game.current_player.opposite()

        # Handle movement/flying
        else:
            success = new_game.board.move_piece(move.from_pos, move.to_pos)
            if not success:
                raise ValueError(f"Invalid move: {move.from_pos} → {move.to_pos}")

            # Check if mill formed
            if new_game.board.is_mill(move.to_pos, new_game.current_player):
                new_game.pending_removal = True
                # Don't switch turns yet
            else:
                # Switch turns
                new_game.current_player = new_game.current_player.opposite()

        new_game._check_game_over()
        return new_game

    def _check_game_over(self):
        """Check if game is over"""
        # Win by reducing opponent to 2 pieces
        white_count = self.board.count_pieces(Color.WHITE)
        black_count = self.board.count_pieces(Color.BLACK)

        if white_count < 3 and self.white_pieces_placed >= 9:
            self.winner = Color.BLACK
            return

        if black_count < 3 and self.black_pieces_placed >= 9:
            self.winner = Color.WHITE
            return

        # Win by blocking all opponent moves (only in movement/flying phase)
        if self.phase != GamePhase.PLACEMENT and not self.pending_removal:
            legal_moves = self.get_legal_moves()
            if not legal_moves:
                # Current player has no moves → opponent wins
                self.winner = self.current_player.opposite()
                return

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
        phase_str = self.phase.value if self.phase else "unknown"
        return f"Morris - {self.current_player} to move ({phase_str})\n{self.board}"
