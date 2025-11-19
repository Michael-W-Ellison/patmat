"""
Arimaa game logic with multi-step turns.

Game Rules:
- Each turn consists of up to 4 steps
- A step can be: move own piece, push enemy piece, pull enemy piece
- Push: Move enemy piece, then move your stronger piece into its space
- Pull: Move your piece, then move weaker enemy piece into vacated space
- Rabbits cannot move backward
- Goal: Get rabbit to opponent's goal row OR eliminate all opponent rabbits
"""

from typing import Tuple, List, Optional
from enum import Enum
from .arimaa_board import ArimaaBoard, Color, Piece, PieceType


class StepType(Enum):
    """Types of steps in Arimaa"""
    MOVE = "move"      # Normal movement
    PUSH = "push"      # Push enemy piece
    PULL = "pull"      # Pull enemy piece


class ArimaaStep:
    """Represents a single step in Arimaa"""

    def __init__(self, step_type: StepType, from_pos: Tuple[int, int],
                 to_pos: Tuple[int, int], enemy_pos: Optional[Tuple[int, int]] = None):
        self.step_type = step_type
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.enemy_pos = enemy_pos  # For push/pull

    def __repr__(self):
        if self.step_type == StepType.MOVE:
            return f"Move({self.from_pos}→{self.to_pos})"
        elif self.step_type == StepType.PUSH:
            return f"Push({self.enemy_pos}→{self.to_pos}, {self.from_pos}→{self.enemy_pos})"
        else:  # PULL
            return f"Pull({self.from_pos}→{self.to_pos}, {self.enemy_pos}→{self.from_pos})"

    def __eq__(self, other):
        if not isinstance(other, ArimaaStep):
            return False
        return (self.step_type == other.step_type and
                self.from_pos == other.from_pos and
                self.to_pos == other.to_pos and
                self.enemy_pos == other.enemy_pos)

    def __hash__(self):
        return hash((self.step_type, self.from_pos, self.to_pos, self.enemy_pos))


class ArimaaGame:
    """Arimaa game engine"""

    def __init__(self, board: Optional[ArimaaBoard] = None,
                 current_player: Color = Color.GOLD):
        """Initialize game"""
        self.board = board if board is not None else ArimaaBoard()
        self.current_player = current_player
        self.winner: Optional[Color] = None
        self.is_draw = False
        self.steps_taken = 0  # Steps in current turn (max 4)

    def copy(self):
        """Create a copy of the game state"""
        new_game = ArimaaGame(
            board=self.board.copy(),
            current_player=self.current_player
        )
        new_game.winner = self.winner
        new_game.is_draw = self.is_draw
        new_game.steps_taken = self.steps_taken
        return new_game

    def get_legal_steps(self) -> List[ArimaaStep]:
        """
        Generate all legal steps for current player.

        Returns list of ArimaaStep objects (moves, pushes, pulls).
        """
        if self.is_game_over():
            return []

        steps = []
        my_pieces = self.board.get_pieces(self.current_player)

        for piece in my_pieces:
            # Skip frozen pieces
            if self.board.is_frozen(piece.position):
                continue

            from_pos = piece.position

            # Try normal moves
            for to_pos in self.board.get_adjacent_positions(from_pos):
                if self.board.get_piece(to_pos) is not None:
                    continue  # Can't move to occupied square

                # Check rabbit backward movement
                if piece.type == PieceType.RABBIT:
                    if not self._is_rabbit_forward(from_pos, to_pos, piece.color):
                        continue

                steps.append(ArimaaStep(StepType.MOVE, from_pos, to_pos))

            # Try pushes (move enemy to adjacent, then move into enemy's spot)
            for enemy_pos in self.board.get_adjacent_positions(from_pos):
                enemy = self.board.get_piece(enemy_pos)
                if enemy is None or enemy.color == piece.color:
                    continue
                if not piece.is_stronger_than(enemy):
                    continue  # Can only push weaker pieces

                # Find where to push enemy
                for push_to in self.board.get_adjacent_positions(enemy_pos):
                    if self.board.get_piece(push_to) is not None:
                        continue  # Push destination must be empty

                    # Check rabbit backward movement for enemy
                    if enemy.type == PieceType.RABBIT:
                        if not self._is_rabbit_forward(enemy_pos, push_to, enemy.color):
                            continue

                    steps.append(ArimaaStep(StepType.PUSH, from_pos, enemy_pos, push_to))

            # Try pulls (move self, then pull weaker enemy to vacated square)
            for to_pos in self.board.get_adjacent_positions(from_pos):
                if self.board.get_piece(to_pos) is not None:
                    continue  # Destination must be empty

                # Check rabbit backward movement
                if piece.type == PieceType.RABBIT:
                    if not self._is_rabbit_forward(from_pos, to_pos, piece.color):
                        continue

                # Find enemy to pull
                for enemy_pos in self.board.get_adjacent_positions(from_pos):
                    if enemy_pos == to_pos:
                        continue  # Can't pull from where we're moving
                    enemy = self.board.get_piece(enemy_pos)
                    if enemy is None or enemy.color == piece.color:
                        continue
                    if not piece.is_stronger_than(enemy):
                        continue  # Can only pull weaker pieces

                    # Check rabbit backward movement for enemy
                    if enemy.type == PieceType.RABBIT:
                        if not self._is_rabbit_forward(enemy_pos, from_pos, enemy.color):
                            continue

                    steps.append(ArimaaStep(StepType.PULL, from_pos, to_pos, enemy_pos))

        return steps

    def make_step(self, step: ArimaaStep) -> 'ArimaaGame':
        """
        Execute a single step and return new game state.

        Returns new ArimaaGame.
        """
        new_game = self.copy()

        if step.step_type == StepType.MOVE:
            new_game.board.move_piece(step.from_pos, step.to_pos)

        elif step.step_type == StepType.PUSH:
            # Push: Move enemy to push_to, then move self to enemy's position
            enemy_from = step.enemy_pos
            enemy_to = step.to_pos  # This is where enemy goes
            self_from = step.from_pos
            self_to = enemy_from  # Move into enemy's vacated square

            new_game.board.move_piece(enemy_from, enemy_to)
            new_game.board.move_piece(self_from, self_to)

        elif step.step_type == StepType.PULL:
            # Pull: Move self first, then pull enemy to vacated square
            self_from = step.from_pos
            self_to = step.to_pos
            enemy_from = step.enemy_pos
            enemy_to = self_from  # Enemy moves to vacated square

            new_game.board.move_piece(self_from, self_to)
            new_game.board.move_piece(enemy_from, enemy_to)

        # Check traps after step
        new_game.board.check_and_remove_trapped()

        # Increment step counter
        new_game.steps_taken += 1

        # Check for win condition
        new_game._check_game_over()

        # If turn is complete (4 steps) or game over, switch players
        if new_game.steps_taken >= 4 or new_game.is_game_over():
            if not new_game.is_game_over():
                new_game.current_player = new_game.current_player.opposite()
                new_game.steps_taken = 0

        return new_game

    def _is_rabbit_forward(self, from_pos: Tuple[int, int],
                           to_pos: Tuple[int, int], color: Color) -> bool:
        """Check if rabbit move is forward (not backward)"""
        from_row = from_pos[0]
        to_row = to_pos[0]

        if color == Color.GOLD:
            # Gold moves up (increasing row)
            return to_row >= from_row
        else:
            # Silver moves down (decreasing row)
            return to_row <= from_row

    def _check_game_over(self):
        """Check if game is over"""
        # Win by getting rabbit to goal
        if self.board.has_rabbit_on_goal(Color.GOLD):
            self.winner = Color.GOLD
            return
        if self.board.has_rabbit_on_goal(Color.SILVER):
            self.winner = Color.SILVER
            return

        # Win by eliminating all opponent rabbits
        if self.board.count_rabbits(Color.GOLD) == 0:
            self.winner = Color.SILVER
            return
        if self.board.count_rabbits(Color.SILVER) == 0:
            self.winner = Color.GOLD
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
        gold_rabbits = self.board.count_rabbits(Color.GOLD)
        silver_rabbits = self.board.count_rabbits(Color.SILVER)
        return (f"Arimaa - {self.current_player} to move (Step {self.steps_taken}/4)\n"
                f"Gold: {len(self.board.pieces_gold)} pieces ({gold_rabbits} rabbits)\n"
                f"Silver: {len(self.board.pieces_silver)} pieces ({silver_rabbits} rabbits)\n"
                f"{self.board}")
