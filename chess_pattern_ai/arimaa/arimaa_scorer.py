"""
Arimaa differential scoring system.

Scoring philosophy:
    score = my_advantage - opponent_advantage

Components:
1. Rabbit advancement (closer to goal = better)
2. Rabbit count (preserve rabbits)
3. Material strength (piece values)
4. Trap control (pieces near traps)
5. Piece advancement (forward progress)

The AI learns which patterns lead to wins through observation.
"""

from .arimaa_board import ArimaaBoard, Color, PieceType
from .arimaa_game import ArimaaGame, ArimaaStep, StepType
from typing import Tuple


class ArimaaScorer:
    """
    Differential scorer for Arimaa.

    Evaluates position strength based on rabbit advancement,
    material, trap control, and piece positioning.
    """

    def __init__(self):
        """Initialize scorer with default weights"""
        # Rabbit advancement (most important)
        self.RABBIT_ON_GOAL = 10000  # Win condition
        self.RABBIT_ADVANCEMENT = 200  # Per row advanced

        # Rabbit count
        self.RABBIT_VALUE = 500

        # Material values
        self.PIECE_VALUES = {
            PieceType.ELEPHANT: 1000,
            PieceType.CAMEL: 700,
            PieceType.HORSE: 500,
            PieceType.DOG: 400,
            PieceType.CAT: 300,
            PieceType.RABBIT: 500  # High value!
        }

        # Positional
        self.ADVANCEMENT_VALUE = 10
        self.TRAP_CONTROL = 50

    def score(self, board: ArimaaBoard, color: Color) -> float:
        """
        Calculate differential score for a position.

        Returns: my_score - opponent_score
        """
        my_score = self._evaluate_color(board, color)
        opponent_score = self._evaluate_color(board, color.opposite())

        return my_score - opponent_score

    def _evaluate_color(self, board: ArimaaBoard, color: Color) -> float:
        """Evaluate position strength for one color"""
        score = 0.0

        # Check for rabbit on goal (win)
        if board.has_rabbit_on_goal(color):
            return self.RABBIT_ON_GOAL

        pieces = board.get_pieces(color)

        for piece in pieces:
            # Material value
            score += self.PIECE_VALUES.get(piece.type, 0)

            # Rabbit advancement
            if piece.type == PieceType.RABBIT:
                row = piece.position[0]
                if color == Color.GOLD:
                    # Gold advances up (row 0→7)
                    advancement = row
                else:
                    # Silver advances down (row 7→0)
                    advancement = 7 - row
                score += advancement * self.RABBIT_ADVANCEMENT

            # General advancement
            row = piece.position[0]
            if color == Color.GOLD:
                score += row * self.ADVANCEMENT_VALUE
            else:
                score += (7 - row) * self.ADVANCEMENT_VALUE

            # Trap control (being adjacent to trap)
            for trap_pos in board.TRAP_SQUARES:
                if self._is_adjacent(piece.position, trap_pos):
                    score += self.TRAP_CONTROL

        return score

    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent (4-directional)"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def score_game(self, game: ArimaaGame, color: Color) -> float:
        """Score a complete game state"""
        return self.score(game.board, color)

    def get_move_category(self, game_before: ArimaaGame,
                          game_after: ArimaaGame,
                          step: ArimaaStep,
                          color: Color) -> str:
        """
        Categorize a step for pattern learning.

        Categories:
        - 'rabbit_goal' - Rabbit reaches goal
        - 'rabbit_advance' - Rabbit moves forward
        - 'capture' - Traps enemy piece
        - 'push' - Pushes enemy piece
        - 'pull' - Pulls enemy piece
        - 'trap_control' - Moves near trap
        - 'advance' - General forward movement
        - 'quiet' - Normal move
        """
        # Check if wins
        if game_after.board.has_rabbit_on_goal(color):
            return 'rabbit_goal'

        # Check step type
        if step.step_type == StepType.PUSH:
            return 'push'
        elif step.step_type == StepType.PULL:
            return 'pull'

        # Check if captured piece via trap
        pieces_before = len(game_before.board.get_pieces(color.opposite()))
        pieces_after = len(game_after.board.get_pieces(color.opposite()))
        if pieces_after < pieces_before:
            return 'capture'

        # Check if moved piece
        moved_piece = game_before.board.get_piece(step.from_pos)
        if moved_piece is None:
            return 'quiet'

        # Check if rabbit advanced
        if moved_piece.type == PieceType.RABBIT:
            from_row = step.from_pos[0]
            to_row = step.to_pos[0]
            if color == Color.GOLD and to_row > from_row:
                return 'rabbit_advance'
            elif color == Color.SILVER and to_row < from_row:
                return 'rabbit_advance'

        # Check trap control
        for trap_pos in game_after.board.TRAP_SQUARES:
            if self._is_adjacent(step.to_pos, trap_pos):
                return 'trap_control'

        # Check general advancement
        from_row = step.from_pos[0]
        to_row = step.to_pos[0]
        if color == Color.GOLD and to_row > from_row:
            return 'advance'
        elif color == Color.SILVER and to_row < from_row:
            return 'advance'

        return 'quiet'

    def get_distance_metric(self, step: ArimaaStep, board: ArimaaBoard) -> int:
        """
        Get distance metric for pattern learning.

        Use Manhattan distance of the step.
        """
        from_row, from_col = step.from_pos
        to_row, to_col = step.to_pos

        distance = abs(to_row - from_row) + abs(to_col - from_col)

        # For push/pull, add extra distance
        if step.step_type in [StepType.PUSH, StepType.PULL]:
            distance += 1

        return min(distance, 5)
