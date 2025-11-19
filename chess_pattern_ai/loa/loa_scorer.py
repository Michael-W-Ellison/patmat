"""
Lines of Action differential scoring system.

Scoring philosophy:
    score = my_advantage - opponent_advantage

Components:
1. Connectivity (fewer groups = better)
2. Material count (pieces on board)
3. Mobility (number of legal moves)
4. Centralization (pieces near center)

The AI learns which patterns lead to wins through observation.
"""

from .loa_board import LOABoard, Color
from .loa_game import LOAGame
from typing import Tuple


class LOAScorer:
    """
    Differential scorer for Lines of Action.

    Evaluates position strength based on:
    - Group count (fewer = better, 1 group = win)
    - Material count
    - Mobility
    - Centralization
    """

    def __init__(self):
        """Initialize scorer with default weights"""
        # Connectivity is most important
        self.GROUP_PENALTY = -500  # Penalty per group (fewer groups = better)
        self.CONNECTED_BONUS = 10000  # Bonus for being fully connected

        # Material
        self.PIECE_VALUE = 100

        # Mobility
        self.MOBILITY_VALUE = 10

        # Centralization (being near center)
        self.CENTER_VALUE = 15

    def score(self, board: LOABoard, color: Color) -> float:
        """
        Calculate differential score for a position.

        Returns: my_score - opponent_score
        """
        my_score = self._evaluate_color(board, color)
        opponent_score = self._evaluate_color(board, color.opposite())

        return my_score - opponent_score

    def _evaluate_color(self, board: LOABoard, color: Color) -> float:
        """Evaluate position strength for one color"""
        score = 0.0

        # Check if connected (win condition)
        if board.is_connected(color):
            return self.CONNECTED_BONUS

        # Count groups (fewer is better)
        groups = board.count_groups(color)
        score += groups * self.GROUP_PENALTY

        # Material count
        pieces = board.get_pieces(color)
        score += len(pieces) * self.PIECE_VALUE

        # Centralization
        for piece in pieces:
            score += self._position_value(piece.position)

        return score

    def score_game(self, game: LOAGame, color: Color) -> float:
        """Score a complete game state"""
        return self.score(game.board, color)

    def _position_value(self, pos: Tuple[int, int]) -> float:
        """Get value of a position based on centrality"""
        row, col = pos
        # Distance from center (3.5, 3.5)
        center_row, center_col = 3.5, 3.5
        distance = abs(row - center_row) + abs(col - center_col)

        # Closer to center = higher value
        return self.CENTER_VALUE * (7 - distance) / 7

    def get_move_category(self, game_before: LOAGame,
                          game_after: LOAGame,
                          move,
                          color: Color) -> str:
        """
        Categorize a move for pattern learning.

        Categories:
        - 'winning' - Connects all pieces
        - 'grouping' - Reduces number of groups
        - 'capture' - Captures opponent piece
        - 'centralizing' - Moves toward center
        - 'connecting' - Connects to another piece
        - 'mobility' - Increases legal moves
        - 'quiet' - Normal move
        """
        # Check if move wins
        if game_after.board.is_connected(color):
            return 'winning'

        # Count groups before and after
        groups_before = game_before.board.count_groups(color)
        groups_after = game_after.board.count_groups(color)

        # Reduced groups
        if groups_after < groups_before:
            return 'grouping'

        # Check if capture
        from_piece = game_before.board.get_piece(move.from_pos)
        to_piece = game_before.board.get_piece(move.to_pos)
        if to_piece is not None and to_piece.color != color:
            return 'capture'

        # Check if connecting to another piece (adjacent)
        to_row, to_col = move.to_pos
        has_adjacent = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_pos = (to_row + dr, to_col + dc)
                if not game_after.board._is_valid_position(adj_pos):
                    continue
                adj_piece = game_after.board.get_piece(adj_pos)
                if adj_piece is not None and adj_piece.color == color:
                    has_adjacent = True
                    break
            if has_adjacent:
                break

        if has_adjacent and from_piece is not None:
            # Check if wasn't adjacent before
            from_row, from_col = move.from_pos
            was_adjacent = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_pos = (from_row + dr, from_col + dc)
                    if not game_before.board._is_valid_position(adj_pos):
                        continue
                    adj_piece = game_before.board.get_piece(adj_pos)
                    if adj_piece is not None and adj_piece.color == color:
                        was_adjacent = True
                        break
                if was_adjacent:
                    break

            if not was_adjacent and has_adjacent:
                return 'connecting'

        # Check if centralizing
        from_val = self._position_value(move.from_pos)
        to_val = self._position_value(move.to_pos)
        if to_val > from_val + 2:  # Significant centralization
            return 'centralizing'

        # Default: quiet move
        return 'quiet'

    def get_distance_metric(self, move, board: LOABoard) -> int:
        """
        Get distance metric for pattern learning.

        For LOA, use Manhattan distance of the move.
        """
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos

        distance = abs(to_row - from_row) + abs(to_col - from_col)

        # Clamp to reasonable range
        return min(distance, 10)
