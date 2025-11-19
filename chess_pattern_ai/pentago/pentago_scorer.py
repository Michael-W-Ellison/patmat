"""
Pentago differential scoring system.

Scoring philosophy:
    score = my_advantage - opponent_advantage

Components:
1. Threat-based scoring (4-in-row, 3-in-row, etc.)
2. Position control (center positions more valuable)
3. Rotation potential (marbles in key positions)

The AI learns which patterns lead to wins through observation.
"""

from .pentago_board import PentagoBoard, Color
from typing import Optional


class PentagoScorer:
    """
    Differential scorer for Pentago.

    Evaluates position strength based on:
    - 5-in-a-row (win)
    - 4-in-a-row threats
    - 3-in-a-row threats
    - 2-in-a-row threats
    - Center control
    """

    def __init__(self):
        """Initialize scorer with default weights"""
        # Threat weights
        self.THREAT_5 = 10000  # Win
        self.THREAT_4 = 500    # Very strong
        self.THREAT_3 = 100    # Strong
        self.THREAT_2 = 20     # Moderate

        # Position weights
        self.CENTER_VALUE = 10      # Center positions (2,2), (2,3), (3,2), (3,3)
        self.INNER_VALUE = 5        # Inner ring
        self.CORNER_VALUE = 3       # Corners
        self.EDGE_VALUE = 2         # Edges

    def score(self, board: PentagoBoard, color: Color) -> float:
        """
        Calculate differential score for a position.

        Returns: my_score - opponent_score
        """
        my_score = self._evaluate_color(board, color)
        opponent_score = self._evaluate_color(board, color.opposite())

        return my_score - opponent_score

    def _evaluate_color(self, board: PentagoBoard, color: Color) -> float:
        """Evaluate position strength for one color"""
        score = 0.0

        # Check for win
        if board.check_five_in_row(color):
            return self.THREAT_5

        # Count threats of different lengths
        threats_4 = board.count_threats(color, 4)
        threats_3 = board.count_threats(color, 3)
        threats_2 = board.count_threats(color, 2)

        score += threats_4 * self.THREAT_4
        score += threats_3 * self.THREAT_3
        score += threats_2 * self.THREAT_2

        # Position value
        marbles = board.get_marbles(color)
        for marble in marbles:
            score += self._position_value(marble.position)

        return score

    def _position_value(self, pos: tuple) -> float:
        """Get value of a position based on centrality"""
        row, col = pos

        # Center 4 positions (most valuable)
        if (row, col) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
            return self.CENTER_VALUE

        # Inner ring (good)
        if 1 <= row <= 4 and 1 <= col <= 4:
            return self.INNER_VALUE

        # Corners (moderate)
        if (row, col) in [(0, 0), (0, 5), (5, 0), (5, 5)]:
            return self.CORNER_VALUE

        # Edges (less valuable)
        return self.EDGE_VALUE

    def get_move_category(self, board_before: PentagoBoard,
                          board_after: PentagoBoard,
                          move,
                          color: Color) -> str:
        """
        Categorize a move for pattern learning.

        Categories:
        - 'winning' - Creates 5 in a row
        - 'threat_4' - Creates 4 in a row
        - 'threat_3' - Creates 3 in a row
        - 'block_4' - Blocks opponent's 4 in a row
        - 'rotation_trap' - Rotation creates/enhances threat
        - 'center' - Places in center area
        - 'quiet' - Normal positional move
        """
        # Check if move wins
        if board_after.check_five_in_row(color):
            return 'winning'

        # Count threats before and after
        my_4_before = board_before.count_threats(color, 4)
        my_4_after = board_after.count_threats(color, 4)

        my_3_before = board_before.count_threats(color, 3)
        my_3_after = board_after.count_threats(color, 3)

        opp_4_before = board_before.count_threats(color.opposite(), 4)
        opp_4_after = board_after.count_threats(color.opposite(), 4)

        # Created a 4-in-row threat
        if my_4_after > my_4_before:
            return 'threat_4'

        # Blocked opponent's 4-in-row
        if opp_4_after < opp_4_before:
            return 'block_4'

        # Created a 3-in-row threat
        if my_3_after > my_3_before:
            return 'threat_3'

        # Check if rotation was key (changed threat count significantly)
        # This happens when rotation aligns marbles
        placement_pos = move.placement
        placement_row, placement_col = placement_pos

        # If placed marble was rotated to different position
        marble_after = board_after.get_marble(placement_pos)
        if marble_after is None or marble_after.position != placement_pos:
            # Rotation moved the placed marble → rotation trap
            return 'rotation_trap'

        # Center placement
        if placement_pos in [(2, 2), (2, 3), (3, 2), (3, 3)]:
            return 'center'

        # Default: quiet move
        return 'quiet'

    def get_distance_metric(self, move, board: PentagoBoard) -> int:
        """
        Get distance metric for pattern learning.

        For Pentago, use Manhattan distance from center.
        Center positions = distance 0
        Edges = distance 4-5
        """
        row, col = move.placement

        # Distance from center point (2.5, 2.5)
        center_row, center_col = 2.5, 2.5
        distance = abs(row - center_row) + abs(col - center_col)

        return int(distance)
