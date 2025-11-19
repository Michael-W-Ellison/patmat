"""
Nine Men's Morris differential scoring system.

Scoring philosophy:
    score = my_advantage - opponent_advantage

Components:
1. Material count (pieces on board)
2. Mills formed (3 in a row)
3. Potential mills (2 in a row with empty third position)
4. Mobility (number of legal moves)
5. Phase-specific bonuses

The AI learns which patterns lead to wins through observation.
"""

from .morris_board import MorrisBoard, Color
from .morris_game import MorrisGame, GamePhase
from typing import Union


class MorrisScorer:
    """
    Differential scorer for Nine Men's Morris.

    Evaluates position strength based on:
    - Piece count (material)
    - Complete mills
    - Potential mills (2 pieces + empty)
    - Blocked opponent mills
    - Mobility (legal moves available)
    """

    def __init__(self):
        """Initialize scorer with default weights"""
        # Material
        self.PIECE_VALUE = 100

        # Mills
        self.COMPLETE_MILL = 500
        self.TWO_PIECE_MILL = 80   # Has 2 pieces, needs 1 more
        self.ONE_PIECE_MILL = 20   # Has 1 piece, needs 2 more

        # Mobility
        self.MOBILITY_VALUE = 10

        # Phase bonuses
        self.PLACEMENT_BONUS = 5   # Bonus per piece in placement phase
        self.FLYING_BONUS = 50     # Bonus for having flying ability

    def score(self, board: MorrisBoard, color: Color, phase: GamePhase,
              pieces_placed_white: int = 9, pieces_placed_black: int = 9) -> float:
        """
        Calculate differential score for a position.

        Returns: my_score - opponent_score
        """
        my_score = self._evaluate_color(board, color, phase, pieces_placed_white, pieces_placed_black)
        opponent_score = self._evaluate_color(board, color.opposite(), phase,
                                              pieces_placed_white, pieces_placed_black)

        return my_score - opponent_score

    def _evaluate_color(self, board: MorrisBoard, color: Color, phase: GamePhase,
                        pieces_placed_white: int, pieces_placed_black: int) -> float:
        """Evaluate position strength for one color"""
        score = 0.0

        # Material count
        piece_count = board.count_pieces(color)
        score += piece_count * self.PIECE_VALUE

        # Complete mills
        mills = board.count_mills(color)
        score += mills * self.COMPLETE_MILL

        # Potential mills (2 pieces, need 1 more)
        potential_2 = board.count_potential_mills(color, pieces_needed=1)
        score += potential_2 * self.TWO_PIECE_MILL

        # Weak potential mills (1 piece, need 2 more)
        potential_1 = board.count_potential_mills(color, pieces_needed=2)
        score += potential_1 * self.ONE_PIECE_MILL

        # Placement phase bonus (more pieces placed = better)
        if phase == GamePhase.PLACEMENT:
            pieces_placed = pieces_placed_white if color == Color.WHITE else pieces_placed_black
            score += pieces_placed * self.PLACEMENT_BONUS

        # Flying bonus (if player has exactly 3 pieces → can fly)
        if piece_count == 3:
            score += self.FLYING_BONUS

        return score

    def score_game(self, game: MorrisGame, color: Color) -> float:
        """Score a complete game state"""
        return self.score(game.board, color, game.phase,
                         game.white_pieces_placed, game.black_pieces_placed)

    def get_move_category(self, game_before: MorrisGame,
                          game_after: MorrisGame,
                          move,
                          color: Color) -> str:
        """
        Categorize a move for pattern learning.

        Categories:
        - 'form_mill' - Creates a mill
        - 'remove_piece' - Removes opponent piece (after mill)
        - 'block_mill' - Blocks opponent's potential mill
        - 'create_2mill' - Creates 2-in-row potential
        - 'mobility' - Increases mobility
        - 'placement' - Placement phase move
        - 'movement' - Movement phase move
        - 'flying' - Flying phase move
        - 'quiet' - Normal positional move
        """
        # Check if move is a removal
        if isinstance(move, int):
            return 'remove_piece'

        # Check if mill was formed
        if game_after.pending_removal:
            return 'form_mill'

        # Get position where piece ended up
        if move.is_placement:
            end_pos = move.position
            is_placement = True
        else:
            end_pos = move.to_pos
            is_placement = False

        # Count potential mills before and after
        my_2mills_before = game_before.board.count_potential_mills(color, pieces_needed=1)
        my_2mills_after = game_after.board.count_potential_mills(color, pieces_needed=1)

        opp_2mills_before = game_before.board.count_potential_mills(color.opposite(), pieces_needed=1)
        opp_2mills_after = game_after.board.count_potential_mills(color.opposite(), pieces_needed=1)

        # Created a 2-piece potential mill
        if my_2mills_after > my_2mills_before:
            return 'create_2mill'

        # Blocked opponent's potential mill
        if opp_2mills_after < opp_2mills_before:
            return 'block_mill'

        # Phase-specific categories
        if is_placement:
            return 'placement'
        elif game_before.phase == GamePhase.FLYING:
            return 'flying'
        elif game_before.phase == GamePhase.MOVEMENT:
            return 'movement'

        # Default: quiet move
        return 'quiet'

    def get_distance_metric(self, move, game: MorrisGame) -> int:
        """
        Get distance metric for pattern learning.

        For Morris, use:
        - Placement phase: position number (0-23)
        - Movement/Flying: Manhattan distance to board center (pos 17)
        """
        if isinstance(move, int):
            # Removal move - use removed piece position
            return move % 8  # Group by ring (0-7, 8-15, 16-23)

        if move.is_placement:
            pos = move.position
        else:
            pos = move.to_pos

        # Distance to center of board (position 17 is center of inner ring)
        # Map positions to rings
        if pos < 8:
            ring = 0  # Outer
        elif pos < 16:
            ring = 1  # Middle
        else:
            ring = 2  # Inner

        return ring
