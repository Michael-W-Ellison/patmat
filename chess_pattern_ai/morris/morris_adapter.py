#!/usr/bin/env python3
"""Nine Men's Morris adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .morris_board import Color, MorrisBoard
from .morris_scorer import MorrisScorer


# Point index -> (row, col) in the standard 7x7 drawing of the board (outer,
# middle and inner squares, clockwise from the top-left corner; see the layout
# in morris_board.py)
POINT_CELLS = (
    (0, 0), (0, 3), (0, 6), (3, 6), (6, 6), (6, 3), (6, 0), (3, 0),
    (1, 1), (1, 3), (1, 5), (3, 5), (5, 5), (5, 3), (5, 1), (3, 1),
    (2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (3, 2),
)
_VALID = [[(r, c) in POINT_CELLS for c in range(7)] for r in range(7)]
_ADJACENCY = {POINT_CELLS[p]: [POINT_CELLS[q] for q in nbs]
              for p, nbs in MorrisBoard.ADJACENCIES.items()}
_LINE_SETS = [[POINT_CELLS[p] for p in mill] for mill in MorrisBoard.MILLS]


class MorrisAdapter:
    """
    State is a MorrisGame; color is Color.WHITE / Color.BLACK.

    Moves are the game's own legal moves for the current phase: placements,
    adjacent moves or flying moves, and (as integers) the removal of an
    opponent piece after a mill. Forming a mill keeps the turn for that
    removal, so such moves are seen as giving an extra turn.
    """

    def __init__(self):
        self.scorer = MorrisScorer()

    def legal_moves(self, game, color):
        # MorrisGame generates the moves of the player to move (color is that player)
        return game.get_legal_moves()

    def apply(self, game, move):
        # MorrisGame.make_move works on a copy and returns the new state
        return game.make_move(move)

    def advantage(self, game, color):
        """Scorer's evaluation (material, mills, potential mills, phase bonuses), color minus opponent"""
        return self.scorer.score_game(game, color)

    def grid_values(self, color):
        """Values that to_grid uses for color's pieces"""
        return {1} if color == Color.WHITE else {2}

    def to_grid(self, game):
        """
        'graph' Grid: the 24 points embedded in the standard 7x7 layout
        (POINT_CELLS; valid only at those points), adjacency = the board's
        ADJACENCIES, line_sets = the board's MILLS (each an ordered 3-point
        line). Empty 0; white 1, black 2.
        """
        cells = [[0] * 7 for _ in range(7)]
        for p, piece in enumerate(game.board.positions):
            if piece is not None:
                r, c = POINT_CELLS[p]
                cells[r][c] = 1 if piece.color == Color.WHITE else 2
        return Grid(cells, 'graph', valid=_VALID,
                    adjacency=_ADJACENCY, line_sets=_LINE_SETS)

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.current_player

    def opponent(self, color):
        return color.opposite()
