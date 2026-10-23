#!/usr/bin/env python3
"""Dots and Boxes adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .dots_boxes_board import Color
from .dots_boxes_game import DotsBoxesGame
from .dots_boxes_scorer import DotsBoxesScorer
from grid_vocabulary import Grid


class DotsBoxesAdapter:
    """State is a DotsBoxesGame; color is Color.RED / Color.BLUE"""

    def __init__(self):
        self.scorer = DotsBoxesScorer()

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        after = DotsBoxesGame(game.board.copy())
        after.current_player = game.current_player
        after.last_move_completed_box = game.last_move_completed_box
        after.make_move(move)
        return after

    # Cell values of to_grid
    WALL = 1
    BOX_OWNER = {Color.RED: 2, Color.BLUE: 3}

    def grid_values(self, color):
        """Values that to_grid uses for boxes owned by color"""
        return {2} if color == Color.RED else {3}

    def to_grid(self, game) -> Grid:
        """
        The board as a (2*BOXES+1) x (2*BOXES+1) lattice Grid (9 x 9 for the
        5x5-dot board, topology 'square'), row 0 = the board's dot row 0:

        - dot (r, c)                            -> cell (2r, 2c):      1 (wall)
        - horizontal edge (r, c)-(r, c+1)       -> cell (2r, 2c+1):    1 if drawn, else 0
        - vertical edge (r, c)-(r+1, c)         -> cell (2r+1, 2c):    1 if drawn, else 0
        - box (r, c) (top-left dot (r, c))      -> cell (2r+1, 2c+1):  0 if unowned,
          2 if owned by Color.RED, 3 if owned by Color.BLUE

        Drawn edges and dots share the wall value 1, so a completed box is a
        single box-centre cell enclosed by walls (enclosed(grid, walls=[1])).
        The game is not changed.
        """
        board = game.board
        size = 2 * board.BOXES + 1
        cells = [[0] * size for _ in range(size)]
        for r in range(board.DOTS):
            for c in range(board.DOTS):
                cells[2 * r][2 * c] = self.WALL
        for edge in board.edges:
            (r1, c1), (r2, c2) = edge.dot1, edge.dot2
            cells[r1 + r2][c1 + c2] = self.WALL     # midpoint of the two dots
        for (r, c), owner in board.boxes.items():
            if owner is not None:
                cells[2 * r + 1][2 * c + 1] = self.BOX_OWNER[owner]
        return Grid(cells, topology='square')

    def advantage(self, game, color):
        """Box advantage from the scorer (boxes and 3-edge boxes, color minus opponent)"""
        return self.scorer.calculate_box_advantage(game.board, color)

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        # Completing a box keeps the turn (DotsBoxesGame.make_move does not switch)
        return game.current_player

    def opponent(self, color):
        return Color.BLUE if color == Color.RED else Color.RED
