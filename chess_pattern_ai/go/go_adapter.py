#!/usr/bin/env python3
"""Go adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .go_board import Color, GoBoard, Move, Stone
from .go_game import GoGame
from .go_scorer import GoScorer


class GoAdapter:
    """
    State is a GoGame; color is Color.BLACK / Color.WHITE.

    Moves are the game's own legal moves, including the pass (the trainer's
    move list contains it too); two consecutive passes end the game.
    """

    def __init__(self):
        self.scorer = GoScorer()

    @staticmethod
    def _copy_board(board):
        """
        Same position as GoBoard.copy(), but cheaper: stones are immutable, so
        the copy shares them and only copies the grid rows and the stone set
        (GoBoard.copy() also starts with an empty move history)
        """
        new_board = GoBoard.__new__(GoBoard)
        new_board.size = board.size
        new_board.board = [row[:] for row in board.board]
        new_board.stones = set(board.stones)
        new_board.turn = board.turn
        new_board.move_history = []
        new_board.consecutive_passes = board.consecutive_passes
        new_board.captured_by_black = board.captured_by_black
        new_board.captured_by_white = board.captured_by_white
        return new_board

    def legal_moves(self, game, color):
        """
        Same moves, in the same order, as GoGame.get_legal_moves(), which tests
        every empty point on its own board copy; here one scratch copy is used,
        placing and removing the test stone at each point
        """
        test_board = self._copy_board(game.board)
        grid = test_board.board
        neighbors = self._neighbors(test_board.size)
        moves = []
        for row in range(test_board.size):
            for col in range(test_board.size):
                if grid[row][col] is not None:
                    continue
                pos = (row, col)
                # An empty neighbor is a liberty of the new stone's group, so the
                # game's test would pass; only otherwise run the full test
                if not any(grid[r][c] is None for r, c in neighbors[pos]):
                    grid[row][col] = Stone(color, pos)
                    legal = (bool(test_board.get_captured_groups(pos))
                             or test_board.get_liberties(test_board.get_group(pos)) > 0)
                    grid[row][col] = None
                    if not legal:
                        continue
                moves.append(Move(color, pos))
        moves.append(Move(color, None))  # Pass is always legal
        return moves

    def apply(self, game, move):
        after = GoGame(self._copy_board(game.board))
        # A fresh Move: GoBoard.make_move stores the captured stones on the move,
        # and the trainer reads move.captured of its own move objects
        after.make_move(Move(move.color, move.position))
        return after

    _neighbor_tables = {}

    @classmethod
    def _neighbors(cls, size):
        """Orthogonal neighbors of every point (as GoBoard.get_neighbors), cached per size"""
        table = cls._neighbor_tables.get(size)
        if table is None:
            table = {}
            for row in range(size):
                for col in range(size):
                    table[(row, col)] = [(row + dr, col + dc)
                                         for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                         if 0 <= row + dr < size and 0 <= col + dc < size]
            cls._neighbor_tables[size] = table
        return table

    def _territories(self, board):
        """
        GoBoard.count_territory() for both colors in one pass: an empty region
        bordered by stones of only one color is that color's territory
        """
        grid = board.board
        neighbors = self._neighbors(board.size)
        territory = {Color.BLACK: 0, Color.WHITE: 0}
        visited = set()
        for row in range(board.size):
            for col in range(board.size):
                start = (row, col)
                if grid[row][col] is not None or start in visited:
                    continue
                visited.add(start)
                stack = [start]
                size = 0
                borders = set()
                while stack:
                    pos = stack.pop()
                    size += 1
                    for r, c in neighbors[pos]:
                        stone = grid[r][c]
                        if stone is not None:
                            borders.add(stone.color)
                        elif (r, c) not in visited:
                            visited.add((r, c))
                            stack.append((r, c))
                if len(borders) == 1:
                    territory[borders.pop()] += size
        return territory

    def advantage(self, game, color):
        """
        Scorer's material (territory + captures, GoScorer.evaluate_position),
        color minus opponent; territory is counted for both colors in one pass
        """
        board = game.board
        territory = self._territories(board)
        black = territory[Color.BLACK] + board.captured_by_black
        white = territory[Color.WHITE] + board.captured_by_white
        return black - white if color == Color.BLACK else white - black

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.board.turn

    def opponent(self, color):
        return Color.WHITE if color == Color.BLACK else Color.BLACK
