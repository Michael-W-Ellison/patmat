#!/usr/bin/env python3
"""Hex adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .hex_board import Color, HexBoard
from .hex_game import HexGame
from .hex_scorer import HexScorer


class HexAdapter:
    """
    State is a HexGame; color is Color.RED / Color.BLUE

    Feature extraction evaluates hundreds of positions per move with the
    scorer's flood fills, so the boards this adapter creates (never changed
    after apply) get exact speed-ups of the board's own methods:
    - get_neighbors: precomputed table (same lists)
    - is_connected_to_start: memoized per color, and start cells already in a
      found group are not flood-filled again (same set of stones)
    - get_connection_strength: same breadth-first search, stopping at the
      first goal cell reached (the minimum the full search returns), and
      cached by the start-connected positions it depends on
    Values are the same as the game's own methods.
    """

    def __init__(self):
        self.scorer = HexScorer()
        self._neighbor_tables = {}
        self._distances = {}   # (size, color, start-connected positions) -> distance

    def _neighbors(self, size):
        table = self._neighbor_tables.get(size)
        if table is None:
            plain = HexBoard(size)
            table = {(r, c): plain.get_neighbors((r, c))
                     for r in range(size) for c in range(size)}
            self._neighbor_tables[size] = table
        return table

    @staticmethod
    def _distance_to_goal(neighbors, visited, axis, goal):
        """HexBoard.get_connection_strength's search, stopping at the first goal cell"""
        frontier = []
        for pos in visited.copy():
            for neighbor_pos in neighbors[pos]:
                if neighbor_pos not in visited:
                    if neighbor_pos[axis] == goal:
                        return 0
                    visited.add(neighbor_pos)
                    frontier.append(neighbor_pos)
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for pos in frontier:
                for neighbor_pos in neighbors[pos]:
                    if neighbor_pos not in visited:
                        if neighbor_pos[axis] == goal:
                            return dist
                        visited.add(neighbor_pos)
                        next_frontier.append(neighbor_pos)
            frontier = next_frontier
        return float('inf')

    def legal_moves(self, game, color):
        return game.get_legal_moves(color)

    def apply(self, game, move):
        board = game.board
        new = HexBoard.__new__(HexBoard)
        new.size = board.size
        new.board = [row[:] for row in board.board]   # Stones are never mutated
        new.stones = set(board.stones)
        new.turn = board.turn
        new.move_history = []                        # board.copy() does not keep it either
        new.make_move(move)

        self._speed_up(new)
        return HexGame(new)

    def _speed_up(self, board):
        size = board.size
        neighbors = self._neighbors(size)
        board.get_neighbors = neighbors.__getitem__
        memo = {}

        def is_connected_to_start(color):
            connected = memo.get(color)
            if connected is None:
                if color == Color.RED:
                    start_positions = [(0, col) for col in range(size)]
                else:
                    start_positions = [(row, 0) for row in range(size)]
                connected = set()
                found = set()
                for pos in start_positions:
                    if pos not in found:
                        group = board.get_connected_group(pos)
                        connected.update(group)
                        found.update(stone.position for stone in group)
                memo[color] = connected
            return connected

        distances = self._distances

        def get_connection_strength(color):
            connected = is_connected_to_start(color)
            if not connected:
                return (0, float('inf'))
            # The search ignores stones, so the distance depends only on these positions
            visited = frozenset(stone.position for stone in connected)
            key = (size, color, visited)
            dist = distances.get(key)
            if dist is None:
                if len(distances) > 200000:
                    distances.clear()
                dist = distances[key] = self._distance_to_goal(
                    neighbors, set(visited), 0 if color == Color.RED else 1, size - 1)
            return (len(connected), dist)

        board.is_connected_to_start = is_connected_to_start
        board.get_connection_strength = get_connection_strength

    def advantage(self, game, color):
        """Connection strength difference (scorer's group size + distance-to-goal bonus)"""
        return self.scorer.evaluate_position(game.board, color)

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        # Players strictly alternate; the board switches turn after every move
        return game.board.turn

    def opponent(self, color):
        return Color.BLUE if color == Color.RED else Color.RED
