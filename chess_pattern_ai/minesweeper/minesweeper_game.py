#!/usr/bin/env python3
"""
Minesweeper rules

The board has rows x cols cells and a number of mines. A move reveals one
hidden cell: a mine loses the game; a safe cell shows how many of its 8
neighbours are mines, and a cell showing 0 also reveals its neighbours
(repeatedly, so a whole empty region opens). The game is won when every
safe cell is revealed.

Mines are placed when the first cell is revealed, never on that cell or
its neighbours, so the first move always opens a region. The placement is
drawn from the game's own random generator, so a seed repeats a game.
"""

import random
from typing import List, Optional, Set, Tuple

Pos = Tuple[int, int]
HIDDEN = -1        # value of a hidden cell in an observation
MINE = -2          # value of a mine in the view shown after the game (or of a filled cell shown as given)
SHOWN_EMPTY = -3   # a board cell shown empty, without a number (used by other constraint puzzles)
OUTSIDE = -4       # a cell outside the board, such as an unused margin cell (other constraint puzzles)


class MinesweeperGame:
    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10, seed: Optional[int] = None):
        if mines > rows * cols - 9:
            raise ValueError('too many mines to keep the first move and its neighbours clear')
        self.rows, self.cols, self.n_mines = rows, cols, mines
        self.rng = random.Random(seed)
        self.mines: Set[Pos] = set()          # hidden from the player
        self.revealed: Set[Pos] = set()
        self.exploded: Optional[Pos] = None
        self.moves = 0

    def copy(self) -> 'MinesweeperGame':
        g = MinesweeperGame.__new__(MinesweeperGame)
        g.rows, g.cols, g.n_mines = self.rows, self.cols, self.n_mines
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        g.mines, g.revealed = set(self.mines), set(self.revealed)
        g.exploded, g.moves = self.exploded, self.moves
        return g

    def neighbours(self, pos: Pos) -> List[Pos]:
        r, c = pos
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]

    def count(self, pos: Pos) -> int:
        """Mines among the cell's neighbours (the number a revealed cell shows)"""
        return sum(1 for n in self.neighbours(pos) if n in self.mines)

    def _place_mines(self, first: Pos):
        clear = {first, *self.neighbours(first)}
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in clear]
        self.mines = set(self.rng.sample(cells, self.n_mines))

    def legal_moves(self) -> List[Pos]:
        if self.is_over():
            return []
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.revealed]

    def reveal(self, pos: Pos):
        if self.is_over() or pos in self.revealed:
            raise ValueError(f'cannot reveal {pos}')
        if not self.moves:
            self._place_mines(pos)
        self.moves += 1
        if pos in self.mines:
            self.exploded = pos
            return
        stack = [pos]
        while stack:
            p = stack.pop()
            if p in self.revealed:
                continue
            self.revealed.add(p)
            if self.count(p) == 0:
                stack.extend(n for n in self.neighbours(p) if n not in self.revealed)

    def lost(self) -> bool:
        return self.exploded is not None

    def won(self) -> bool:
        return not self.lost() and len(self.revealed) == self.rows * self.cols - self.n_mines

    def is_over(self) -> bool:
        return self.lost() or self.won()

    def final_view(self) -> List[List[int]]:
        """After the game: the observation with every mine shown as MINE (as the game shows at the end)"""
        if not self.is_over():
            raise ValueError('the mines are shown only when the game is over')
        obs = self.observation()
        for r, c in self.mines:
            obs[r][c] = MINE
        return obs

    def observation(self) -> List[List[int]]:
        """What the player sees: HIDDEN for hidden cells, the neighbour count for revealed ones"""
        return [[self.count((r, c)) if (r, c) in self.revealed else HIDDEN for c in range(self.cols)]
                for r in range(self.rows)]
