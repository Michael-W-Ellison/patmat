#!/usr/bin/env python3
"""
Constraint puzzles in the view format of the Minesweeper constraint learner

Each puzzle is a grid of filled and empty cells. Clue cells show a number
from the start; the player uncovers cells it believes are empty, and
uncovering a filled cell ends the attempt. The puzzle is solved when every
empty playable cell is uncovered. After the attempt the filled cells are
shown, as the mines are after a game of Minesweeper.

Views use the learner's values: a number >= 0 for a clue, HIDDEN for a
cell not yet uncovered, EMPTY for a cell shown without a number (an
uncovered empty cell, an unused margin cell), and in the final view MINE
for a filled cell.

Puzzles (each generated from a seed):
- orthogonal:  clue = filled cells among the clue cell's 4 orthogonal
               neighbours (a static Minesweeper variant)
- mosaic:      clue = filled cells in the 3x3 block including the clue cell
               itself, which may be filled (Fill-a-Pix)
- totals:      margin clues = filled cells in their row or column
- nonogram:    margin clues = lengths of the runs of filled cells in their
               row or column, in order
- placement:   one filled cell in every row, column and 3x3 box of a 9x9
               grid (the cells of one Sudoku digit); no clues are shown
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from minesweeper.minesweeper_game import HIDDEN, MINE

Pos = Tuple[int, int]
EMPTY = -3


class BinaryPuzzle:
    def __init__(self, rows: int, cols: int, filled: Set[Pos], clues: Dict[Pos, int], playable: Set[Pos]):
        self.rows, self.cols = rows, cols
        self.filled, self.clues, self.playable = set(filled), dict(clues), set(playable)
        self.revealed: Set[Pos] = set()
        self.exploded: Optional[Pos] = None

    def legal_moves(self) -> List[Pos]:
        return [] if self.is_over() else sorted(self.playable - self.revealed)

    def reveal(self, pos: Pos):
        if self.is_over() or pos not in self.playable or pos in self.revealed:
            raise ValueError(f'cannot uncover {pos}')
        if pos in self.filled:
            self.exploded = pos
        else:
            self.revealed.add(pos)

    def lost(self) -> bool:
        return self.exploded is not None

    def won(self) -> bool:
        return not self.lost() and self.revealed == self.playable - self.filled

    def is_over(self) -> bool:
        return self.lost() or self.won()

    def progress(self) -> float:
        """Share of the empty playable cells uncovered"""
        empty = self.playable - self.filled
        return len(self.revealed) / len(empty) if empty else 1.0

    def observation(self) -> List[List[int]]:
        view = [[EMPTY] * self.cols for _ in range(self.rows)]
        for (r, c), v in self.clues.items():
            view[r][c] = v
        for r, c in self.playable - self.revealed:
            view[r][c] = HIDDEN
        return view

    def final_view(self) -> List[List[int]]:
        if not self.is_over():
            raise ValueError('the filled cells are shown only when the attempt is over')
        view = self.observation()
        for r, c in self.filled & self.playable:
            view[r][c] = MINE
        return view


def _random_cells(rng, rows, cols, density) -> Set[Pos]:
    return {(r, c) for r in range(rows) for c in range(cols) if rng.random() < density}


def orthogonal(seed: int, rows: int = 8, cols: int = 8, density: float = 0.2, clue_share: float = 0.4) -> BinaryPuzzle:
    rng = random.Random(seed)
    filled = _random_cells(rng, rows, cols, density)
    clues = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in filled and rng.random() < clue_share:
                clues[(r, c)] = sum((r + dr, c + dc) in filled for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    cells = {(r, c) for r in range(rows) for c in range(cols)}
    return BinaryPuzzle(rows, cols, filled, clues, cells - set(clues))


def mosaic(seed: int, rows: int = 8, cols: int = 8, density: float = 0.45, clue_share: float = 0.5) -> BinaryPuzzle:
    rng = random.Random(seed)
    filled = _random_cells(rng, rows, cols, density)
    clues = {}
    for r in range(rows):
        for c in range(cols):
            if rng.random() < clue_share:
                clues[(r, c)] = sum((r + dr, c + dc) in filled for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    cells = {(r, c) for r in range(rows) for c in range(cols)}
    return BinaryPuzzle(rows, cols, filled, clues, cells - set(clues))


def totals(seed: int, n: int = 6, density: float = 0.4) -> BinaryPuzzle:
    rng = random.Random(seed)
    inner = {(r + 1, c + 1) for r, c in _random_cells(rng, n, n, density)}
    clues = {}
    for i in range(1, n + 1):
        clues[(i, 0)] = sum((i, c) in inner for c in range(1, n + 1))
        clues[(0, i)] = sum((r, i) in inner for r in range(1, n + 1))
    playable = {(r, c) for r in range(1, n + 1) for c in range(1, n + 1)}
    return BinaryPuzzle(n + 1, n + 1, inner, clues, playable)


def _runs(line: List[bool]) -> List[int]:
    runs, n = [], 0
    for v in line + [False]:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return runs or [0]


def nonogram(seed: int, n: int = 6, density: float = 0.5) -> BinaryPuzzle:
    rng = random.Random(seed)
    m = (n + 1) // 2                      # margin: the most runs a line of n cells can have
    cells = _random_cells(rng, n, n, density)
    filled = {(r + m, c + m) for r, c in cells}
    clues = {}
    for i in range(n):
        row = _runs([(i, c) in cells for c in range(n)])
        for k, v in enumerate(reversed(row)):
            clues[(i + m, m - 1 - k)] = v
        col = _runs([(r, i) in cells for r in range(n)])
        for k, v in enumerate(reversed(col)):
            clues[(m - 1 - k, i + m)] = v
    playable = {(r + m, c + m) for r in range(n) for c in range(n)}
    return BinaryPuzzle(n + m, n + m, filled, clues, playable)


def placement(seed: int) -> BinaryPuzzle:
    rng = random.Random(seed)

    def place(row, used_cols, used_boxes):
        if row == 9:
            return []
        cols = list(range(9))
        rng.shuffle(cols)
        for c in cols:
            box = (row // 3, c // 3)
            if c not in used_cols and box not in used_boxes:
                rest = place(row + 1, used_cols | {c}, used_boxes | {box})
                if rest is not None:
                    return [(row, c)] + rest
        return None
    filled = set(place(0, frozenset(), frozenset()))
    cells = {(r, c) for r in range(9) for c in range(9)}
    return BinaryPuzzle(9, 9, filled, {}, cells)


PUZZLES = {'orthogonal': orthogonal, 'mosaic': mosaic, 'totals': totals, 'nonogram': nonogram,
           'placement': placement}
