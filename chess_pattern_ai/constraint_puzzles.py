#!/usr/bin/env python3
"""
Constraint puzzles in the view format of the constraint learner

Each puzzle is a board of filled and empty cells. Clue cells show a number
from the start, and some board cells may be shown from the start (empty or
filled). The player uncovers cells it believes are empty; uncovering a
filled cell ends the attempt. The puzzle is solved when every empty
playable cell is uncovered. After the attempt every filled cell is shown,
as the mines are after a game of Minesweeper.

Views use the learner's values: a number >= 0 for a clue, HIDDEN for a
board cell not yet uncovered, EMPTY for a board cell shown empty, MINE
for a filled cell shown as given (and, in the final view, for every
filled cell), OUTSIDE for a cell outside the board (an unused margin).

Every generated puzzle has exactly one solution. A random solution is
drawn, and while the exact solver (count_solutions, which uses the
puzzle's true rule) finds a second solution, information that rules it out
is added at a cell where the two differ: a clue at or next to it
(orthogonal, mosaic), or the cell shown empty from the start (totals,
nonogram, placement; their solutions all have the same number of filled
cells, so any other solution fills a cell that is truly empty).

Puzzles (each generated from a seed):
- orthogonal:  clue = filled cells among the clue cell's 4 orthogonal
               neighbours (clue cells are empty; a static Minesweeper variant)
- mosaic:      clue = filled cells in the 3x3 block including the clue cell
               itself, which may be filled (Fill-a-Pix)
- totals:      margin clues = filled cells in their row or column
- nonogram:    margin clues = lengths of the runs of filled cells in their
               row or column, in order
- placement:   one filled cell in every row, column and 3x3 box of a 9x9
               grid (the cells of one Sudoku digit); no clues, some filled
               cells shown from the start
"""

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from minesweeper.constraint_learner import _placements, _runs
from minesweeper.minesweeper_game import HIDDEN, MINE, OUTSIDE, SHOWN_EMPTY

Pos = Tuple[int, int]
EMPTY = SHOWN_EMPTY
MAX_TRIES = 50


class BinaryPuzzle:
    def __init__(self, rows: int, cols: int, filled: Set[Pos], clues: Dict[Pos, int], board: Set[Pos],
                 shown_empty: Set[Pos] = frozenset(), shown_filled: Set[Pos] = frozenset()):
        self.rows, self.cols = rows, cols
        self.filled, self.clues, self.board = set(filled), dict(clues), set(board)
        self.shown_filled = set(shown_filled)
        self.playable = self.board - set(self.clues) - self.shown_filled
        self.shown_empty = set(shown_empty)
        self.revealed: Set[Pos] = set(shown_empty)
        self.exploded: Optional[Pos] = None
        self.constraints = None     # (variables, sums, runs, known) the uniqueness was checked with

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
        """Share of the empty playable cells uncovered by the player (cells shown from the start excluded)"""
        empty = self.playable - self.filled - self.shown_empty
        return len(self.revealed - self.shown_empty) / len(empty) if empty else 1.0

    def observation(self) -> List[List[int]]:
        view = [[OUTSIDE] * self.cols for _ in range(self.rows)]
        for r, c in self.board:
            view[r][c] = EMPTY if (r, c) in self.revealed else MINE if (r, c) in self.shown_filled else HIDDEN
        for (r, c), v in self.clues.items():
            view[r][c] = v
        return view

    def final_view(self) -> List[List[int]]:
        if not self.is_over():
            raise ValueError('the filled cells are shown only when the attempt is over')
        view = self.observation()
        for r, c in self.filled:            # filled clue cells too (a player keeps the clues it saw)
            view[r][c] = MINE
        return view


# ------------------------------------------------------------ exact solver

def count_solutions(variables: Sequence[Pos], sums: List[Tuple[List[Pos], int]],
                    runs: List[Tuple[List[Pos], List[int]]], known: Dict[Pos, bool],
                    limit: int = 2) -> List[Dict[Pos, bool]]:
    """
    The assignments of the variables (up to `limit` of them) satisfying
    every sum ("these cells hold exactly n filled") and run constraint.
    Backtracking with propagation: a sum already met forces its open cells
    empty, a sum met only by filling every open cell forces them filled,
    and a run line forces the cells all its placements agree on.
    """
    cons = [('sum', list(cells), n) for cells, n in sums] + [('runs', list(cells), r) for cells, r in runs]
    by_var: Dict[Pos, List[int]] = {}
    for i, (_, cells, _) in enumerate(cons):
        for p in cells:
            by_var.setdefault(p, []).append(i)
    variables = list(variables)
    found: List[Dict[Pos, bool]] = []

    def propagate(assign, queue):
        """Apply forced values; False on a contradiction"""
        while queue:
            i = queue.pop()
            kind, cells, n = cons[i]
            open_cells = [p for p in cells if p not in assign]
            if kind == 'sum':
                filled = sum(1 for p in cells if assign.get(p) is True)
                if not filled <= n <= filled + len(open_cells):
                    return False
                if not open_cells or (filled != n and filled + len(open_cells) != n):
                    continue
                value = filled + len(open_cells) == n and filled != n
                forced = [(p, value) for p in open_cells]
            else:
                options = _placements(n, [assign.get(p) for p in cells])
                if not options:
                    return False
                forced = [(p, options[0][j]) for j, p in enumerate(cells)
                          if p not in assign and all(o[j] == options[0][j] for o in options)]
            for p, v in forced:
                assign[p] = v
                queue.extend(by_var.get(p, ()))
        return True

    def search(assign):
        if len(found) >= limit:
            return
        open_vars = [p for p in variables if p not in assign]
        if not open_vars:
            found.append(dict(assign))
            return
        # branch on a cell of the constraint with the fewest open cells
        best, best_open = None, None
        for p in open_vars:
            for i in by_var.get(p, ()):
                k = sum(1 for q in cons[i][1] if q not in assign)
                if best_open is None or k < best_open:
                    best, best_open = p, k
        p = best if best is not None else open_vars[0]
        for v in (False, True):
            trial = dict(assign)
            trial[p] = v
            if propagate(trial, list(by_var.get(p, ()))):
                search(trial)

    assign = dict(known)
    if propagate(assign, list(range(len(cons)))):
        search(assign)
    return found


def _random_cells(rng, cells, density) -> Set[Pos]:
    return {p for p in cells if rng.random() < density}


def _square(rows, cols) -> List[Pos]:
    return [(r, c) for r in range(rows) for c in range(cols)]


# -------------------------------------------------------------- generators

def _local(seed, rows, cols, density, offsets, clue_may_be_filled):
    """Orthogonal / mosaic: add clues where solutions differ until only one remains"""
    rng = random.Random(seed)
    cells = _square(rows, cols)
    for _ in range(MAX_TRIES):
        filled = _random_cells(rng, cells, density)
        count = lambda p: sum((p[0] + dr, p[1] + dc) in filled for dr, dc in offsets)
        scope = lambda p: [(p[0] + dr, p[1] + dc) for dr, dc in offsets
                           if 0 <= p[0] + dr < rows and 0 <= p[1] + dc < cols]
        clues: Dict[Pos, int] = {}
        for _ in range(rows * cols):
            variables = [p for p in cells if clue_may_be_filled or p not in clues]
            known = {} if clue_may_be_filled else {p: False for p in clues}
            sums = [(scope(p), v) for p, v in clues.items()]
            sols = count_solutions(variables, sums, [], known)
            if len(sols) == 1:
                puzzle = BinaryPuzzle(rows, cols, filled, clues, set(cells))
                puzzle.constraints = (variables, sums, [], known)
                return puzzle
            other = next(s for s in sols if any(s.get(p, False) != (p in filled) for p in cells))
            diff = [p for p in cells if other.get(p, False) != (p in filled)]
            # a clue at a differing cell, or one whose scope holds it (clue cells must be empty unless
            # they may be filled: then a clue at an empty differing cell also fixes that cell)
            candidates = sorted({q for t in diff for q in [t] + scope(t)
                                 if q not in clues and (clue_may_be_filled or q not in filled)
                                 and (t in scope(q) or q == t)})
            if not candidates:
                break               # this solution cannot be pinned down with clues; draw another
            q = rng.choice(candidates)
            clues[q] = count(q)
    raise RuntimeError('no uniquely solvable puzzle found')


def orthogonal(seed: int, rows: int = 8, cols: int = 8, density: float = 0.2) -> BinaryPuzzle:
    return _local(seed, rows, cols, density, [(-1, 0), (1, 0), (0, -1), (0, 1)], False)


def mosaic(seed: int, rows: int = 8, cols: int = 8, density: float = 0.45) -> BinaryPuzzle:
    return _local(seed, rows, cols, density, [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], True)


def _show_empty_until_unique(rng, board, filled, sums, runs, known):
    """Show empty board cells from the start until the solution is unique; (shown, constraints)"""
    shown = set()
    while True:
        variables = [p for p in board if p not in known and p not in shown]
        fixed = {**known, **{p: False for p in shown}}
        sols = count_solutions(variables, sums, runs, fixed)
        if len(sols) == 1:
            return shown, (variables, sums, runs, fixed)
        other = next(s for s in sols if any(s[p] != (p in filled) for p in board))
        diff = [p for p in board if other[p] and p not in filled]
        shown.add(rng.choice(diff))


def totals(seed: int, n: int = 6, density: float = 0.4) -> BinaryPuzzle:
    rng = random.Random(seed)
    board = [(r + 1, c + 1) for r in range(n) for c in range(n)]
    filled = _random_cells(rng, board, density)
    clues, sums = {}, []
    for i in range(1, n + 1):
        row, col = [(i, c) for c in range(1, n + 1)], [(r, i) for r in range(1, n + 1)]
        clues[(i, 0)] = sum(p in filled for p in row)
        clues[(0, i)] = sum(p in filled for p in col)
        sums += [(row, clues[(i, 0)]), (col, clues[(0, i)])]
    shown, constraints = _show_empty_until_unique(rng, board, filled, sums, [], {})
    puzzle = BinaryPuzzle(n + 1, n + 1, filled, clues, set(board), shown_empty=shown)
    puzzle.constraints = constraints
    return puzzle


def nonogram(seed: int, n: int = 6, density: float = 0.5) -> BinaryPuzzle:
    rng = random.Random(seed)
    m = (n + 1) // 2                      # margin: the most runs a line of n cells can have
    board = [(r + m, c + m) for r in range(n) for c in range(n)]
    filled = _random_cells(rng, board, density)
    clues, runs = {}, []
    for i in range(n):
        row = [(i + m, c + m) for c in range(n)]
        col = [(r + m, i + m) for r in range(n)]
        row_runs, col_runs = _runs([p in filled for p in row]), _runs([p in filled for p in col])
        for k, v in enumerate(reversed(row_runs)):
            clues[(i + m, m - 1 - k)] = v
        for k, v in enumerate(reversed(col_runs)):
            clues[(m - 1 - k, i + m)] = v
        runs += [(row, row_runs), (col, col_runs)]
    shown, constraints = _show_empty_until_unique(rng, board, filled, [], runs, {})
    puzzle = BinaryPuzzle(n + m, n + m, filled, clues, set(board), shown_empty=shown)
    puzzle.constraints = constraints
    return puzzle


def placement(seed: int, shown_filled: int = 3) -> BinaryPuzzle:
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
    board = _square(9, 9)
    given = set(rng.sample(sorted(filled), shown_filled))
    regions = [[(r, c) for c in range(9)] for r in range(9)] + [[(r, c) for r in range(9)] for c in range(9)] \
        + [[(3 * i + r, 3 * j + c) for r in range(3) for c in range(3)] for i in range(3) for j in range(3)]
    shown, constraints = _show_empty_until_unique(rng, board, filled, [(reg, 1) for reg in regions], [],
                                                  {p: True for p in given})
    puzzle = BinaryPuzzle(9, 9, filled, {}, set(board), shown_empty=shown, shown_filled=given)
    puzzle.constraints = constraints
    return puzzle


PUZZLES = {'orthogonal': orthogonal, 'mosaic': mosaic, 'totals': totals, 'nonogram': nonogram,
           'placement': placement}
