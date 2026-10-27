#!/usr/bin/env python3
"""
Number-placement puzzles (Sudoku, KenKen) with unique solutions

Every cell of an n x n board takes a value 1..N. Some cells are shown
from the start. A KenKen board is also divided into cages; each cage shows
a clue: a target number and an operation symbol ('+', '-', 'x', '/', or
'=' for a one-cell cage). The player fills one cell at a time; a wrong
value ends the attempt. The puzzle is solved when every cell is filled.
After the attempt the whole solution is shown.

The player sees an Observation:
    values  n x n list, 0 for an empty cell
    cages   n x n list of cage ids (-1 where there are no cages)
    clues   cage id -> (target, symbol)

Every generated puzzle has exactly one solution, checked by count_solutions
with the puzzle's true rules; given cells are added where two solutions
differ until only one remains.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Pos = Tuple[int, int]

OPERATIONS: Dict[str, Callable[[Sequence[int]], Optional[int]]] = {
    'sum': lambda v: sum(v),
    'product': lambda v: _product(v),
    'difference': lambda v: max(v) - min(v) if len(v) == 2 else None,
    'quotient': lambda v: max(v) // min(v) if len(v) == 2 and max(v) % min(v) == 0 else None,
    'largest': lambda v: max(v),
    'smallest': lambda v: min(v),
    'cells': lambda v: len(v),
    'value': lambda v: v[0] if len(v) == 1 else None,
}
TRUE_MEANING = {'+': 'sum', '-': 'difference', 'x': 'product', '/': 'quotient', '=': 'value'}


def _product(v):
    p = 1
    for x in v:
        p *= x
    return p


@dataclass
class Observation:
    values: List[List[int]]
    cages: List[List[int]]
    clues: Dict[int, Tuple[int, str]] = field(default_factory=dict)


# ------------------------------------------------------- constraint engine

class Constraints:
    """All-different regions and cage clues over the cells of an n x n board, values 1..N"""

    def __init__(self, n: int, values: int, regions: List[List[Pos]],
                 cages: List[Tuple[List[Pos], Callable, int]]):
        self.n, self.values = n, values
        self.regions = [list(r) for r in regions]
        self.cages = [(list(c), f, t) for c, f, t in cages]
        self.peers: Dict[Pos, set] = {}
        self.of: Dict[Pos, List[int]] = {}
        for i, reg in enumerate(self.regions):
            for p in reg:
                self.peers.setdefault(p, set()).update(q for q in reg if q != p)
        for i, (cells, _, _) in enumerate(self.cages):
            for p in cells:
                self.of.setdefault(p, []).append(i)

    def _cage_support(self, i, dom) -> Optional[Dict[Pos, set]]:
        cells, f, target = self.cages[i]
        support = {p: set() for p in cells}
        for combo in itertools.product(*(sorted(dom[p]) for p in cells)):
            if any(combo[a] == combo[b] for a, b in itertools.combinations(range(len(cells)), 2)
                   if cells[b] in self.peers.get(cells[a], ())):
                continue
            if f(combo) == target:
                for p, v in zip(cells, combo):
                    support[p].add(v)
        return None if any(not s for s in support.values()) else support

    def propagate(self, dom: Dict[Pos, set]) -> bool:
        """Narrow the domains in place to what the constraints allow; False on a contradiction"""
        changed = True
        while changed:
            changed = False
            for p, d in dom.items():
                if not d:
                    return False
                if len(d) == 1:
                    v = next(iter(d))
                    for q in self.peers.get(p, ()):
                        if v in dom[q]:
                            if len(dom[q]) == 1:
                                return False
                            dom[q].discard(v)
                            changed = True
            for reg in self.regions:
                if len(reg) != self.values:
                    continue
                for v in range(1, self.values + 1):
                    places = [p for p in reg if v in dom[p]]
                    if not places:
                        return False
                    if len(places) == 1 and len(dom[places[0]]) > 1:
                        dom[places[0]] = {v}
                        changed = True
            for i in range(len(self.cages)):
                support = self._cage_support(i, dom)
                if support is None:
                    return False
                for p, s in support.items():
                    if s != dom[p]:
                        dom[p] = s
                        changed = True
        return True

    def domains(self, values: List[List[int]]) -> Dict[Pos, set]:
        n = self.n
        return {(r, c): ({values[r][c]} if values[r][c] else set(range(1, self.values + 1)))
                for r in range(n) for c in range(n)}


def count_solutions(cons: Constraints, values: List[List[int]], limit: int = 2) -> List[Dict[Pos, int]]:
    """Solutions (up to `limit`), by backtracking on the cell with the fewest values left"""
    found = []

    def search(dom):
        if len(found) >= limit or not cons.propagate(dom):
            return
        open_cells = [p for p, d in dom.items() if len(d) > 1]
        if not open_cells:
            found.append({p: next(iter(d)) for p, d in dom.items()})
            return
        p = min(open_cells, key=lambda q: len(dom[q]))
        for v in sorted(dom[p]):
            trial = {q: set(d) for q, d in dom.items()}
            trial[p] = {v}
            search(trial)

    search(cons.domains(values))
    return found


# ---------------------------------------------------------------- puzzles

class ValuePuzzle:
    def __init__(self, solution: List[List[int]], given: List[Pos], cages: Optional[List[List[int]]] = None,
                 clues: Optional[Dict[int, Tuple[int, str]]] = None):
        self.n = len(solution)
        self.solution = [list(r) for r in solution]
        self.values = [[solution[r][c] if (r, c) in set(given) else 0 for c in range(self.n)] for r in range(self.n)]
        self.given = set(given)
        self.cages = cages or [[-1] * self.n for _ in range(self.n)]
        self.clues = dict(clues or {})
        self.failed = False
        self.constraints: Optional[Constraints] = None

    def observation(self) -> Observation:
        return Observation([list(r) for r in self.values], [list(r) for r in self.cages], dict(self.clues))

    def legal_moves(self) -> List[Pos]:
        return [] if self.is_over() else [(r, c) for r in range(self.n) for c in range(self.n) if not self.values[r][c]]

    def fill(self, pos: Pos, value: int):
        r, c = pos
        if self.is_over() or self.values[r][c]:
            raise ValueError(f'cannot fill {pos}')
        if self.solution[r][c] != value:
            self.failed = True
        else:
            self.values[r][c] = value

    def won(self) -> bool:
        return not self.failed and all(all(row) for row in self.values)

    def is_over(self) -> bool:
        return self.failed or self.won()

    def progress(self) -> float:
        open_cells = self.n * self.n - len(self.given)
        filled = sum(1 for r in range(self.n) for c in range(self.n) if self.values[r][c]) - len(self.given)
        return filled / open_cells if open_cells else 1.0

    def final_view(self) -> Observation:
        if not self.is_over():
            raise ValueError('the solution is shown only when the attempt is over')
        return Observation([list(r) for r in self.solution], [list(r) for r in self.cages], dict(self.clues))


def _latin_regions(n: int, box: Optional[Tuple[int, int]]) -> List[List[Pos]]:
    regions = [[(r, c) for c in range(n)] for r in range(n)] + [[(r, c) for r in range(n)] for c in range(n)]
    if box:
        a, b = box
        regions += [[(i + r, j + c) for r in range(a) for c in range(b)]
                    for i in range(0, n, a) for j in range(0, n, b)]
    return regions


def _fill_grid(rng, n, box) -> List[List[int]]:
    """A random complete grid with all-different rows, columns (and boxes)"""
    cons = Constraints(n, n, _latin_regions(n, box), [])
    dom = cons.domains([[0] * n for _ in range(n)])

    def search(dom):
        if not cons.propagate(dom):
            return None
        open_cells = [p for p, d in dom.items() if len(d) > 1]
        if not open_cells:
            return dom
        p = min(open_cells, key=lambda q: (len(dom[q]), rng.random()))
        for v in rng.sample(sorted(dom[p]), len(dom[p])):
            trial = {q: set(d) for q, d in dom.items()}
            trial[p] = {v}
            res = search(trial)
            if res:
                return res
        return None
    dom = search(dom)
    return [[next(iter(dom[(r, c)])) for c in range(n)] for r in range(n)]


def _give_until_unique(rng, solution, cons, given):
    """Show cells (where two solutions differ) until the solution is unique"""
    n = len(solution)
    given = set(given)
    while True:
        values = [[solution[r][c] if (r, c) in given else 0 for c in range(n)] for r in range(n)]
        sols = count_solutions(cons, values)
        if len(sols) == 1:
            return given
        other = next(s for s in sols if any(s[(r, c)] != solution[r][c] for r in range(n) for c in range(n)))
        diff = [(r, c) for r in range(n) for c in range(n) if other[(r, c)] != solution[r][c]]
        given.add(rng.choice(diff))


def sudoku(seed: int, n: int = 9, box: Tuple[int, int] = (3, 3)) -> ValuePuzzle:
    """Sudoku: start from a full grid, empty cells in random order while the solution stays unique"""
    rng = random.Random(seed)
    solution = _fill_grid(rng, n, box)
    cons = Constraints(n, n, _latin_regions(n, box), [])
    given = {(r, c) for r in range(n) for c in range(n)}
    for p in rng.sample(sorted(given), len(given)):
        trial = given - {p}
        values = [[solution[r][c] if (r, c) in trial else 0 for c in range(n)] for r in range(n)]
        if len(count_solutions(cons, values)) == 1:
            given = trial
    puzzle = ValuePuzzle(solution, sorted(given))
    puzzle.constraints = cons
    return puzzle


def _cages(rng, n, max_size=4) -> List[List[Pos]]:
    left = {(r, c) for r in range(n) for c in range(n)}
    cages = []
    while left:
        start = min(left) if rng.random() < 0.5 else rng.choice(sorted(left))
        cage = [start]
        left.discard(start)
        size = rng.choice([1, 2, 2, 2, 3, 3, 4][:max_size + 3])
        while len(cage) < size:
            options = sorted({(r + dr, c + dc) for r, c in cage for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
                             & left)
            if not options:
                break
            p = rng.choice(options)
            cage.append(p)
            left.discard(p)
        cages.append(cage)
    return cages


def kenken(seed: int, n: int = 4) -> ValuePuzzle:
    rng = random.Random(seed)
    solution = _fill_grid(rng, n, None)
    cages = _cages(rng, n)
    cage_map = [[-1] * n for _ in range(n)]
    clues, cage_cons = {}, []
    for i, cells in enumerate(cages):
        vals = [solution[r][c] for r, c in cells]
        if len(cells) == 1:
            symbol = '='
        elif len(cells) == 2:
            options = ['+', 'x', '-'] + (['/'] if max(vals) % min(vals) == 0 else [])
            symbol = rng.choice(options)
        else:
            symbol = rng.choice(['+', 'x'])
        f = OPERATIONS[TRUE_MEANING[symbol]]
        clues[i] = (f(vals), symbol)
        for r, c in cells:
            cage_map[r][c] = i
        cage_cons.append((cells, f, clues[i][0]))
    cons = Constraints(n, n, _latin_regions(n, None), cage_cons)
    given = _give_until_unique(rng, solution, cons, set())
    puzzle = ValuePuzzle(solution, sorted(given), cage_map, clues)
    puzzle.constraints = cons
    return puzzle


PUZZLES = {
    'sudoku 9x9': lambda seed: sudoku(seed),
    'kenken 4x4': lambda seed: kenken(seed, 4),
    'kenken 6x6': lambda seed: kenken(seed, 6),
}
