#!/usr/bin/env python3
"""
Connectivity puzzles (Nurikabe, Slitherlink) in the constraint learner's view format

- Nurikabe (n x n): cells are sea (filled) or island (empty). Some island
  cells show a number. Rules: every island holds exactly one number, equal
  to its size; all sea cells are connected; no 2x2 block is all sea.
- Slitherlink (rows x cols cells), shown as a lattice of (2 rows + 1) x
  (2 cols + 1) positions: dots at even/even positions (shown as DOT), cell
  centres at odd/odd (a clue number, or OUTSIDE), and edges in between (the
  board; an edge on the loop is filled). Rules: a number equals the loop
  edges around its cell; every dot touches 0 or 2 loop edges; the loop
  edges are connected through the dots (one closed loop).

The player uncovers cells it believes are empty (sea or loop edges end the
attempt), as in constraint_puzzles. Every puzzle has exactly one solution,
checked by solve() with the true rules; where two solutions differ, a cell
is shown from the start (Nurikabe) or a clue is added (Slitherlink).
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from constraint_puzzles import BinaryPuzzle
from global_constraints import SIDES, consistent, holds

Pos = Tuple[int, int]
DOT = -5
NURIKABE_RULES = [('islands', 'not-mine', 'one clue = size'), ('connected', 'mine', 'sides'), ('no 2x2', 'mine', '')]
SLITHERLINK_RULES = [('degree', DOT, frozenset({0, 2})), ('connected', 'mine', f'via {DOT}')]


def solve(board: Set[Pos], clues: Dict[Pos, int], markers: Dict[Pos, int], rules, counts,
          known: Dict[Pos, bool], limit: int = 2) -> List[Set[Pos]]:
    """
    Solutions (sets of filled cells, up to `limit`) of a binary board under
    global rules and clue counts (clue cell -> cells whose filled count it is)
    """
    found = []

    def ok(mines, empty, unknown):
        for cell, cells in counts.items():
            m = sum(p in mines for p in cells)
            u = sum(p in unknown for p in cells)
            if not m <= clues[cell] <= m + u:
                return False
        return all(consistent(r, mines, empty, unknown, clues, markers) for r in rules)

    order = sorted(board - set(known))

    def search(i, mines, empty):
        if len(found) >= limit:
            return
        unknown = set(order[i:])
        if not ok(mines, empty, unknown):
            return
        if i == len(order):
            found.append(set(mines))
            return
        p = order[i]
        for filled in (False, True):
            (mines if filled else empty).add(p)
            search(i + 1, mines, empty)
            (mines if filled else empty).discard(p)

    search(0, {p for p, v in known.items() if v}, {p for p, v in known.items() if not v})
    return found


# --------------------------------------------------------------- Nurikabe

def _nurikabe_solution(rng, n) -> Optional[Tuple[Set[Pos], List[Set[Pos]]]]:
    cells = {(r, c) for r in range(n) for c in range(n)}
    islands: List[Set[Pos]] = []
    sea = set(cells)

    def owner(p):
        return next((i for i, isl in enumerate(islands) if p in isl), None)
    for _ in range(n * n * 4):
        if not any(b <= sea for b in [{(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                                      for r in range(n - 1) for c in range(n - 1)]):
            break
        p = rng.choice(sorted(sea))
        touching = {owner(q) for q in [(p[0] + dr, p[1] + dc) for dr, dc in SIDES]} - {None}
        if len(touching) > 1:
            continue
        if touching:
            i = touching.pop()
            if len(islands[i]) >= 6:
                continue
        else:
            i = None
        trial = sea - {p}
        if not holds(('connected', 'mine', 'sides'), trial, set(), {}, {}):
            continue
        sea = trial
        if i is None:
            islands.append({p})
        else:
            islands[i].add(p)
    if any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= sea for r in range(n - 1) for c in range(n - 1)):
        return None
    return sea, islands


def nurikabe(seed: int, n: int = 6) -> BinaryPuzzle:
    rng = random.Random(seed)
    while True:
        made = _nurikabe_solution(rng, n)
        if made and any(len(isl) > 1 for isl in made[1]):
            break                   # (islands all of size 1 would leave nothing to uncover)
    sea, islands = made
    clues = {}
    for isl in islands:
        clues[rng.choice(sorted(isl))] = len(isl)
    board = {(r, c) for r in range(n) for c in range(n)} - set(clues)
    known: Dict[Pos, bool] = {}
    while True:
        sols = solve(board, clues, {}, NURIKABE_RULES, {}, known)
        if len(sols) == 1:
            break
        other = next(s for s in sols if s != sea)
        p = rng.choice(sorted(other ^ sea))
        known[p] = p in sea
    puzzle = BinaryPuzzle(n, n, sea, clues, board,
                          shown_empty={p for p, v in known.items() if not v},
                          shown_filled={p for p, v in known.items() if v})
    puzzle.rules = NURIKABE_RULES
    return puzzle


# ------------------------------------------------------------ Slitherlink

def _region(rng, rows, cols) -> Set[Pos]:
    """A random connected region of cells without holes or corner-only contacts (its outline is one loop)"""
    cells = {(r, c) for r in range(rows) for c in range(cols)}
    target = rng.randint(rows * cols // 3, rows * cols * 2 // 3)
    region = {rng.choice(sorted(cells))}
    for _ in range(rows * cols * 20):
        if len(region) >= target:
            break
        options = sorted({(r + dr, c + dc) for r, c in region for dr, dc in SIDES} & (cells - region))
        p = rng.choice(options)
        trial = region | {p}
        outside = (cells - trial)
        # outside cells must reach the border (no holes)
        border = {q for q in outside if q[0] in (0, rows - 1) or q[1] in (0, cols - 1)}
        reach, queue = set(border), list(border)
        while queue:
            q = queue.pop()
            for dr, dc in SIDES:
                x = (q[0] + dr, q[1] + dc)
                if x in outside and x not in reach:
                    reach.add(x)
                    queue.append(x)
        if reach != outside:
            continue
        # no two region cells touching only at a corner with both other cells outside
        pinch = any((r, c) in trial and (r + 1, c + 1) in trial and (r + 1, c) not in trial and (r, c + 1) not in trial
                    or (r + 1, c) in trial and (r, c + 1) in trial and (r, c) not in trial and (r + 1, c + 1) not in trial
                    for r in range(-1, rows) for c in range(-1, cols))
        if not pinch:
            region = trial
    return region


def slitherlink(seed: int, rows: int = 4, cols: int = 4) -> BinaryPuzzle:
    rng = random.Random(seed)
    region = _region(rng, rows, cols)
    R, C = 2 * rows + 1, 2 * cols + 1
    edges = {(r, c) for r in range(R) for c in range(C) if (r + c) % 2 == 1}
    dots = {(r, c): DOT for r in range(0, R, 2) for c in range(0, C, 2)}
    centre = {(r, c): (2 * r + 1, 2 * c + 1) for r in range(rows) for c in range(cols)}

    def inside(r, c):
        return (r, c) in region
    loop = set()
    for (r, c), (y, x) in centre.items():
        for dr, dc in SIDES:
            if inside(r, c) != inside(r + dr, c + dc):
                loop.add((y + dr, x + dc))
    number = {(y, x): sum((y + dr, x + dc) in loop for dr, dc in SIDES) for y, x in centre.values()}
    order = sorted(number)
    rng.shuffle(order)
    clues = {p: number[p] for p in order[:len(order) // 2]}
    while True:
        counts = {p: [(p[0] + dr, p[1] + dc) for dr, dc in SIDES] for p in clues}
        sols = solve(edges, clues, dots, SLITHERLINK_RULES, counts, {})
        if len(sols) == 1:
            break
        other = next(s for s in sols if s != loop)
        diff = other ^ loop
        near = sorted({c for c in number if c not in clues and any((c[0] + dr, c[1] + dc) in diff for dr, dc in SIDES)})
        p = rng.choice(near or sorted(set(number) - set(clues)))
        clues[p] = number[p]
    markers = dict(dots)
    puzzle = BinaryPuzzle(R, C, loop, clues, edges, markers=markers)
    puzzle.rules = SLITHERLINK_RULES
    return puzzle


PUZZLES = {
    'nurikabe 6x6': lambda seed: nurikabe(seed),
    'slitherlink 4x4': lambda seed: slitherlink(seed),
}
