#!/usr/bin/env python3
"""
Shapes: the representation of objects shared by the game-playing agent
(object_agent.py) and the ARC pattern solver (arc_pattern_solver.py)

- pattern(grid, cells): an object as the values of its cells over their
  bounding box (-1 off the cells); the agent keys its contact rules, its
  goals and its own shape by it
- GEOMETRY: the turns and mirrors of a pattern; the agent learns which one
  an action applies to its pattern (a Tetris piece turning), the solver
  tries them when a shape is copied or fitted
- fits_in(cells, targets): whether a shape (cells with their values),
  translated, falls on cells of the same values in one of the targets;
  the agent prunes its plans by it (its shape must stay inside a target
  object to reach the target's shape), the solver matches pieces to holes
  by the same test on silhouettes
- components(grid, background): the objects of a grid (4-connected cells
  that are not background, of any colours)
- images(pattern, operations): the pattern under each operation, as
  (name, pattern) pairs
- silhouette(pattern), stamp(grid, pattern, top_left), holes(grid, bg)
- symmetries(pattern), canonical(pattern): a pattern's images under the
  eight turns and mirrors of the square, and the least of them (the same
  key for a shape however it is turned or mirrored); the agent keys the
  neighbourhoods of the cells it places marks on by it
"""

from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Pos = Tuple[int, int]
SIDES = ((-1, 0), (1, 0), (0, -1), (0, 1))
TURNS = ('turn right', 'turn left', 'half turn')
MIRRORS = ('mirror left-right', 'mirror up-down', 'transpose')
GEOMETRY = {
    'turn right': lambda a: np.rot90(a, -1), 'turn left': lambda a: np.rot90(a, 1), 'half turn': lambda a: np.rot90(a, 2),
    'mirror left-right': np.fliplr, 'mirror up-down': np.flipud, 'transpose': np.transpose,
}


def pattern(grid, cells) -> tuple:
    """Values of the cells over their bounding box (-1 off the cells)"""
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    r0, c0 = min(rows), min(cols)
    out = [[-1] * (max(cols) - c0 + 1) for _ in range(max(rows) - r0 + 1)]
    for p in cells:
        out[p[0] - r0][p[1] - c0] = int(grid[p])
    return tuple(map(tuple, out))


def as_tuple(a: np.ndarray) -> tuple:
    return tuple(map(tuple, np.asarray(a).tolist()))


def images(pat: tuple, operations: Sequence[str] = ()) -> List[Tuple[str, tuple]]:
    """The pattern itself ('same') and under each operation, without repeats"""
    out, seen = [('same', pat)], {pat}
    for op in operations:
        img = as_tuple(GEOMETRY[op](np.array(pat)))
        if img not in seen:
            seen.add(img)
            out.append((op, img))
    return out


def symmetries(pat: tuple) -> List[tuple]:
    """The pattern under the eight turns and mirrors of the square (with repeats)"""
    a = np.array(pat)
    out = []
    for b in (a, a.T):
        for k in range(4):
            out.append(as_tuple(np.rot90(b, k)))
    return out


def canonical(pat: tuple) -> tuple:
    """The least of the pattern's images under the turns and mirrors of the square"""
    return min(symmetries(pat))


def silhouette(pat: tuple) -> tuple:
    """Which places of the pattern's box the shape covers"""
    return tuple(tuple(v != -1 for v in row) for row in pat)


def fits_in(cells: Sequence[Tuple[Pos, int]], targets: Iterable[Dict[Pos, int]]) -> bool:
    """Whether the cells, translated, fall on cells of the same values in one of the targets"""
    cells = list(cells)
    (a0, v0) = cells[0]
    for t in targets:
        if len(t) < len(cells):
            continue
        for q, v in t.items():
            if v == v0 and all(t.get((p[0] - a0[0] + q[0], p[1] - a0[1] + q[1])) == w for p, w in cells):
                return True
    return False


def components(grid: np.ndarray, background: int, diagonal: bool = False, one_colour: bool = False) \
        -> List[FrozenSet[Pos]]:
    """The objects: groups of connected cells that are not background (of any colours, or of one colour)"""
    sides = SIDES + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    seen, out = np.zeros(grid.shape, bool), []
    for r, c in zip(*np.nonzero(grid != background)):
        if seen[r, c]:
            continue
        group, stack = [], [(int(r), int(c))]
        seen[r, c] = True
        while stack:
            p = stack.pop()
            group.append(p)
            for dr, dc in sides:
                q = (p[0] + dr, p[1] + dc)
                if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and not seen[q] and grid[q] != background \
                        and (not one_colour or grid[q] == grid[p]):
                    seen[q] = True
                    stack.append(q)
        out.append(frozenset(group))
    return out


def holes(grid: np.ndarray, background: int) -> List[FrozenSet[Pos]]:
    """Groups of connected background cells that do not reach the grid's edge (enclosed by objects)"""
    h, w = grid.shape
    out = []
    for g in components((grid == background).astype(int), 0):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in g):
            out.append(g)
    return out


def stamp(grid: np.ndarray, pat: tuple, top_left: Pos) -> Optional[np.ndarray]:
    """The grid with the pattern's cells painted with their box's corner at top_left (None if it leaves the grid)"""
    out = grid.copy()
    for i, row in enumerate(pat):
        for j, v in enumerate(row):
            if v == -1:
                continue
            r, c = top_left[0] + i, top_left[1] + j
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                return None
            out[r, c] = v
    return out


def box(cells) -> Tuple[int, int]:
    """The top left corner of the cells' bounding box"""
    return min(p[0] for p in cells), min(p[1] for p in cells)
