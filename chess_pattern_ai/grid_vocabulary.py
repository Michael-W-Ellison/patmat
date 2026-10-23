#!/usr/bin/env python3
"""
Grid Vocabulary - one set of grid relations for game boards and ARC grids

A board or puzzle is a Grid: integer cell values on rows x columns, plus a
topology saying which cells are neighbours and which cells form straight
lines. The relations below are defined only in those terms, so the same
relation means the same thing on a Go board, a Hex board, an Othello board
or an ARC grid:

- components: connected regions of equal value (or of any non-background
  value, "objects"), with size and bounding box
- adjacent_cells: cells bordering a region (e.g. a Go group's liberties)
- touches: whether cells reach a side of the grid
- lines / longest_line: maximal straight runs of equal value
- between / bracketed: cells on a straight segment between two cells, and
  runs of one value enclosed by another (Othello flips)
- enclosed: regions of non-wall cells that cannot reach the border
- symmetries: which reflections / rotations leave a grid or region unchanged
- counts / cells_of / describe: value counts, cells holding given
  values, and a summary of all of the above

Topologies:
- 'square': 8 compass directions; connectivity uses 4 neighbours
  (N, S, E, W) unless diagonal=True
- 'hex': offset rows where odd rows are shifted right (6 neighbours, as in
  hex/hex_board.py); lines run along the three hex axes
- 'graph': explicit adjacency and explicit line sets for boards that are
  not grids of cells (e.g. Nine Men's Morris); cells are positions in a
  grid embedding, other cells marked invalid
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Pos = Tuple[int, int]

SQUARE_STEPS = {
    'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1),
    'NE': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (1, -1),
}
SQUARE_ORTHOGONAL = ('N', 'S', 'E', 'W')
SQUARE_AXES = (('W', 'E'), ('N', 'S'), ('SW', 'NE'), ('NW', 'SE'))

# Odd rows shifted right: steps depend on row parity
HEX_STEPS = {
    0: {'E': (0, 1), 'W': (0, -1), 'NE': (-1, 0), 'NW': (-1, -1), 'SE': (1, 0), 'SW': (1, -1)},
    1: {'E': (0, 1), 'W': (0, -1), 'NE': (-1, 1), 'NW': (-1, 0), 'SE': (1, 1), 'SW': (1, 0)},
}
HEX_AXES = (('W', 'E'), ('SW', 'NE'), ('NW', 'SE'))
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E', 'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'}


class Grid:
    """
    Cell values plus topology.

    Args:
        cells: 2-D sequence of ints
        topology: 'square', 'hex' or 'graph'
        empty: value of an empty cell (used for open line ends and liberties)
        valid: optional 2-D bool mask; cells outside it are not part of the board
        adjacency: for 'graph', {pos: [neighbour positions]}
        line_sets: for 'graph', sequences of positions that form straight lines
    """

    def __init__(self, cells, topology: str = 'square', empty: int = 0,
                 valid=None, adjacency: Optional[Dict[Pos, Sequence[Pos]]] = None,
                 line_sets: Optional[Sequence[Sequence[Pos]]] = None):
        self.cells = np.asarray(cells, dtype=int)
        if self.cells.ndim != 2:
            raise ValueError("cells must be 2-D")
        if topology not in ('square', 'hex', 'graph'):
            raise ValueError(f"unknown topology {topology!r}")
        if topology == 'graph' and adjacency is None:
            raise ValueError("graph topology needs adjacency")
        self.topology = topology
        self.empty = empty
        self.valid = (np.ones(self.cells.shape, dtype=bool) if valid is None
                      else np.asarray(valid, dtype=bool))
        self.adjacency = {tuple(k): [tuple(p) for p in v] for k, v in (adjacency or {}).items()}
        self.line_sets = [tuple(tuple(p) for p in line) for line in (line_sets or [])]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cells.shape

    def __getitem__(self, pos: Pos) -> int:
        return int(self.cells[pos])

    def positions(self) -> Iterable[Pos]:
        rows, cols = np.nonzero(self.valid)
        return zip(rows.tolist(), cols.tolist())

    def inside(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.cells.shape[0] and 0 <= c < self.cells.shape[1] and bool(self.valid[r, c])

    def step(self, pos: Pos, direction: str) -> Optional[Pos]:
        """Neighbouring position in a named direction, or None off the board"""
        r, c = pos
        if self.topology == 'square':
            dr, dc = SQUARE_STEPS[direction]
        elif self.topology == 'hex':
            dr, dc = HEX_STEPS[r % 2][direction]
        else:
            raise ValueError("graph topology has no directions")
        nxt = (r + dr, c + dc)
        return nxt if self.inside(nxt) else None

    def neighbors(self, pos: Pos, diagonal: bool = False) -> List[Pos]:
        if self.topology == 'graph':
            return [p for p in self.adjacency.get(pos, []) if self.inside(p)]
        if self.topology == 'hex':
            names = HEX_STEPS[0].keys()
        else:
            names = SQUARE_STEPS.keys() if diagonal else SQUARE_ORTHOGONAL
        return [p for p in (self.step(pos, d) for d in names) if p is not None]

    def axes(self) -> Tuple[Tuple[str, str], ...]:
        return HEX_AXES if self.topology == 'hex' else SQUARE_AXES

    def border_cells(self) -> List[Pos]:
        """Valid cells with fewer neighbours than an interior cell would have"""
        if self.topology == 'graph':
            return []
        return [p for p in self.positions()
                if len(self.neighbors(p)) < (6 if self.topology == 'hex' else 4)]


@dataclass(frozen=True)
class Region:
    value: Optional[int]          # None for a multicolour object
    cells: FrozenSet[Pos]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return min(rows), min(cols), max(rows), max(cols)


@dataclass(frozen=True)
class Line:
    value: int
    cells: Tuple[Pos, ...]        # in order along the line
    axis: Tuple[str, str]
    open_ends: int                # ends next to an empty cell

    @property
    def length(self) -> int:
        return len(self.cells)


# ---------------------------------------------------------------- regions

def components(grid: Grid, values: Optional[Iterable[int]] = None, diagonal: bool = False,
               multicolor: bool = False, background: Optional[int] = None) -> List[Region]:
    """
    Connected regions.

    Default: regions of equal value (every value, or only `values`).
    multicolor=True: regions of any value other than `background` (default:
    the grid's empty value), i.e. objects made of several colours.
    """
    wanted = None if values is None else set(values)
    bg = grid.empty if background is None else background
    seen = set()
    regions = []
    for start in grid.positions():
        if start in seen:
            continue
        v = grid[start]
        if multicolor:
            if v == bg:
                continue
        elif wanted is not None and v not in wanted:
            continue
        cells = {start}
        queue = deque([start])
        seen.add(start)
        while queue:
            pos = queue.popleft()
            for nb in grid.neighbors(pos, diagonal):
                if nb in seen:
                    continue
                nv = grid[nb]
                if (nv != bg) if multicolor else (nv == v):
                    seen.add(nb)
                    cells.add(nb)
                    queue.append(nb)
        regions.append(Region(None if multicolor else v, frozenset(cells)))
    return regions


def adjacent_cells(grid: Grid, cells: Iterable[Pos], value: Optional[int] = None,
                   diagonal: bool = False) -> set:
    """Cells bordering a region (optionally only those holding `value`)"""
    cells = set(cells)
    out = set()
    for pos in cells:
        for nb in grid.neighbors(pos, diagonal):
            if nb not in cells and (value is None or grid[nb] == value):
                out.add(nb)
    return out


def touches(grid: Grid, cells: Iterable[Pos], side: str) -> bool:
    """Whether any cell lies on a side of the grid: 'top', 'bottom', 'left', 'right'"""
    rows, cols = grid.shape
    test = {'top': lambda p: p[0] == 0, 'bottom': lambda p: p[0] == rows - 1,
            'left': lambda p: p[1] == 0, 'right': lambda p: p[1] == cols - 1}[side]
    return any(test(p) for p in cells)


def enclosed(grid: Grid, walls: Iterable[int], diagonal: bool = False) -> List[Region]:
    """
    Regions of non-wall cells that cannot reach the grid border without
    crossing a wall (diagonal=True lets regions leak through diagonal gaps)
    """
    wall_values = set(walls)
    open_regions = components(
        Grid(np.where(np.isin(grid.cells, list(wall_values)), 1, 0), grid.topology,
             empty=1, valid=grid.valid, adjacency=grid.adjacency, line_sets=grid.line_sets),
        values=[0], diagonal=diagonal)
    border = set(grid.border_cells())
    return [Region(None, r.cells) for r in open_regions if not (r.cells & border)]


# ------------------------------------------------------------------ lines

def lines(grid: Grid, value: Optional[int] = None, min_length: int = 2,
          include_empty: bool = False) -> List[Line]:
    """Maximal straight runs of equal value (empty runs only if include_empty)"""
    found = []
    if grid.topology == 'graph':
        for line in grid.line_sets:
            run = []
            for pos in list(line) + [None]:
                if pos is not None and run and grid[pos] == grid[run[-1]]:
                    run.append(pos)
                    continue
                if run:
                    v = grid[run[0]]
                    if (len(run) >= min_length and (value is None or v == value)
                            and (include_empty or v != grid.empty)):
                        found.append(Line(v, tuple(run), ('line', 'line'), 0))
                run = [pos] if pos is not None else []
        return found

    for axis in grid.axes():
        back, fwd = axis
        for start in grid.positions():
            v = grid[start]
            if (value is not None and v != value) or (not include_empty and v == grid.empty):
                continue
            prev = grid.step(start, back)
            if prev is not None and grid[prev] == v:
                continue                      # not the start of a run
            run = [start]
            nxt = grid.step(start, fwd)
            while nxt is not None and grid[nxt] == v:
                run.append(nxt)
                nxt = grid.step(nxt, fwd)
            if len(run) >= min_length:
                open_ends = sum(1 for end in (prev, nxt) if end is not None and grid[end] == grid.empty)
                found.append(Line(v, tuple(run), axis, open_ends))
    return found


def longest_line(grid: Grid, value: int) -> int:
    """Length of the longest straight run of `value` (0 if the value is absent)"""
    runs = lines(grid, value, min_length=1, include_empty=True)
    return max((ln.length for ln in runs), default=0)


def between(grid: Grid, a: Pos, b: Pos) -> Optional[List[Pos]]:
    """Cells strictly between a and b if they lie on one straight line, else None"""
    if grid.topology == 'graph':
        for line in grid.line_sets:
            if a in line and b in line:
                i, j = sorted((line.index(a), line.index(b)))
                return list(line[i + 1:j])
        return None
    for axis in grid.axes():
        for direction in axis:
            path = []
            pos = grid.step(a, direction)
            while pos is not None:
                if pos == b:
                    return path
                path.append(pos)
                pos = grid.step(pos, direction)
    return None


def bracketed(grid: Grid, start: Pos, own: int, other: int) -> List[Pos]:
    """
    Cells of `other` enclosed in a straight line between `start` and a cell
    of `own` (in every direction from start): e.g. the discs an Othello move
    at `start` would flip
    """
    if grid.topology == 'graph':
        raise ValueError("bracketed needs a directional topology")
    out = []
    for axis in grid.axes():
        for direction in axis:
            run = []
            pos = grid.step(start, direction)
            while pos is not None and grid[pos] == other:
                run.append(pos)
                pos = grid.step(pos, direction)
            if run and pos is not None and grid[pos] == own:
                out.extend(run)
    return out


# --------------------------------------------------------------- symmetry

def _views(a: np.ndarray) -> Dict[str, np.ndarray]:
    views = {'flip_lr': np.fliplr(a), 'flip_ud': np.flipud(a), 'rot180': np.rot90(a, 2)}
    if a.shape[0] == a.shape[1]:
        views.update({'transpose': a.T, 'anti_transpose': np.rot90(a, 2).T, 'rot90': np.rot90(a)})
    return views


def symmetries(grid_or_array, region: Optional[Region] = None) -> Dict[str, bool]:
    """
    Which reflections and rotations leave the grid (or a region's bounding
    box, with cells outside the region treated as absent) unchanged
    """
    a = grid_or_array.cells if isinstance(grid_or_array, Grid) else np.asarray(grid_or_array)
    if region is not None:
        r0, c0, r1, c1 = region.bbox
        box = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
        for r, c in region.cells:
            box[r - r0, c - c0] = a[r, c]
        a = box
    return {name: bool(np.array_equal(a, v)) for name, v in _views(a).items()}


# --------------------------------------------------------------- summary

def counts(grid: Grid) -> Counter:
    """How many valid cells hold each value (the empty value included)"""
    return Counter(int(grid.cells[p]) for p in grid.positions())


def cells_of(grid: Grid, values: Iterable[int]) -> List[Pos]:
    """Valid cells holding any of `values`"""
    wanted = set(values)
    return [p for p in grid.positions() if grid[p] in wanted]


def describe(grid: Grid, diagonal: bool = False) -> Dict:
    """Summary of the grid in vocabulary terms (for comparing boards and puzzles)"""
    value_counts = counts(grid)
    regions = components(grid, diagonal=diagonal)
    objects = components(grid, multicolor=True, diagonal=diagonal)
    per_value = {}
    for v in sorted(value_counts):
        mine = [r for r in regions if r.value == v]
        per_value[v] = {
            'cells': value_counts[v],
            'regions': len(mine),
            'largest_region': max((r.size for r in mine), default=0),
            'longest_line': longest_line(grid, v),
        }
    non_empty = [v for v in value_counts if v != grid.empty]
    return {
        'shape': grid.shape,
        'values': per_value,
        'objects': len(objects),
        'enclosed_cells': sum(r.size for r in enclosed(grid, non_empty, diagonal)) if non_empty else 0,
        'symmetries': symmetries(grid) if grid.topology == 'square' else {},
    }
