#!/usr/bin/env python3
"""
Matrix Puzzles - complete a 3 x 3 matrix of panels whose rows (or columns) follow rules

A matrix puzzle is a 3 x 3 arrangement of panels (integer grids, 0 the
background) with one panel missing, usually the last; each row, or each
column, follows rules such as rotation, colour change, size progression or
the combination of two panels into the third, often several at once. The
answer is the missing panel; some puzzles also offer answer choices.

Reading a puzzle (MatrixPuzzle): from panels directly, or from one grid in
which the panels are separated by lines of one colour (from_grid), with the
missing panel filled with a marker value.

Solving (solve): every rule family below is tried along the rows and along
the columns (the matrix transposed). A rule is kept only if it reproduces
every complete row (or column) exactly; each kept rule predicts the missing
panel. Predictions are ranked by family, simplest first (the order of
FAMILIES), rows before columns. With answer choices, the answer is the
best-ranked prediction among the choices; without, the best-ranked
prediction.

Rule families:
- distribution: every row holds the same three panels, in some order; the
  missing panel is the one the last row lacks.
- attributes: each panel that is one object of one colour is described by
  its attributes, and each attribute follows its own rule, so several rules
  hold at once (rotation and colour and size):
      colour  constant along the row, the same in each column, a fixed
              mapping from each panel's colour to the next panel's (learned
              from every step; a colour change), or a distribution of three
      scale   the factor the object's base shape is enlarged by (a shape
              drawn at 1x, 2x, 3x); an arithmetic progression along the row
              (the step may differ between rows: it is read from the row's
              first two panels), constant, the same in each column, or a
              distribution of three
      shape   the base shape (the object reduced by its scale): each next
              panel's is the previous one's under one fixed turn or mirror
              (identity: constant; rot90: a rotation), the same in each
              column, or a distribution of three
      place   where the object is (its top-left corner, or its centre): an
              arithmetic progression, constant, or the same in each column
  The predicted attributes are drawn into an empty panel.
- combination: the third panel is made cell by cell from the first two:
  each cell's value depends on whether each of the two cells is background
  and whether they are equal, and is background, the first cell's value,
  the second's, or a colour fixed for the matrix (so OR, AND, XOR and
  differences in any colours); the result may then be turned or mirrored by
  one fixed operation.
- arc_step / arc_combination: the puzzle put to the ARC pattern solver
  (arc_pattern_solver): each step from a panel to the next panel as a
  training pair (the last row's last step is the test), or the first two
  panels side by side (separated by a line of SEPARATOR) as the input and the
  third as the output. Slow, so tried only when the families above predict
  nothing (or none of their predictions is among the choices).

Steps that change by different amounts (a shape drawn at 1x, 2x, 3x) are
an arithmetic progression of the scale attribute; chains of three rules
(rotation, colour and combination) are the combination family's operation
followed by a turn and a fixed colour, or independent attribute rules.
"""

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
SEPARATOR = 10          # arc_combination: the separator line between the two input panels (a colour no panel uses)
FAMILIES = ('distribution', 'attributes', 'combination', 'arc_step', 'arc_combination')

# the 8 turns and mirrors of a square grid
SYMMETRIES: Dict[str, Callable[[Array], Array]] = {
    'identity': lambda a: a,
    'rot90': lambda a: np.rot90(a, 1),
    'rot180': lambda a: np.rot90(a, 2),
    'rot270': lambda a: np.rot90(a, 3),
    'flip_lr': np.fliplr,
    'flip_ud': np.flipud,
    'transpose': lambda a: a.T,
    'antitranspose': lambda a: np.rot90(a, 2).T,
}
INVERSE = {'identity': 'identity', 'rot90': 'rot270', 'rot180': 'rot180', 'rot270': 'rot90', 'flip_lr': 'flip_lr',
           'flip_ud': 'flip_ud', 'transpose': 'transpose', 'antitranspose': 'antitranspose'}


@dataclass
class MatrixPuzzle:
    """panels: 3 rows of 3 panels, None for the missing one; choices: candidate answers (optional)"""
    panels: List[List[Optional[Array]]]
    choices: List[Array] = field(default_factory=list)

    def __post_init__(self):
        self.panels = [[None if p is None else np.asarray(p, dtype=int) for p in row] for row in self.panels]
        self.choices = [np.asarray(c, dtype=int) for c in self.choices]
        missing = [(r, c) for r in range(3) for c in range(3) if self.panels[r][c] is None]
        if len(self.panels) != 3 or any(len(row) != 3 for row in self.panels) or len(missing) != 1:
            raise ValueError('a matrix puzzle is 3 x 3 panels with exactly one missing')
        self.missing = missing[0]


def from_grid(grid, separator: int, missing_value: int, choices: Sequence = ()) -> MatrixPuzzle:
    """
    A puzzle drawn as one grid: 3 x 3 panels separated by full lines of separator; the missing panel is the one
    whose every cell holds missing_value
    """
    a = np.asarray(grid, dtype=int)

    def bands(n, lines):
        bounds = [-1] + lines + [n]
        return [(s + 1, e) for s, e in zip(bounds, bounds[1:]) if e - s > 1]

    rows = bands(a.shape[0], [i for i in range(a.shape[0]) if np.all(a[i, :] == separator)])
    cols = bands(a.shape[1], [j for j in range(a.shape[1]) if np.all(a[:, j] == separator)])
    if len(rows) != 3 or len(cols) != 3:
        raise ValueError('expected 3 x 3 panels between separator lines')
    panels = []
    for r0, r1 in rows:
        row = []
        for c0, c1 in cols:
            p = a[r0:r1, c0:c1]
            row.append(None if np.all(p == missing_value) else p.copy())
        panels.append(row)
    return MatrixPuzzle(panels, list(choices))


# ------------------------------------------------------------------ framing

def _oriented(puzzle: MatrixPuzzle, columns: bool) -> Optional[List[List[Optional[Array]]]]:
    """
    The panels as rows (or the columns as rows), with the missing panel's row last and, if the panel is first in
    its row, every row reversed, so that it is the last row's last panel; None for a missing middle panel
    """
    m = [list(row) for row in puzzle.panels]
    if columns:
        m = [list(col) for col in zip(*m)]
    r, c = puzzle.missing if not columns else puzzle.missing[::-1]
    if c == 1:
        return None
    m = [m[i] for i in range(3) if i != r] + [m[r]]
    if c == 0:
        m = [row[::-1] for row in m]
    return m


# ------------------------------------------------------------------ families

def _same(a: Optional[Array], b: Optional[Array]) -> bool:
    return a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)


def distribution(m) -> List[Tuple[str, Array]]:
    """Every row holds the same three panels in some order"""
    def key(p):
        return (p.shape, p.tobytes())
    full = [sorted(key(p) for p in row) for row in m[:2]]
    if full[0] != full[1]:
        return []
    last = [key(p) for p in m[2][:2]]
    rest = list(full[0])
    for k in last:
        if k not in rest:
            return []
        rest.remove(k)
    if len(rest) != 1:
        return []
    panel = next(p for row in m[:2] for p in row if key(p) == rest[0])
    return [('distribution of three', panel.copy())]


@dataclass
class Described:
    color: int
    scale: int
    base: Array             # the object's shape reduced by its scale, as a 0/1 mask
    top_left: Tuple[int, int]
    centre2: Tuple[int, int]  # twice the centre of the bounding box (integers)
    shape: Tuple[int, int]  # the panel's size


def describe(p: Array) -> Optional[Described]:
    """A panel as one object of one colour (None if it is not one)"""
    cells = np.argwhere(p != 0)
    if len(cells) == 0:
        return None
    colors = np.unique(p[p != 0])
    if len(colors) != 1:
        return None
    (r0, c0), (r1, c1) = cells.min(axis=0), cells.max(axis=0)
    mask = (p[r0:r1 + 1, c0:c1 + 1] != 0).astype(int)
    scale = 1
    for k in range(min(mask.shape), 1, -1):
        if mask.shape[0] % k == 0 and mask.shape[1] % k == 0:
            small = mask[::k, ::k]
            if np.array_equal(np.kron(small, np.ones((k, k), int)), mask):
                scale = k
                break
    return Described(int(colors[0]), scale, mask[::scale, ::scale], (int(r0), int(c0)),
                     (int(r0 + r1), int(c0 + c1)), p.shape)


def _draw(d: Described, color, scale, base, place, by_centre) -> Optional[Array]:
    out = np.zeros(d.shape, dtype=int)
    mask = np.kron(base, np.ones((scale, scale), int))
    h, w = mask.shape
    if by_centre:
        if (place[0] - (h - 1)) % 2 or (place[1] - (w - 1)) % 2:
            return None
        r0, c0 = (place[0] - (h - 1)) // 2, (place[1] - (w - 1)) // 2
    else:
        r0, c0 = place
    if r0 < 0 or c0 < 0 or r0 + h > out.shape[0] or c0 + w > out.shape[1]:
        return None
    out[r0:r0 + h, c0:c0 + w] = np.where(mask > 0, color, 0)
    return out


def _value_rules(rows: List[List], equal, numeric: bool = False, mapping: bool = False) -> List[Tuple[str, object]]:
    """
    Rules for one attribute, rows[r][j] its value in row r, place j (rows[2][2] unknown); each consistent rule
    with its predicted value, simplest first
    """
    out = []
    if all(equal(row[1], row[2]) for row in rows[:2]) and equal(rows[2][0], rows[2][1]) \
            and all(equal(row[0], row[1]) for row in rows[:2]):
        out.append(('constant', rows[2][1]))
    if all(equal(rows[0][j], rows[1][j]) for j in range(3)) and all(equal(rows[0][j], rows[2][j]) for j in range(2)):
        out.append(('same in each column', rows[0][2]))
    if numeric:
        steps = [(row[1][0] - row[0][0], row[1][1] - row[0][1]) if isinstance(row[0], tuple) else row[1] - row[0]
                 for row in rows]
        if all(equal(_add(row[1], steps[i]), row[2]) for i, row in enumerate(rows[:2])):
            out.append(('arithmetic progression', _add(rows[2][1], steps[2])))
    if mapping:
        table = {}
        consistent = True
        for row in rows:
            for j in (0, 1):
                if row is rows[2] and j == 1:
                    continue
                if table.setdefault(row[j], row[j + 1]) != row[j + 1]:
                    consistent = False
        if consistent and rows[2][1] in table:
            out.append(('mapping from panel to panel', table[rows[2][1]]))
    sets = [list(row) for row in rows[:2]]
    if all(any(equal(x, y) for y in sets[1]) for x in sets[0]) and all(any(equal(x, y) for y in sets[0]) for x in sets[1]):
        rest = list(sets[0])
        ok = True
        for x in rows[2][:2]:
            hit = next((i for i, y in enumerate(rest) if equal(x, y)), None)
            if hit is None:
                ok = False
                break
            rest.pop(hit)
        if ok and len(rest) == 1:
            out.append(('distribution of three', rest[0]))
    return out


def _add(x, step):
    return (x[0] + step[0], x[1] + step[1]) if isinstance(x, tuple) else x + step


def attributes(m) -> List[Tuple[str, Array]]:
    """Each attribute of a one-object panel follows its own rule (see the module docstring)"""
    ds = [[describe(p) if p is not None else None for p in row] for row in m]
    if any(d is None for row in ds[:2] for d in row) or ds[2][0] is None or ds[2][1] is None:
        return []
    eq = lambda a, b: a == b
    color = _value_rules([[d.color if d else None for d in row] for row in ds], eq, mapping=True)
    scale = _value_rules([[d.scale if d else None for d in row] for row in ds], eq, numeric=True)
    shape = []
    for name, op in SYMMETRIES.items():
        if all(_same(op(row[j].base), row[j + 1].base) for row in ds[:2] for j in (0, 1)) \
                and _same(op(ds[2][0].base), ds[2][1].base):
            shape.append(('each next ' + name if name != 'identity' else 'constant', op(ds[2][1].base)))
    shape += [(n, v) for n, v in _value_rules([[d.base if d else None for d in row] for row in ds], _same)
              if n != 'constant']
    places = []
    for by_centre in (False, True):
        values = [[(d.centre2 if by_centre else d.top_left) if d else None for d in row] for row in ds]
        places += [(n, v, by_centre) for n, v in _value_rules(values, eq, numeric=True)
                   if n != 'distribution of three']
    out = []
    for (cn, c), (sn, s), (bn, b), (pn, p, by_centre) in itertools.product(color, scale, shape, places):
        if s is None or s < 1:
            continue
        panel = _draw(ds[2][1], c, s, b, p, by_centre)
        if panel is not None and not any(_same(panel, q) for _, q in out):
            out.append((f'colour {cn}, scale {sn}, shape {bn}, place {pn}{" (centre)" if by_centre else ""}',
                        panel))
    return out


def _cell_key(a, b):
    """Per cell: (a is background, b is background, a == b), as a small integer code"""
    return (a == 0) * 4 + (b == 0) * 2 + (a == b)


def combination(m) -> List[Tuple[str, Array]]:
    """The third panel cell by cell from the first two, then possibly one turn or mirror (module docstring)"""
    if not all(p.shape == m[0][0].shape for row in m for p in row if p is not None):
        return []
    colors = sorted({int(v) for row in m for p in row if p is not None for v in np.unique(p)} - {0})
    outputs = ['bg', 'first', 'second'] + colors

    def value(o, a, b):
        if o == 'bg':
            return np.zeros(a.shape, dtype=int)
        if o == 'first':
            return a
        if o == 'second':
            return b
        return np.full(a.shape, o)

    square = m[0][0].shape[0] == m[0][0].shape[1]
    out = []
    for name, op in SYMMETRIES.items():
        if not square and name not in ('identity', 'rot180', 'flip_lr', 'flip_ud'):
            continue
        undo = SYMMETRIES[INVERSE[name]]
        allowed: Dict[int, set] = {}
        for row in m[:2]:
            a, b, c = row[0], row[1], undo(row[2])
            keys = _cell_key(a, b)
            for k in np.unique(keys):
                where = keys == k
                fits = {o for o in outputs if np.array_equal(c[where], value(o, a, b)[where])}
                allowed[int(k)] = allowed.get(int(k), fits) & fits
        a, b = m[2][0], m[2][1]
        keys = _cell_key(a, b)
        if any(not allowed.get(int(k)) for k in np.unique(keys)):
            continue
        result = np.zeros(a.shape, dtype=int)
        for k in np.unique(keys):
            o = min(allowed[int(k)], key=outputs.index)     # the simplest output that fits
            result = np.where(keys == k, value(o, a, b), result)
        panel = op(result)
        if not any(_same(panel, q) for _, q in out):
            out.append((f'cell by cell, then {name}', panel))
    return out


def _arc_task(train, test_input):
    return {'train': [{'input': a.tolist(), 'output': b.tolist()} for a, b in train],
            'test': [{'input': test_input.tolist()}]}


def _arc(task) -> List[Tuple[str, Array]]:
    from arc_pattern_solver import find_patterns, rank
    from arc_relation_solver import load_relation_lifts
    import os
    lifts = load_relation_lifts(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learned_relations.json'))
    consistent, _ = find_patterns(task)
    return [(e['name'], np.array(e['preds'][0])) for e in rank(consistent, task, lifts)]


def arc_step(m) -> List[Tuple[str, Array]]:
    steps = [(row[j], row[j + 1]) for row in m[:2] for j in (0, 1)] + [(m[2][0], m[2][1])]
    return [('ARC: ' + n, p) for n, p in _arc(_arc_task(steps, m[2][1]))]


def arc_combination(m) -> List[Tuple[str, Array]]:
    def joined(row):
        if row[0].shape[0] != row[1].shape[0]:
            return None
        return np.hstack([row[0], np.full((row[0].shape[0], 1), SEPARATOR), row[1]])
    inputs = [joined(row) for row in m]
    if any(i is None for i in inputs):
        return []
    return [('ARC: ' + n, p) for n, p in _arc(_arc_task([(inputs[0], m[0][2]), (inputs[1], m[1][2])], inputs[2]))]


FAMILY_FUNCTIONS = {'distribution': distribution, 'attributes': attributes, 'combination': combination,
                    'arc_step': arc_step, 'arc_combination': arc_combination}
SLOW = ('arc_step', 'arc_combination')


def predictions(puzzle: MatrixPuzzle, families: Sequence[str] = FAMILIES,
                use_slow: Optional[bool] = None) -> List[Dict]:
    """
    Every kept rule's prediction, best-ranked first: [{'family', 'along' ('rows' / 'columns'), 'rule', 'panel'}].
    use_slow: whether to try the ARC families (None: only if the other families predict nothing, or nothing
    among the choices)
    """
    out = []

    def run(names):
        for name in names:
            for columns in (False, True):
                m = _oriented(puzzle, columns)
                if m is None:
                    continue
                try:
                    found = FAMILY_FUNCTIONS[name](m)
                except Exception:
                    found = []
                for rule, panel in found:
                    out.append({'family': name, 'along': 'columns' if columns else 'rows', 'rule': rule,
                                'panel': panel})

    run([f for f in families if f not in SLOW])
    slow = [f for f in families if f in SLOW]
    if use_slow is None:
        use_slow = not out or (puzzle.choices and not any(_same(p['panel'], c) for p in out
                                                         for c in puzzle.choices))
    if use_slow:
        run(slow)
    return out


def solve(puzzle: MatrixPuzzle, **kwargs) -> Optional[Dict]:
    """
    The answer: the best-ranked prediction (among the choices, if the puzzle has them), as a predictions()
    entry with, for a puzzle with choices, 'choice' (its index); None if no rule fits
    """
    for p in predictions(puzzle, **kwargs):
        if not puzzle.choices:
            return p
        for i, c in enumerate(puzzle.choices):
            if _same(p['panel'], c):
                return dict(p, choice=i)
    return None
