#!/usr/bin/env python3
"""
ARC Relation Solver - plan step 3

Solves ARC tasks by search over transformations built from the shared grid
vocabulary (grid_vocabulary.py), ordered by what the games taught
(learned_relations.json, from relation_learning.py):

1. Candidates: every transformation below with every parameter value the
   task's training pairs suggest (colours that appear, directions, ...).
2. A candidate is consistent if it reproduces every training output
   exactly; only consistent candidates may answer.
3. Ordering: a candidate is scored by how it changes the vocabulary
   relations from the test input to its predicted output (foreground
   cells against background), summed with the cross-game lifts of those
   changes learned from the games. When several consistent candidates
   disagree on the test input, the highest-scoring one answers.

Transformations (vocabulary relation used):
- geometric: rotations, reflections, transposes (symmetries)
- symmetry_complete: fill background from the mirrored cell (symmetries)
- color_map: cell colour mapping consistent over the pairs (counts)
- fill_enclosed: background cells walled off from the border (enclosed)
- connect: background cells between two same-colour cells (between)
- extend_lines: runs extended along their axis to an obstacle (lines)
- ray: cells of a colour extended in one direction (lines)
- halo: background next to objects (adjacent_cells)
- keep/remove objects by size, recolour largest/smallest (components)
- crop to an object or to all foreground, optionally then geometric (components)
- gravity: cells slid to one side until blocked (touches)
"""

import json
import os
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grid_vocabulary import Grid, adjacent_cells, between, components, enclosed, lines
from relation_learning import RELATIONS, _side_measures

Array = np.ndarray

GEOMETRIC = {
    'rot90': lambda a: np.rot90(a, 1), 'rot180': lambda a: np.rot90(a, 2),
    'rot270': lambda a: np.rot90(a, 3), 'flip_lr': np.fliplr, 'flip_ud': np.flipud,
    'transpose': lambda a: a.T, 'anti_transpose': lambda a: np.rot90(a, 2).T,
}
DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1),
              'NE': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (1, -1)}


class Candidate:
    def __init__(self, name: str, relation: str, fn: Callable[[Array], Optional[Array]], order: int):
        self.name = name
        self.relation = relation
        self.fn = fn
        self.order = order          # position in the fixed listing

    def __call__(self, grid: Array) -> Optional[Array]:
        try:
            out = self.fn(grid)
        except (ValueError, IndexError):
            return None
        if out is None or out.ndim != 2 or out.size == 0:
            return None
        return out


# ---------------------------------------------------------- transformations

def _objects(a: Array, bg: int, diagonal: bool):
    return components(Grid(a, empty=bg), multicolor=True, background=bg, diagonal=diagonal)


def _paint(a: Array, cells, color: int) -> Array:
    out = a.copy()
    for r, c in cells:
        out[r, c] = color
    return out


def fill_enclosed(a, bg, color, walls, diagonal):
    wall_values = [v for v in np.unique(a).tolist() if v != bg] if walls is None else [walls]
    if not wall_values:
        return None
    grid = Grid(a, empty=bg)
    cells = [p for region in enclosed(grid, wall_values, diagonal) for p in region.cells if a[p] == bg]
    return _paint(a, cells, color) if cells else None


def connect(a, bg, color, diagonal):
    grid = Grid(a, empty=bg)
    out = a.copy()
    changed = False
    by_color: Dict[int, List[Tuple[int, int]]] = {}
    for r, c in zip(*np.nonzero(a != bg)):
        by_color.setdefault(int(a[r, c]), []).append((int(r), int(c)))
    for v, cells in by_color.items():
        if len(cells) > 60:
            continue
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                (r1, c1), (r2, c2) = cells[i], cells[j]
                if not diagonal and r1 != r2 and c1 != c2:
                    continue
                path = between(grid, cells[i], cells[j])
                if path and all(a[p] == bg for p in path):
                    for p in path:
                        out[p] = v if color is None else color
                    changed = True
    return out if changed else None


def extend_lines(a, bg):
    out = a.copy()
    changed = False
    grid = Grid(a, empty=bg)
    for ln in lines(grid, min_length=2):
        if ln.value == bg:
            continue
        for direction, end in ((ln.axis[0], ln.cells[0]), (ln.axis[1], ln.cells[-1])):
            pos = grid.step(end, direction)
            while pos is not None and a[pos] == bg:
                out[pos] = ln.value
                changed = True
                pos = grid.step(pos, direction)
    return out if changed else None


def ray(a, bg, color, direction):
    dr, dc = DIRECTIONS[direction]
    out = a.copy()
    changed = False
    h, w = a.shape
    for r, c in zip(*np.nonzero(a == color)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w and a[rr, cc] == bg:
            out[rr, cc] = color
            changed = True
            rr, cc = rr + dr, cc + dc
    return out if changed else None


def halo(a, bg, color, diagonal):
    grid = Grid(a, empty=bg)
    fg = [(int(r), int(c)) for r, c in zip(*np.nonzero(a != bg))]
    cells = [p for p in adjacent_cells(grid, fg, diagonal=diagonal) if a[p] == bg]
    return _paint(a, cells, color) if cells else None


def symmetry_complete(a, bg, axis):
    views = {'lr': [np.fliplr], 'ud': [np.flipud], 'both': [np.fliplr, np.flipud, lambda x: np.rot90(x, 2)],
             'rot180': [lambda x: np.rot90(x, 2)]}
    if axis in ('transpose', 'rot90'):
        if a.shape[0] != a.shape[1]:
            return None
        views['transpose'] = [lambda x: x.T]
        views['rot90'] = [lambda x: np.rot90(x, k) for k in (1, 2, 3)]
    out = a.copy()
    for view in views[axis]:
        mirrored = view(out)
        mask = (out == bg) & (mirrored != bg)
        out[mask] = mirrored[mask]
    return out if not np.array_equal(out, a) else None


def select_objects(a, bg, rule, diagonal):
    objs = _objects(a, bg, diagonal)
    if len(objs) < 2:
        return None
    sizes = [o.size for o in objs]
    target = {'keep_largest': max(sizes), 'keep_smallest': min(sizes),
              'remove_largest': max(sizes), 'remove_smallest': min(sizes)}[rule]
    keep = (lambda o: o.size == target) if rule.startswith('keep') else (lambda o: o.size != target)
    out = np.full_like(a, bg)
    for o in objs:
        if keep(o):
            for p in o.cells:
                out[p] = a[p]
    return out


def recolor_by_size(a, bg, which, color, diagonal):
    objs = _objects(a, bg, diagonal)
    if not objs:
        return None
    sizes = [o.size for o in objs]
    target = max(sizes) if which == 'largest' else min(sizes)
    out = a.copy()
    for o in objs:
        if o.size == target:
            for p in o.cells:
                out[p] = color
    return out


def crop(a, bg, which, diagonal):
    if which == 'foreground':
        rows, cols = np.nonzero(a != bg)
        if rows.size == 0:
            return None
        return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    objs = _objects(a, bg, diagonal)
    if not objs:
        return None
    if which in ('largest', 'smallest'):
        sizes = [o.size for o in objs]
        target = max(sizes) if which == 'largest' else min(sizes)
        chosen = [o for o in objs if o.size == target]
    else:   # the object whose colour set is unique among objects
        sets = [frozenset(int(a[p]) for p in o.cells) for o in objs]
        counts = Counter(sets)
        chosen = [o for o, s in zip(objs, sets) if counts[s] == 1]
    if len(chosen) != 1:
        return None
    r0, c0, r1, c1 = chosen[0].bbox
    return a[r0:r1 + 1, c0:c1 + 1]


def gravity(a, bg, direction):
    k = {'S': 0, 'E': 3, 'N': 2, 'W': 1}[direction]   # rotate so the direction points down
    b = np.rot90(a, k)
    out = np.full_like(b, bg)
    for c in range(b.shape[1]):
        column = [v for v in b[:, c] if v != bg]
        if column:
            out[-len(column):, c] = column
    return np.rot90(out, -k)


def learn_color_map(pairs) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        for x, y in zip(a.ravel().tolist(), b.ravel().tolist()):
            if mapping.setdefault(x, y) != y:
                return None
    if all(k == v for k, v in mapping.items()):
        return None
    return mapping


# --------------------------------------------------------------- candidates

def task_colors(pairs) -> Tuple[int, List[int], List[int]]:
    """Background (most common input colour), colours in inputs, colours new in outputs"""
    counts = Counter()
    for a, _ in pairs:
        counts.update(a.ravel().tolist())
    bg = counts.most_common(1)[0][0]
    in_colors = sorted(set().union(*[set(np.unique(a).tolist()) for a, _ in pairs]) - {bg})
    out_colors = sorted(set().union(*[set(np.unique(b).tolist()) for _, b in pairs]) - {bg})
    new = [c for c in out_colors if c not in in_colors]
    return bg, in_colors, new


def generate_candidates(pairs) -> List[Candidate]:
    bg, in_colors, new_colors = task_colors(pairs)
    paint_colors = new_colors + [c for c in in_colors if c not in new_colors]
    cands: List[Candidate] = []

    def add(name, relation, fn):
        cands.append(Candidate(name, relation, fn, len(cands)))

    for g, f in GEOMETRIC.items():
        add(g, 'symmetries', f)
    cmap = learn_color_map(pairs)
    if cmap:
        add('color_map', 'counts', lambda a, m=cmap: np.vectorize(lambda v: m.get(int(v), int(v)))(a))
    for axis in ('lr', 'ud', 'both', 'rot180', 'transpose', 'rot90'):
        add(f'symmetry_complete:{axis}', 'symmetries', lambda a, x=axis: symmetry_complete(a, bg, x))
    for c in paint_colors:
        for walls in [None] + in_colors:
            for diag in (False, True):
                add(f'fill_enclosed:{c}:walls={walls}:diag={diag}', 'enclosed',
                    lambda a, c=c, w=walls, d=diag: fill_enclosed(a, bg, c, w, d))
    for color in [None] + paint_colors:
        for diag in (False, True):
            add(f'connect:{color}:diag={diag}', 'between', lambda a, c=color, d=diag: connect(a, bg, c, d))
    add('extend_lines', 'lines', lambda a: extend_lines(a, bg))
    for c in in_colors:
        for d in DIRECTIONS:
            add(f'ray:{c}:{d}', 'lines', lambda a, c=c, d=d: ray(a, bg, c, d))
    for c in paint_colors:
        for diag in (False, True):
            add(f'halo:{c}:diag={diag}', 'adjacent_cells', lambda a, c=c, d=diag: halo(a, bg, c, d))
    for rule in ('keep_largest', 'keep_smallest', 'remove_largest', 'remove_smallest'):
        for diag in (False, True):
            add(f'{rule}:diag={diag}', 'components', lambda a, r=rule, d=diag: select_objects(a, bg, r, d))
    for which in ('largest', 'smallest'):
        for c in paint_colors:
            for diag in (False, True):
                add(f'recolor_{which}:{c}:diag={diag}', 'components',
                    lambda a, w=which, c=c, d=diag: recolor_by_size(a, bg, w, c, d))
    for which in ('foreground', 'largest', 'smallest', 'unique_colors'):
        for diag in (False, True):
            if which == 'foreground' and diag:
                continue
            base = lambda a, w=which, d=diag: crop(a, bg, w, d)
            add(f'crop:{which}:diag={diag}', 'components', base)
            for g, f in GEOMETRIC.items():
                add(f'crop:{which}:diag={diag}+{g}', 'components',
                    lambda a, b=base, f=f: (lambda x: None if x is None else f(x))(b(a)))
    for d in ('S', 'N', 'E', 'W'):
        add(f'gravity:{d}', 'touches', lambda a, d=d: gravity(a, bg, d))
    return cands


# ----------------------------------------------------------------- ordering

def load_relation_lifts(path: Optional[str] = None) -> Dict[Tuple[str, str], float]:
    """Cross-game mean lift of each (relation, change) from relation_learning's output"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learned_relations.json')
    data = json.load(open(path))
    return {(e['relation'], e['change']): e['mean_lift'] for e in data['ranking']}


def relation_measures(a: Array, bg: int) -> Dict[str, float]:
    """The step-2 relations of a puzzle grid: foreground cells as 'own', no opponent"""
    side = Grid((a != bg).astype(int), empty=0)
    empty = {(int(r), int(c)) for r, c in zip(*np.nonzero(a == bg))}
    return _side_measures(side, 1, 2, empty)


def game_score(cand_outputs: Sequence[Array], inputs: Sequence[Array], bg: int,
               lifts: Dict[Tuple[str, str], float]) -> float:
    """Mean over the given inputs of the summed learned lifts of the relation changes the candidate makes"""
    total = 0.0
    for a, out in zip(inputs, cand_outputs):
        before, after = relation_measures(a, bg), relation_measures(out, bg)
        for r in RELATIONS:
            d = after[r] - before[r]
            change = '+' if d > 1e-9 else '-' if d < -1e-9 else '0'
            total += lifts.get((r, change), 0.0)
    return total / max(1, len(inputs))


# -------------------------------------------------------------------- solve

def consistent_candidates(task: Dict, lifts) -> Tuple[List[Dict], int]:
    """Consistent candidates with their test predictions and game scores; and how many were tried"""
    pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    bg = task_colors(pairs)[0]
    tests = [np.array(t['input']) for t in task['test']]
    cands = generate_candidates(pairs)
    found = []
    for cand in cands:
        outs = []
        ok = True
        for a, b in pairs:
            out = cand(a)
            if out is None or out.shape != b.shape or not np.array_equal(out, b):
                ok = False
                break
            outs.append(out)
        if not ok:
            continue
        preds = [cand(t) for t in tests]
        if any(p is None for p in preds):
            continue
        found.append({'name': cand.name, 'relation': cand.relation, 'order': cand.order,
                      # Scored on the test inputs: on the training inputs every consistent
                      # candidate makes the same changes (it reproduces the same outputs)
                      'preds': preds, 'score': game_score(preds, tests, bg, lifts)})
    return found, len(cands)


def solves(found_entry: Dict, task: Dict) -> bool:
    return all(np.array_equal(p, np.array(t['output'])) for p, t in zip(found_entry['preds'], task['test']))


def choose(found: List[Dict], ordering: str, rng: Optional[random.Random] = None, k: int = 1) -> List[Dict]:
    """Up to k consistent candidates with distinct predictions, in the given ordering"""
    if ordering == 'game':
        ranked = sorted(found, key=lambda e: (-e['score'], e['order']))
    elif ordering == 'reverse_game':
        ranked = sorted(found, key=lambda e: (e['score'], e['order']))
    elif ordering == 'fixed':
        ranked = sorted(found, key=lambda e: e['order'])
    else:
        ranked = list(found)
        rng.shuffle(ranked)
    out, seen = [], []
    for e in ranked:
        key = tuple(p.tobytes() + bytes(str(p.shape), 'ascii') for p in e['preds'])
        if key in seen:
            continue
        seen.append(key)
        out.append(e)
        if len(out) == k:
            break
    return out
