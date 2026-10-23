#!/usr/bin/env python3
"""
ARC Pattern Solver - find the elements, form patterns from the examples, test them

For a task, every element of every grid is found (arc_elements: objects
under four segmentations, each with its properties). Patterns are then
formed from the training pairs: how the output (the win state) relates to
the elements of the input. A pattern is kept only if it reproduces every
training output exactly; among those, the simplest pattern answers, with
the learned game relations (arc_relation_solver.game_score on the test
prediction) breaking ties, and a second distinct answer as attempt two.

Pattern families:
  A object_actions   each object is kept, deleted or recoloured according to
                     one of its properties (value -> action learned from pairs)
  B select_crop      the output is the region of one object, chosen by a
                     property (largest, smallest, the unique value, a given
                     value), optionally rotated / reflected
  C transform        whole-grid transformations from arc_relation_solver
                     (fill, connect, extend, symmetry, gravity, colour map, ...)
  D move             objects shifted by a vector learned per property value,
                     or slid in a direction until blocked
  E resize           scaling, tiling with per-tile transforms, self-similar tiling
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_elements import RULE_PROPERTIES, SEGMENTATIONS, Obj, background, segment
from arc_relation_solver import GEOMETRIC, game_score, generate_candidates, load_relation_lifts

Array = np.ndarray
TRANSFORMS = dict({'identity': lambda a: a}, **GEOMETRIC)
# Very specific properties cost more (a rule on an exact shape is less general)
PROPERTY_COST = {'pattern': 3, 'shape': 2, 'shape_canon': 1}


@dataclass
class Pattern:
    family: str
    name: str
    complexity: float
    predict: Callable[[Array], Optional[Array]]


def _safe(fn, a):
    try:
        out = fn(a)
    except (ValueError, IndexError, KeyError, TypeError):
        return None
    if out is None or not isinstance(out, np.ndarray) or out.ndim != 2 or out.size == 0:
        return None
    return out


# ------------------------------------------------------------- family A

def _object_action(o: Obj, a: Array, b: Array, bg: int):
    inp = [int(a[p]) for p in o.cells]
    out = [int(b[p]) for p in o.cells]
    if inp == out:
        return 'keep'
    if all(v == bg for v in out):
        return 'delete'
    if len(set(out)) == 1:
        return ('recolor', out[0])
    return None


def _apply_actions(a: Array, objs: List[Obj], actions, bg: int) -> Array:
    out = a.copy()
    for o, act in zip(objs, actions):
        if act == 'delete':
            for p in o.cells:
                out[p] = bg
        elif isinstance(act, tuple):
            for p in o.cells:
                out[p] = act[1]
    return out


def object_action_patterns(pairs, bg) -> List[Pattern]:
    if any(a.shape != b.shape for a, b in pairs):
        return []
    found = []
    for mode in SEGMENTATIONS:
        rows = []           # (props, action)
        ok = True
        for a, b in pairs:
            objs = segment(a, bg, mode)
            covered = np.zeros(a.shape, dtype=bool)
            for o in objs:
                for p in o.cells:
                    covered[p] = True
            if not np.array_equal(a[~covered], b[~covered]):
                ok = False
                break
            for o in objs:
                act = _object_action(o, a, b, bg)
                if act is None:
                    ok = False
                    break
                rows.append((o.props, act))
            if not ok:
                break
        if not ok or not rows or all(act == 'keep' for _, act in rows):
            continue
        for prop in RULE_PROPERTIES:
            mapping = {}
            for props, act in rows:
                v = props[prop]
                if mapping.setdefault(v, act) != act:
                    mapping = None
                    break
            if mapping is None:
                continue

            def predict(t, mode=mode, prop=prop, mapping=mapping):
                objs = segment(t, bg, mode)
                acts = [mapping.get(o.props[prop]) for o in objs]
                if any(x is None for x in acts):
                    return None
                return _apply_actions(t, objs, acts, bg)

            found.append(Pattern('A', f'objects[{mode}] by {prop}: {len(mapping)} rules',
                                 1 + len(mapping) + PROPERTY_COST.get(prop, 0), predict))
    return found


# ------------------------------------------------------------- family B

def _crops(a: Array, o: Obj, bg: int) -> Dict[str, Array]:
    r0, c0, r1, c1 = o.bbox
    box = a[r0:r1 + 1, c0:c1 + 1]
    only = np.where(o.mask(), box, bg)
    return {'bbox': box, 'object': only}


def _select(objs: List[Obj], prop: str, how) -> Optional[Obj]:
    vals = [o.props[prop] for o in objs]
    if how == 'unique':
        counts = Counter(vals)
        chosen = [o for o, v in zip(objs, vals) if counts[v] == 1]
    elif how in ('max', 'min'):
        try:
            target = max(vals) if how == 'max' else min(vals)
        except TypeError:
            return None
        chosen = [o for o, v in zip(objs, vals) if v == target]
    else:
        chosen = [o for o, v in zip(objs, vals) if v == how[1]]
    return chosen[0] if len(chosen) == 1 else None


def select_crop_patterns(pairs, bg) -> List[Pattern]:
    found = []
    for mode in SEGMENTATIONS:
        seg = [segment(a, bg, mode) for a, _ in pairs]
        if any(not objs for objs in seg):
            continue
        for crop_mode in ('bbox', 'object'):
            for tname, tf in TRANSFORMS.items():
                # objects whose crop gives the output, per pair
                match = []
                for (a, b), objs in zip(pairs, seg):
                    ids = {i for i, o in enumerate(objs)
                           if (lambda c: c.shape == b.shape and np.array_equal(c, b))(tf(_crops(a, o, bg)[crop_mode]))}
                    match.append(ids)
                if any(not m for m in match):
                    continue
                for prop in RULE_PROPERTIES:
                    selectors = ['unique', 'max', 'min']
                    first = seg[0][next(iter(match[0]))].props[prop]
                    selectors.append(('eq', first))
                    for how in selectors:
                        good = True
                        for objs, m in zip(seg, match):
                            sel = _select(objs, prop, how)
                            if sel is None or objs.index(sel) not in m:
                                good = False
                                break
                        if not good:
                            continue

                        def predict(t, mode=mode, prop=prop, how=how, cm=crop_mode, tf=tf):
                            objs = segment(t, bg, mode)
                            sel = _select(objs, prop, how) if objs else None
                            return None if sel is None else tf(_crops(t, sel, bg)[cm])

                        cost = 2 + (1 if isinstance(how, tuple) else 0) + (tname != 'identity') \
                            + (crop_mode == 'object') + PROPERTY_COST.get(prop, 0)
                        found.append(Pattern('B', f'crop[{mode},{crop_mode}] {prop} {how if isinstance(how, str) else "=" + str(how[1])[:20]} +{tname}',
                                             cost, predict))
    return found


# ------------------------------------------------------------- family C

def transform_patterns(pairs, bg) -> List[Pattern]:
    return [Pattern('C', c.name, 2 + c.name.count(':') + c.name.count('+'), c)
            for c in generate_candidates(pairs)]


# ------------------------------------------------------------- family D

def _slide(a: Array, objs: List[Obj], moving: List[bool], direction: str, bg: int) -> Array:
    dr, dc = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}[direction]
    out = a.copy()
    order = sorted(range(len(objs)), key=lambda i: -(objs[i].bbox[2] * dr + objs[i].bbox[3] * dc
                                                     + objs[i].bbox[0] * dr + objs[i].bbox[1] * dc))
    h, w = a.shape
    for i in order:
        if not moving[i]:
            continue
        cells = objs[i].cells
        colors = {p: int(out[p]) for p in cells}
        for p in cells:
            out[p] = bg
        shift = 0
        while True:
            nxt = [(r + dr * (shift + 1), c + dc * (shift + 1)) for r, c in cells]
            if any(not (0 <= r < h and 0 <= c < w) or out[r, c] != bg for r, c in nxt):
                break
            shift += 1
        for (r, c), v in colors.items():
            out[r + dr * shift, c + dc * shift] = v
    return out


def move_patterns(pairs, bg) -> List[Pattern]:
    if any(a.shape != b.shape for a, b in pairs):
        return []
    found = []
    for mode in SEGMENTATIONS:
        seg = [segment(a, bg, mode) for a, _ in pairs]
        if any(not objs for objs in seg):
            continue
        # Slide until blocked: all objects, or those with one property value
        selectors = [('all', None)]
        for prop in ('color', 'is_largest', 'is_smallest', 'unique_color', 'unique_shape',
                     'touches_border', 'size'):
            for v in sorted({o.props[prop] for objs in seg for o in objs}, key=str):
                selectors.append((prop, v))
        for direction in ('N', 'S', 'E', 'W'):
            for prop, v in selectors:
                sel = (lambda o, prop=prop, v=v: True if prop == 'all' else o.props[prop] == v)
                if all(np.array_equal(_slide(a, objs, [sel(o) for o in objs], direction, bg), b)
                       for (a, b), objs in zip(pairs, seg)):
                    def predict(t, mode=mode, sel=sel, d=direction):
                        objs = segment(t, bg, mode)
                        return _slide(t, objs, [sel(o) for o in objs], d, bg)
                    found.append(Pattern('D', f'slide[{mode}] {direction} {prop}={v}',
                                         3 + (prop != 'all'), predict))
        # Constant shift per property value
        for prop in ('color', 'size', 'shape_canon', 'is_largest', 'unique_color', 'all'):
            shifts: Dict[object, set] = {}
            ok = True
            for (a, b), objs in zip(pairs, seg):
                h, w = a.shape
                for o in objs:
                    key = 'all' if prop == 'all' else o.props[prop]
                    # Shifts that put the object's first cell on a cell of its colour
                    r0, c0 = min(o.cells)
                    starts = zip(*np.nonzero(b == a[r0, c0]))
                    cand = {(int(r) - r0, int(c) - c0) for r, c in starts
                            if all(0 <= rr + int(r) - r0 < h and 0 <= cc + int(c) - c0 < w
                                   and b[rr + int(r) - r0, cc + int(c) - c0] == a[rr, cc]
                                   for rr, cc in o.cells)}
                    shifts[key] = shifts.get(key, cand) & cand
                    if not shifts[key]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            chosen = {k: min(v, key=lambda d: (abs(d[0]) + abs(d[1]), d)) for k, v in shifts.items()}
            if all(d == (0, 0) for d in chosen.values()):
                continue

            def render(t, objs, chosen=chosen, prop=prop):
                out = np.full_like(t, bg)
                h, w = t.shape
                for o in objs:
                    key = 'all' if prop == 'all' else o.props[prop]
                    if key not in chosen:
                        return None
                    dr, dc = chosen[key]
                    for r, c in o.cells:
                        if not (0 <= r + dr < h and 0 <= c + dc < w):
                            return None
                        out[r + dr, c + dc] = t[r, c]
                return out

            if all((lambda o: o is not None and np.array_equal(o, b))(render(a, objs))
                   for (a, b), objs in zip(pairs, seg)):
                def predict(t, mode=mode, render=render):
                    return render(t, segment(t, bg, mode))
                found.append(Pattern('D', f'shift[{mode}] by {prop}: {len(chosen)} vectors',
                                     3 + len(chosen), predict))
    return found


# ------------------------------------------------------------- family E

def resize_patterns(pairs, bg) -> List[Pattern]:
    found = []
    a0, b0 = pairs[0]
    # Upscale / downscale by integer factors
    if b0.shape[0] % a0.shape[0] == 0 and b0.shape[1] % a0.shape[1] == 0:
        k, m = b0.shape[0] // a0.shape[0], b0.shape[1] // a0.shape[1]
        if (k, m) != (1, 1):
            up = lambda a, k=k, m=m: np.kron(a, np.ones((k, m), dtype=a.dtype))
            if all(np.array_equal(up(a), b) for a, b in pairs):
                found.append(Pattern('E', f'upscale {k}x{m}', 2, up))
            # Tiling with a transform per tile
            per_tile = {}
            ok = True
            for a, b in pairs:
                h, w = a.shape
                if b.shape != (h * k, w * m):
                    ok = False
                    break
                for i in range(k):
                    for j in range(m):
                        tile = b[i * h:(i + 1) * h, j * w:(j + 1) * w]
                        ts = {n for n, f in TRANSFORMS.items()
                              if f(a).shape == tile.shape and np.array_equal(f(a), tile)}
                        per_tile[(i, j)] = per_tile.get((i, j), ts) & ts
                        if not per_tile[(i, j)]:
                            ok = False
                if not ok:
                    break
            if ok:
                plan = {ij: sorted(ts)[0] if 'identity' not in ts else 'identity' for ij, ts in per_tile.items()}

                def tile_fn(a, plan=plan, k=k, m=m):
                    h, w = a.shape
                    out = np.zeros((h * k, w * m), dtype=a.dtype)
                    for (i, j), n in plan.items():
                        t = TRANSFORMS[n](a)
                        if t.shape != (h, w):
                            return None
                        out[i * h:(i + 1) * h, j * w:(j + 1) * w] = t
                    return out
                found.append(Pattern('E', f'tile {k}x{m}', 2 + len(set(plan.values())), tile_fn))
    if a0.shape[0] % b0.shape[0] == 0 and a0.shape[1] % b0.shape[1] == 0 and a0.shape != b0.shape:
        k, m = a0.shape[0] // b0.shape[0], a0.shape[1] // b0.shape[1]
        down = lambda a, k=k, m=m: a[::k, ::m] if a.shape[0] % k == 0 and a.shape[1] % m == 0 else None
        if all((lambda o: o is not None and np.array_equal(o, b))(down(a)) for a, b in pairs):
            found.append(Pattern('E', f'downscale {k}x{m}', 2, down))
    # Self-similar tiling: block (i, j) is the input where input[i, j] is (or is not) background
    if b0.shape == (a0.shape[0] ** 2, a0.shape[1] ** 2):
        for invert in (False, True):
            def fractal(a, invert=invert):
                h, w = a.shape
                out = np.full((h * h, w * w), bg, dtype=a.dtype)
                for i in range(h):
                    for j in range(w):
                        if (a[i, j] != bg) != invert:
                            out[i * h:(i + 1) * h, j * w:(j + 1) * w] = a
                return out
            if all(np.array_equal(fractal(a), b) for a, b in pairs):
                found.append(Pattern('E', f'self-tile invert={invert}', 3, fractal))
    return found


# --------------------------------------------------------------- solve

FAMILIES = {'A': object_action_patterns, 'B': select_crop_patterns, 'C': transform_patterns,
            'D': move_patterns, 'E': resize_patterns}


def find_patterns(task: Dict, families: Sequence[str] = tuple(FAMILIES)) -> Tuple[List[Dict], int]:
    """Patterns consistent with every training pair, with their test predictions"""
    pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    tests = [np.array(t['input']) for t in task['test']]
    bg = background([a for a, _ in pairs])
    proposed = 0
    consistent = []
    for fam in families:
        for pat in FAMILIES[fam](pairs, bg):
            proposed += 1
            if not all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(pat.predict, a))
                       for a, b in pairs):
                continue
            preds = [_safe(pat.predict, t) for t in tests]
            if any(p is None for p in preds):
                continue
            consistent.append({'family': pat.family, 'name': pat.name, 'complexity': pat.complexity,
                               'preds': preds})
    return consistent, proposed


def rank(consistent: List[Dict], task: Dict, lifts) -> List[Dict]:
    """Distinct predictions, simplest pattern first, learned game relations breaking ties"""
    pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    tests = [np.array(t['input']) for t in task['test']]
    bg = background([a for a, _ in pairs])
    for e in consistent:
        same_shape = all(p.shape == t.shape for p, t in zip(e['preds'], tests))
        e['game_score'] = game_score(e['preds'], tests, bg, lifts) if same_shape else 0.0
    ordered = sorted(consistent, key=lambda e: (e['complexity'], -e['game_score']))
    out, seen = [], set()
    for e in ordered:
        key = tuple(p.tobytes() + bytes(str(p.shape), 'ascii') for p in e['preds'])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def solve(task: Dict, lifts=None, attempts: int = 2) -> List[List[Array]]:
    """Up to `attempts` predictions (each a list of test outputs)"""
    lifts = load_relation_lifts() if lifts is None else lifts
    consistent, _ = find_patterns(task)
    return [e['preds'] for e in rank(consistent, task, lifts)[:attempts]]
