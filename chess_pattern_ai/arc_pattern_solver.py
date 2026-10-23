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
  A object_actions   each object is kept, deleted, recoloured or takes a touching
                     element's colour, according to its properties
  B select_crop      the output is the region of one object, chosen by a
                     property (largest, smallest, the unique value, a given
                     value), optionally rotated / reflected
  C transform        whole-grid transformations from arc_relation_solver
                     (fill, connect, extend, symmetry, gravity, colour map, ...)
  D move             objects shifted by a vector learned per property value,
                     or slid in a direction until blocked
  E resize           scaling, tiling with per-tile transforms, self-similar tiling
  F relations        objects approach the nearest object in line until blocked;
                     a template object is stamped onto every marker object
  G combinations     a structural first step (rotate / reflect, crop to the
                     foreground or a chosen object, gravity, keep / remove
                     objects) followed by any single pattern on its result;
                     searched only when no single pattern fits
  H count / compare  the output draws a count (objects, cells of a colour,
                     colours, shapes, sizes, holes) as a row, column or filled
                     cells; or is filled with the colour that has the most /
                     fewest cells, objects or largest object
  I build            outputs built from scratch: a constant output; parts of
                     the input (cut at separator lines or into halves /
                     thirds) combined by AND / OR / XOR / NOR / difference or
                     overlaid; or every output cell computed from its input
                     neighbourhood by a rule learned from the examples

Conditions: family A maps one property, or two at once, to an action (keep,
delete, recolour, or adopt the colour of the one element an object
touches); family B selects by one property, or filters by one property's
value and then selects by another.
"""

import itertools
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

def _touching_colors(o: Obj, a: Array, bg: int) -> set:
    """Colours of non-background cells next to the object but not part of it"""
    h, w = a.shape
    out = set()
    for r, c in o.cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in o.cells and a[rr, cc] != bg:
                out.add(int(a[rr, cc]))
    return out


def _object_actions(o: Obj, a: Array, b: Array, bg: int) -> Optional[set]:
    """Every action that explains what happened to the object (None: no simple action does)"""
    inp = [int(a[p]) for p in o.cells]
    out = [int(b[p]) for p in o.cells]
    if inp == out:
        return {'keep'}
    if all(v == bg for v in out):
        return {'delete'}
    if len(set(out)) == 1:
        acts = {('recolor', out[0])}
        if _touching_colors(o, a, bg) == {out[0]}:
            acts.add('adopt')        # took the colour of the one element it touches
        return acts
    return None


# Preferred explanation when several fit every object with a value
_ACTION_ORDER = {'keep': 0, 'delete': 1, 'adopt': 3}


def _apply_actions(a: Array, objs: List[Obj], actions, bg: int) -> Optional[Array]:
    out = a.copy()
    for o, act in zip(objs, actions):
        if act == 'delete':
            for p in o.cells:
                out[p] = bg
        elif act == 'adopt':
            touching = _touching_colors(o, a, bg)
            if len(touching) != 1:
                return None
            color = next(iter(touching))
            for p in o.cells:
                out[p] = color
        elif isinstance(act, tuple):
            for p in o.cells:
                out[p] = act[1]
    return out


def _learn_mapping(rows, key) -> Optional[Dict]:
    """value -> action consistent over all objects (None if some value has no common action)"""
    allowed: Dict[object, set] = {}
    for props, acts in rows:
        v = key(props)
        allowed[v] = allowed.get(v, acts) & acts
        if not allowed[v]:
            return None
    return {v: min(acts, key=lambda x: (_ACTION_ORDER.get(x, 2), str(x))) for v, acts in allowed.items()}


def object_action_patterns(pairs, bg) -> List[Pattern]:
    if any(a.shape != b.shape for a, b in pairs):
        return []
    found = []
    for mode in SEGMENTATIONS:
        rows = []           # (props, acceptable actions)
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
                acts = _object_actions(o, a, b, bg)
                if acts is None:
                    ok = False
                    break
                rows.append((o.props, acts))
            if not ok:
                break
        if not ok or not rows or all(acts == {'keep'} for _, acts in rows):
            continue

        def make(key, mapping, mode=mode):
            def predict(t):
                objs = segment(t, bg, mode)
                acts = [mapping.get(key(o.props)) for o in objs]
                if any(x is None for x in acts):
                    return None
                return _apply_actions(t, objs, acts, bg)
            return predict

        single = 0
        for prop in RULE_PROPERTIES:
            key = lambda props, prop=prop: props[prop]
            mapping = _learn_mapping(rows, key)
            if mapping is None:
                continue
            single += 1
            found.append(Pattern('A', f'objects[{mode}] by {prop}: {len(mapping)} rules',
                                 1 + len(mapping) + PROPERTY_COST.get(prop, 0), make(key, mapping)))
        if single:
            continue
        # Conditions on two properties at once (only when no single property explains it)
        props = [p for p in RULE_PROPERTIES if p not in ('pattern', 'shape')]
        for i, p1 in enumerate(props):
            for p2 in props[i + 1:]:
                key = lambda pr, p1=p1, p2=p2: (pr[p1], pr[p2])
                mapping = _learn_mapping(rows, key)
                if mapping is None:
                    continue
                found.append(Pattern('A', f'objects[{mode}] by {p1}+{p2}: {len(mapping)} rules',
                                     3 + len(mapping) + PROPERTY_COST.get(p1, 0) + PROPERTY_COST.get(p2, 0),
                                     make(key, mapping)))
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
                before = len(found)
                for prop in RULE_PROPERTIES:
                    selectors = ['unique', 'max', 'min']
                    first = seg[0][next(iter(match[0]))].props[prop]
                    selectors.append(('eq', first))
                    for how in selectors:
                        if _selects(seg, match, None, prop, how):
                            cost = 2 + (1 if isinstance(how, tuple) else 0) + (tname != 'identity') \
                                + (crop_mode == 'object') + PROPERTY_COST.get(prop, 0)
                            found.append(Pattern('B', f'crop[{mode},{crop_mode}] {prop} {_how(how)} +{tname}',
                                                 cost, _crop_predict(bg, mode, None, prop, how, crop_mode, tf)))
                if len(found) > before:
                    continue
                # Two properties: keep the objects with one property value, then select by another
                props = [p for p in RULE_PROPERTIES if p not in ('pattern', 'shape')]
                for fprop in props:
                    fval = seg[0][next(iter(match[0]))].props[fprop]
                    for prop in props:
                        if prop == fprop:
                            continue
                        for how in ('unique', 'max', 'min'):
                            if _selects(seg, match, (fprop, fval), prop, how):
                                cost = 4 + (tname != 'identity') + (crop_mode == 'object') \
                                    + PROPERTY_COST.get(prop, 0) + PROPERTY_COST.get(fprop, 0)
                                found.append(Pattern('B', f'crop[{mode},{crop_mode}] {fprop}={str(fval)[:20]} then {prop} {how} +{tname}',
                                                     cost, _crop_predict(bg, mode, (fprop, fval), prop, how, crop_mode, tf)))
    return found


def _how(how) -> str:
    return how if isinstance(how, str) else '=' + str(how[1])[:20]


def _filtered_select(objs: List[Obj], filt, prop: str, how) -> Optional[Obj]:
    if filt is not None:
        objs = [o for o in objs if o.props[filt[0]] == filt[1]]
    return _select(objs, prop, how) if objs else None


def _selects(seg, match, filt, prop, how) -> bool:
    """Whether the selector picks, in every pair, an object whose crop gives the output"""
    for objs, m in zip(seg, match):
        sel = _filtered_select(objs, filt, prop, how)
        if sel is None or objs.index(sel) not in m:
            return False
    return True


def _crop_predict(bg, mode, filt, prop, how, crop_mode, tf):
    def predict(t):
        sel = _filtered_select(segment(t, bg, mode), filt, prop, how)
        return None if sel is None else tf(_crops(t, sel, bg)[crop_mode])
    return predict


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


# ------------------------------------------------------------- family F

def _approach(a: Array, objs: List[Obj], moving: List[bool], bg: int) -> Optional[Array]:
    """Each moving object slides toward the nearest still object in line with it until blocked"""
    out = a.copy()
    for i, o in enumerate(objs):
        if not moving[i]:
            continue
        r0, c0, r1, c1 = o.bbox
        best = None
        for j, t in enumerate(objs):
            if j == i or moving[j]:
                continue
            R0, C0, R1, C1 = t.bbox
            if r0 <= R1 and R0 <= r1:
                d, dist = ('E', C0 - c1) if C0 > c1 else ('W', c0 - C1) if C1 < c0 else (None, 0)
            elif c0 <= C1 and C0 <= c1:
                d, dist = ('S', R0 - r1) if R0 > r1 else ('N', r0 - R1) if R1 < r0 else (None, 0)
            else:
                continue
            if d and (best is None or dist < best[1]):
                best = (d, dist)
        if best is not None:
            out = _slide(out, [o], [True], best[0], bg)
    return out


def _stamp(a: Array, template: Obj, markers: List[Obj], align: str) -> Optional[Array]:
    """Copy the template onto every marker, aligned by the marker's colour or by centre"""
    cells = {p: int(a[p]) for p in template.cells}
    tr0, tc0, tr1, tc1 = template.bbox
    out = a.copy()
    h, w = a.shape
    for m in markers:
        mr0, mc0, mr1, mc1 = m.bbox
        if align == 'color':
            anchors = [p for p, v in cells.items() if v == m.props['color']]
            if len(anchors) != 1:
                return None
            dr, dc = mr0 - anchors[0][0], mc0 - anchors[0][1]
        else:
            if (mr0 + mr1 - tr0 - tr1) % 2 or (mc0 + mc1 - tc0 - tc1) % 2:
                return None
            dr, dc = (mr0 + mr1 - tr0 - tr1) // 2, (mc0 + mc1 - tc0 - tc1) // 2
        for (r, c), v in cells.items():
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            out[rr, cc] = v
    return out


def relation_patterns(pairs, bg) -> List[Pattern]:
    """Patterns that relate elements to each other: approach until touching, stamp at markers"""
    if any(a.shape != b.shape for a, b in pairs):
        return []
    found = []
    for mode in SEGMENTATIONS:
        seg = [segment(a, bg, mode) for a, _ in pairs]
        if any(len(objs) < 2 for objs in seg):
            continue
        # Approach: objects with a property value move toward the others
        for prop in ('color', 'is_largest', 'is_smallest', 'unique_color', 'size', 'n_colors'):
            for v in sorted({o.props[prop] for objs in seg for o in objs}, key=str):
                sel = lambda o, prop=prop, v=v: o.props[prop] == v
                if all((lambda o: o is not None and np.array_equal(o, b))(
                        _approach(a, objs, [sel(o) for o in objs], bg))
                       for (a, b), objs in zip(pairs, seg)):
                    def predict(t, mode=mode, sel=sel):
                        objs = segment(t, bg, mode)
                        return _approach(t, objs, [sel(o) for o in objs], bg)
                    found.append(Pattern('F', f'approach[{mode}] {prop}={v}', 4, predict))
        # Stamp: a template chosen by a property is copied onto every other object
        for prop in ('size', 'n_colors', 'unique_shape', 'unique_color', 'height', 'width'):
            for how in ('max', 'min', 'unique'):
                for align in ('color', 'center'):
                    def run(a, objs, prop=prop, how=how, align=align):
                        template = _select(objs, prop, how)
                        if template is None:
                            return None
                        return _stamp(a, template, [o for o in objs if o is not template], align)
                    if all((lambda o: o is not None and np.array_equal(o, b))(run(a, objs))
                           for (a, b), objs in zip(pairs, seg)):
                        def predict(t, mode=mode, run=run):
                            return run(t, segment(t, bg, mode))
                        found.append(Pattern('F', f'stamp[{mode}] template {prop} {how} align {align}', 5, predict))
    return found


# ------------------------------------------------------------- family H

def _quantities(pairs, bg) -> List[Tuple[str, Callable[[Array], int]]]:
    """Things that can be counted in a grid"""
    colors = sorted({int(v) for a, _ in pairs for v in np.unique(a)} - {bg})
    qs = [('distinct colours', lambda a: len(set(np.unique(a).tolist()) - {bg}))]
    for mode in SEGMENTATIONS:
        qs.append((f'objects[{mode}]', lambda a, m=mode: len(segment(a, bg, m))))
        qs.append((f'largest object[{mode}]',
                   lambda a, m=mode: max((o.props['size'] for o in segment(a, bg, m)), default=0)))
        qs.append((f'smallest object[{mode}]',
                   lambda a, m=mode: min((o.props['size'] for o in segment(a, bg, m)), default=0)))
        qs.append((f'distinct shapes[{mode}]',
                   lambda a, m=mode: len({o.props['shape_canon'] for o in segment(a, bg, m)})))
        qs.append((f'holes[{mode}]', lambda a, m=mode: sum(o.props['holes'] for o in segment(a, bg, m))))
    for c in colors:
        qs.append((f'cells of {c}', lambda a, c=c: int(np.sum(a == c))))
        for mode in ('color4', 'color8'):
            qs.append((f'objects of {c}[{mode}]',
                       lambda a, c=c, m=mode: sum(1 for o in segment(a, bg, m) if o.props['color'] == c)))
    return qs


def _render_count(n: int, how: str, color: int, out_bg: int, shape) -> Optional[Array]:
    if n <= 0 or n > 900:
        return None
    if how == 'row':
        return np.full((1, n), color, dtype=int)
    if how == 'column':
        return np.full((n, 1), color, dtype=int)
    out = np.full(shape, out_bg, dtype=int)
    if n > out.size:
        return None
    flat = out.ravel() if how == 'fill rows' else out.T.ravel()
    flat[:n] = color
    return flat.reshape(out.shape) if how == 'fill rows' else flat.reshape(out.T.shape).T


def count_patterns(pairs, bg) -> List[Pattern]:
    found = []
    outs = [b for _, b in pairs]
    out_bg = background(outs)
    out_colors = sorted({int(v) for b in outs for v in np.unique(b)} - {out_bg}) or [out_bg]
    fixed_shape = outs[0].shape if all(b.shape == outs[0].shape for b in outs) else None
    if all(len(set(np.unique(b).tolist()) - {out_bg}) <= 1 for b in outs):
        renders = ['row', 'column'] + (['fill rows', 'fill columns'] if fixed_shape else [])
        for qname, q in _quantities(pairs, bg):
            counts = [q(a) for a, _ in pairs]
            for how in renders:
                for color in out_colors:
                    if not all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(
                            _render_count(n, how, color, out_bg, fixed_shape)) for n, (_, b) in zip(counts, pairs)):
                        continue
                    fn = lambda a, q=q, how=how, color=color: _render_count(q(a), how, color, out_bg, fixed_shape)
                    found.append(Pattern('H', f'count {qname} as {how} of {color}', 3, fn))
    # Comparing: fill with the colour that wins a comparison between colours
    if all(len(np.unique(b)) == 1 for b in outs):
        shape_rules = [('input shape', lambda a: a.shape)]
        if fixed_shape:
            shape_rules.append((f'{fixed_shape}', lambda a: fixed_shape))
        measures = {
            'cells': lambda a, c: int(np.sum(a == c)),
            'objects': lambda a, c: sum(1 for o in segment(a, bg, 'color8') if o.props['color'] == c),
            'largest object': lambda a, c: max((o.props['size'] for o in segment(a, bg, 'color8')
                                                if o.props['color'] == c), default=0),
        }
        for mname, m in measures.items():
            for how in ('most', 'fewest'):
                def winner(a, m=m, how=how):
                    colors = [int(v) for v in np.unique(a) if v != bg]
                    if not colors:
                        return None
                    vals = [(m(a, c), c) for c in colors]
                    best = max(v for v, _ in vals) if how == 'most' else min(v for v, _ in vals)
                    win = [c for v, c in vals if v == best]
                    return win[0] if len(win) == 1 else None
                for sname, shape_fn in shape_rules:
                    fn = lambda a, w=winner, sf=shape_fn: (lambda c: None if c is None else np.full(sf(a), c, dtype=int))(w(a))
                    if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(fn, a))
                           for a, b in pairs):
                        found.append(Pattern('H', f'colour with {how} {mname}, shape {sname}', 3, fn))
    return found


# ------------------------------------------------------------- family I

def _split(a: Array) -> List[Tuple[str, List[Array]]]:
    """Ways to cut a grid into equal parts: at separator lines, or into equal halves / thirds"""
    out = []
    h, w = a.shape
    for axis in (0, 1):
        uniform = {}
        for i in range(a.shape[axis]):
            line = a[i, :] if axis == 0 else a[:, i]
            if len(np.unique(line)) == 1:
                uniform.setdefault(int(line[0]), []).append(i)
        for color, lines in sorted(uniform.items()):
            bounds = [-1] + lines + [a.shape[axis]]
            parts = [a[s + 1:e, :] if axis == 0 else a[:, s + 1:e]
                     for s, e in zip(bounds, bounds[1:]) if e - s > 1]
            if len(parts) >= 2 and all(p.shape == parts[0].shape for p in parts):
                out.append((f'separators of {color} axis {axis}', parts))
        for k in (2, 3):
            n = a.shape[axis]
            if n % k == 0 and n // k > 0:
                size = n // k
                parts = [a[i * size:(i + 1) * size, :] if axis == 0 else a[:, i * size:(i + 1) * size]
                         for i in range(k)]
                out.append((f'{k} parts axis {axis}', parts))
    return out


_BOOL_OPS = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor,
             'nor': lambda x, y: ~(x | y), 'a_not_b': lambda x, y: x & ~y, 'b_not_a': lambda x, y: ~x & y}


def _local_keys():
    def nb4(a, r, c):
        h, w = a.shape
        return tuple(int(a[r + dr, c + dc]) if 0 <= r + dr < h and 0 <= c + dc < w else -1
                     for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    def nb8(a, r, c):
        h, w = a.shape
        return tuple(int(a[r + dr, c + dc]) if 0 <= r + dr < h and 0 <= c + dc < w else -1
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1))

    def count8(a, r, c, bg):
        return sum(1 for v in nb8(a, r, c) if v not in (-1, bg)) - (a[r, c] != bg)

    return {
        'cell + 4 neighbours': lambda a, r, c, bg: (int(a[r, c]),) + nb4(a, r, c),
        '3x3 neighbourhood': lambda a, r, c, bg: nb8(a, r, c),
        'cell + count of 8 neighbours': lambda a, r, c, bg: (int(a[r, c]), count8(a, r, c, bg)),
        'cell + row/column colours': lambda a, r, c, bg: (int(a[r, c]),
                                                          frozenset(np.unique(a[r, :]).tolist()) - {bg},
                                                          frozenset(np.unique(a[:, c]).tolist()) - {bg}),
    }


def build_patterns(pairs, bg) -> List[Pattern]:
    found = []
    outs = [b for _, b in pairs]
    out_bg = background(outs)
    # Constant output
    if all(b.shape == outs[0].shape and np.array_equal(b, outs[0]) for b in outs) and len(pairs) > 1:
        found.append(Pattern('I', 'constant output', 2, lambda a, c=outs[0].copy(): c.copy()))
    # Split into parts and combine them cell by cell
    splits0 = _split(pairs[0][0])
    for sname, parts0 in splits0:
        if parts0[0].shape != outs[0].shape:
            continue

        def parts_of(a, sname=sname):
            for n, ps in _split(a):
                if n == sname:
                    return ps
            return None

        if len(parts0) == 2:
            colors = sorted({int(v) for b in outs for v in np.unique(b)} - {out_bg})
            for opname, op in _BOOL_OPS.items():
                for color in colors[:3]:
                    def fn(a, op=op, color=color, parts_of=parts_of):
                        ps = parts_of(a)
                        if ps is None or len(ps) != 2:
                            return None
                        pbg = background(ps)
                        m = op(ps[0] != pbg, ps[1] != pbg)
                        return np.where(m, color, out_bg).astype(int)
                    if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(fn, a))
                           for a, b in pairs):
                        found.append(Pattern('I', f'{sname}: {opname} -> {color}', 3, fn))
        for order in (itertools.permutations(range(len(parts0))) if len(parts0) <= 4 else ()):
            def overlay(a, order=order, parts_of=parts_of):
                ps = parts_of(a)
                if ps is None or len(ps) != len(order):
                    return None
                pbg = background(ps)
                out = np.full(ps[0].shape, pbg, dtype=int)
                for i in reversed(order):
                    out = np.where(ps[i] != pbg, ps[i], out)
                return out
            if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(overlay, a))
                   for a, b in pairs):
                found.append(Pattern('I', f'{sname}: overlay order {order}', 3, overlay))
    # Local rules: each output cell from the input cell's neighbourhood
    if all(a.shape == b.shape for a, b in pairs):
        for kname, key in _local_keys().items():
            mapping = {}
            ok = True
            for a, b in pairs:
                for r in range(a.shape[0]):
                    for c in range(a.shape[1]):
                        k = key(a, r, c, bg)
                        if mapping.setdefault(k, int(b[r, c])) != int(b[r, c]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            changes = sum(1 for k, v in mapping.items()
                          if (k[4] if kname == '3x3 neighbourhood' else k[0]) != v)
            if changes == 0:
                continue

            def local(a, key=key, mapping=mapping):
                out = np.empty_like(a)
                for r in range(a.shape[0]):
                    for c in range(a.shape[1]):
                        v = mapping.get(key(a, r, c, bg))
                        if v is None:
                            return None
                        out[r, c] = v
                return out
            found.append(Pattern('I', f'local rule: {kname} ({changes} changing entries)', 3 + changes, local))
    return found


# ------------------------------------------------------------- family G

def _structural_steps(bg) -> List[Tuple[str, Callable, float]]:
    """First steps of a combination: change the grid's frame or keep part of it"""
    from arc_relation_solver import crop, gravity, select_objects
    steps = [(g, f, 1) for g, f in GEOMETRIC.items()]
    steps.append(('crop foreground', lambda a: crop(a, bg, 'foreground', False), 1))
    for d in ('S', 'N', 'E', 'W'):
        steps.append((f'gravity {d}', lambda a, d=d: gravity(a, bg, d), 1))
    for rule in ('keep_largest', 'keep_smallest', 'remove_largest', 'remove_smallest'):
        steps.append((rule, lambda a, r=rule: select_objects(a, bg, r, False), 1))
    for mode in ('multi8', 'color4', 'multi4'):
        for prop in ('size', 'n_colors', 'unique_color', 'unique_shape', 'holes', 'color_count'):
            for how in ('max', 'min', 'unique'):
                def step(a, mode=mode, prop=prop, how=how):
                    sel = _select(segment(a, bg, mode), prop, how)
                    return None if sel is None else _crops(a, sel, bg)['bbox']
                steps.append((f'crop[{mode}] {prop} {how}', step, 2))
    return steps


_LITE = ('color_map', 'fill_enclosed', 'symmetry_complete', 'gravity', 'extend_lines', 'halo',
         'connect') + tuple(GEOMETRIC)


def _lite_transforms(pairs, bg) -> List[Pattern]:
    return [p for p in transform_patterns(pairs, bg) if p.name.split(':')[0] in _LITE]


def combination_patterns(pairs, bg, deadline: float) -> List[Pattern]:
    """A structural first step followed by any single pattern found on its result"""
    import time
    found = []
    second = (object_action_patterns, select_crop_patterns, _lite_transforms, move_patterns,
              resize_patterns, relation_patterns, count_patterns, build_patterns)
    for name1, step1, cost1 in _structural_steps(bg):
        if time.time() > deadline:
            break
        mids = [_safe(step1, a) for a, _ in pairs]
        if any(m is None for m in mids) or all(m.shape == a.shape and np.array_equal(m, a)
                                                for m, (a, _) in zip(mids, pairs)):
            continue
        mid_pairs = [(m, b) for m, (_, b) in zip(mids, pairs)]
        for family in second:
            if time.time() > deadline:
                break
            for pat in family(mid_pairs, bg):
                if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(
                        _safe(pat.predict, m)) for m, b in mid_pairs):
                    def predict(t, s1=step1, p2=pat.predict):
                        m = _safe(s1, t)
                        return None if m is None else p2(m)
                    found.append(Pattern('G', f'{name1} -> {pat.family}:{pat.name}',
                                         cost1 + pat.complexity + 1, predict))
    return found


# --------------------------------------------------------------- solve

FAMILIES = {'A': object_action_patterns, 'B': select_crop_patterns, 'C': transform_patterns,
            'D': move_patterns, 'E': resize_patterns, 'F': relation_patterns,
            'H': count_patterns, 'I': build_patterns}
COMBINATION_SECONDS = 60     # time budget for combinations, used only when no single pattern fits


def find_patterns(task: Dict, families: Sequence[str] = tuple(FAMILIES),
                  combinations: bool = True) -> Tuple[List[Dict], int]:
    """
    Patterns consistent with every training pair, with their test predictions.

    Combinations (family G) are searched only when no single pattern fits,
    within COMBINATION_SECONDS.
    """
    import time
    pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    tests = [np.array(t['input']) for t in task['test']]
    bg = background([a for a, _ in pairs])
    proposed = 0
    consistent = []

    def collect(patterns):
        nonlocal proposed
        for pat in patterns:
            proposed += 1
            if not all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(pat.predict, a))
                       for a, b in pairs):
                continue
            preds = [_safe(pat.predict, t) for t in tests]
            if any(p is None for p in preds):
                continue
            consistent.append({'family': pat.family, 'name': pat.name, 'complexity': pat.complexity,
                               'preds': preds})

    for fam in families:
        collect(FAMILIES[fam](pairs, bg))
    if combinations and not consistent:
        collect(combination_patterns(pairs, bg, time.time() + COMBINATION_SECONDS))
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
