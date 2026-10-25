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
                     thirds) taken, combined by AND / OR / XOR / NOR /
                     difference, or overlaid; or every output cell computed from its input
                     neighbourhood by a rule learned from the examples
  J roles            elements that control others: panels of a grid cut by
                     separator lines (or uniform blocks) - one panel chosen,
                     a summary with one cell per panel, or each panel kept /
                     filled by its role (first, centre, last) or contents;
                     objects of one colour taking the colour of a marker
                     (same shape or size, in line, inside, nearest); a colour
                     map read from a key of two-cell pieces; the region
                     bounded by marker cells
  K rays             every cell of a seed colour casts rays (one direction, a
                     line both ways, the 4 orthogonal / diagonal or all 8
                     directions) in its own or a fixed colour; at an obstacle
                     a ray stops, passes over it, turns left / right, steps
                     around it or reverses; diagonal rays may bounce off the
                     top / bottom, side or all edges; only rays that hit (or miss) an obstacle may be
                     drawn; crossings of different rays may take a colour
  M parameters       amounts taken from the elements: objects move a distance
                     equal to one of their properties (height, width, size,
                     ...; +-1 or doubled) or to a count in the grid; the grid
                     is scaled or tiled by a count
  N symmetry         a symmetry (mirror, rotation, diagonal or all eight)
                     completed about a centre found in the grid: the grid's
                     or foreground's centre, the cells of one colour, a chosen
                     object, or the best-fitting centre; cells under a mask
                     colour repaired from their images (whole grid or patch)
  O relative         the same pattern of cells placed around every object (in
                     cells, or in units of its size; shown in full by at least
                     two objects in the examples); objects lining up with
                     the nearest other object on its row, column or centre
  L multi-step       chains of up to three patterns: each step reaches part
                     of the change (the changes to one colour, the additions,
                     deletions or recolourings; or whatever a partial pattern
                     reaches without a wrong cell) or the output's size, and
                     the next pattern is searched from what is left; searched
                     only when no single pattern fits
  X exceptions       a pattern that is right except near some objects, made
                     exact by a condition read from those exceptions: objects
                     with one property value take no part, or only they do;
                     searched only when no single pattern fits

Conditions: family A maps one property, or two at once, to an action (keep,
delete, recolour, or adopt the colour of the one element an object
touches); family B selects by one property, or filters by one property's
value and then selects by another.
"""

import functools
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

        for k in range(len(parts0)):
            take = lambda a, k=k, parts_of=parts_of: (lambda ps: None if ps is None or k >= len(ps) else ps[k])(parts_of(a))
            if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(take, a))
                   for a, b in pairs):
                found.append(Pattern('I', f'{sname}: part {k}', 2, take))
        if len(parts0) == 2:
            # the colour where the operation holds, and the colour elsewhere (any two output colours)
            colors = sorted({int(v) for b in outs for v in np.unique(b)})
            for opname, op in _BOOL_OPS.items():
                for color, rest in itertools.permutations(colors[:4], 2):
                    def fn(a, op=op, color=color, rest=rest, parts_of=parts_of):
                        ps = parts_of(a)
                        if ps is None or len(ps) != 2:
                            return None
                        m = op(ps[0] != bg, ps[1] != bg)
                        return np.where(m, color, rest).astype(int)
                    if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(fn, a))
                           for a, b in pairs):
                        found.append(Pattern('I', f'{sname}: {opname} -> {color} else {rest}', 3, fn))
        for order in (itertools.permutations(range(len(parts0))) if len(parts0) <= 4 else ()):
            def overlay(a, order=order, parts_of=parts_of):
                ps = parts_of(a)
                if ps is None or len(ps) != len(order):
                    return None
                out = np.full(ps[0].shape, bg, dtype=int)
                for i in reversed(order):
                    out = np.where(ps[i] != bg, ps[i], out)
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


# ------------------------------------------------------------- family J

def _bands(n: int, cuts: List[int]) -> List[Tuple[int, int]]:
    """Index ranges [start, end) between separator lines"""
    out, start = [], 0
    for cut in sorted(cuts) + [n]:
        if cut > start:
            out.append((start, cut))
        start = cut + 1
    return out


def panel_grid(a: Array, bg: int) -> Optional[Dict]:
    """
    The grid as a grid of panels: the regions between full, single lines of
    one colour (the colour giving the most panels), or else the blocks of
    one colour formed by runs of identical rows and columns
    """
    h, w = a.shape
    best = None
    for color in np.unique(a).tolist():
        if color == bg:
            continue
        rows = [i for i in range(h) if np.all(a[i, :] == color)]
        cols = [j for j in range(w) if np.all(a[:, j] == color)]
        if any(y - x == 1 for x, y in zip(rows, rows[1:])) or any(y - x == 1 for x, y in zip(cols, cols[1:])):
            continue        # adjacent lines form a band of colour, not a separator
        rb, cb = _bands(h, rows), _bands(w, cols)
        if (rows or cols) and len(rb) * len(cb) >= 2 and \
                (best is None or len(rb) * len(cb) > len(best['rows']) * len(best['cols'])):
            best = {'kind': 'separators', 'sep': color, 'rows': rb, 'cols': cb}
    if best is None:
        rcuts = [i for i in range(1, h) if not np.array_equal(a[i], a[i - 1])]
        ccuts = [j for j in range(1, w) if not np.array_equal(a[:, j], a[:, j - 1])]
        rb = list(zip([0] + rcuts, rcuts + [h]))
        cb = list(zip([0] + ccuts, ccuts + [w]))
        if 2 <= len(rb) * len(cb) and len(rb) < h and len(cb) < w:
            best = {'kind': 'blocks', 'sep': None, 'rows': rb, 'cols': cb}
    return best


def panels(a: Array, bg: int = None) -> Tuple[Optional[Dict], List[Obj]]:
    """
    Panels as elements, each with its own properties and its role in the
    panel grid. Panels are a structure of one grid, so the grid's own
    background is used.
    """
    bg = background([a])
    g = panel_grid(a, bg)
    if g is None:
        return None, []
    R, C = len(g['rows']), len(g['cols'])
    out = []
    for i, (r0, r1) in enumerate(g['rows']):
        for j, (c0, c1) in enumerate(g['cols']):
            p = a[r0:r1, c0:c1]
            fg = p[p != bg]
            counts = Counter(fg.tolist())
            role = 'first' if (i, j) == (0, 0) else 'last' if (i, j) == (R - 1, C - 1) else \
                'center' if R % 2 and C % 2 and (i, j) == (R // 2, C // 2) else 'other'
            props = {
                'index': (i, j), 'row': i, 'col': j, 'role': role, 'bg': bg,
                'count': int(fg.size), 'empty': fg.size == 0,
                'color': counts.most_common(1)[0][0] if counts else bg,
                'colors': frozenset(counts), 'n_colors': len(counts),
                'pattern': bytes(str(p.shape), 'ascii') + p.astype(np.int16).tobytes(),
                'mask': bytes(str(p.shape), 'ascii') + np.packbits(p != bg).tobytes(),
            }
            out.append(Obj(frozenset(), (r0, c0, r1 - 1, c1 - 1), props))
    patterns = Counter(o.props['pattern'] for o in out)
    masks = Counter(o.props['mask'] for o in out)
    colorsets = Counter(o.props['colors'] for o in out)
    per_color = Counter(c for o in out for c in o.props['colors'])
    counts = [o.props['count'] for o in out]
    for o in out:
        o.props.update({
            'unique_pattern': patterns[o.props['pattern']] == 1,
            'unique_mask': masks[o.props['mask']] == 1,
            'unique_colors': colorsets[o.props['colors']] == 1,
            'has_unique_color': any(per_color[c] == 1 for c in o.props['colors']),
            'is_max_count': o.props['count'] == max(counts),
            'is_min_count': o.props['count'] == min(counts),
        })
    return g, out


def _region(a: Array, o: Obj) -> Array:
    r0, c0, r1, c1 = o.bbox
    return a[r0:r1 + 1, c0:c1 + 1]


PANEL_PROPERTIES = ('role', 'index', 'row', 'col', 'count', 'empty', 'color', 'colors', 'n_colors',
                    'unique_pattern', 'unique_mask', 'unique_colors', 'has_unique_color',
                    'is_max_count', 'is_min_count', 'mask', 'pattern')
PANEL_COST = {'pattern': 3, 'mask': 2, 'index': 1}


def _panel_actions(p: Array, q: Array, props: Dict) -> Optional[set]:
    """Every action that turns panel p into q"""
    bg = props['bg']
    acts = set()
    if np.array_equal(p, q):
        acts.add('keep')
    if len(np.unique(q)) == 1:
        acts.add(('fill', int(q.flat[0])))
    fills = {int(v) for v in np.unique(q[p == bg])} if np.any(p == bg) else set()
    if len(fills) == 1 and np.array_equal(q, np.where(p == bg, next(iter(fills)), p)):
        c = next(iter(fills))
        acts.add(('fill_bg', c))
        if c == props['color'] and c != bg:
            acts.add('fill_bg_own')
    return acts or None


def _apply_panel(p: Array, act, props: Dict) -> Array:
    bg = props['bg']
    if act == 'keep':
        return p
    if act == 'fill_bg_own':
        return np.where(p == bg, props['color'], p)
    kind, c = act
    return np.full(p.shape, c, dtype=int) if kind == 'fill' else np.where(p == bg, c, p)


def panel_patterns(pairs, bg) -> List[Pattern]:
    found = []
    grids = [panels(a, bg) for a, _ in pairs]
    if any(g is None for g, _ in grids):
        return found
    # 1. The output is one panel, chosen by a property
    for tname, tf in TRANSFORMS.items():
        match = [{i for i, o in enumerate(ps) if (lambda c: c.shape == b.shape and np.array_equal(c, b))(
                     tf(_region(a, o)))} for (a, b), (_, ps) in zip(pairs, grids)]
        if any(not m for m in match):
            continue
        for prop in PANEL_PROPERTIES:
            first = grids[0][1][next(iter(match[0]))].props[prop]
            for how in ('unique', 'max', 'min', ('eq', first)):
                if all((lambda sel: sel is not None and ps.index(sel) in m)(_select(ps, prop, how))
                       for (_, ps), m in zip(grids, match)):
                    def predict(t, prop=prop, how=how, tf=tf):
                        sel = _select(panels(t, bg)[1], prop, how)
                        return None if sel is None else tf(_region(t, sel))
                    found.append(Pattern('J', f'panel {prop} {_how(how)} +{tname}',
                                         2 + (tname != 'identity') + isinstance(how, tuple)
                                         + PANEL_COST.get(prop, 0), predict))
    # 2. The output summarises the panels, one cell (or block) per panel
    for transpose in (False, True):
        targets, scale = [], None
        for (a, b), (g, ps) in zip(pairs, grids):
            R, C = len(g['rows']), len(g['cols'])
            if transpose:
                R, C = C, R
            if b.shape[0] % R or b.shape[1] % C:
                break
            k = (b.shape[0] // R, b.shape[1] // C)
            if scale not in (None, k):
                break
            scale = k
            t = b[::k[0], ::k[1]]
            if not np.array_equal(np.kron(t, np.ones(k, dtype=int)), b):
                break
            targets.append(t.T if transpose else t)
        else:
            rows = [(o.props, {int(t[o.props['index']])}) for (_, ps), t in zip(grids, targets) for o in ps]
            for prop in PANEL_PROPERTIES:
                if prop == 'color' and all(o == next(iter(acts)) for o, acts in
                                           ((pr['color'], acts) for pr, acts in rows)):
                    mapping, cost = None, 3
                else:
                    mapping = _learn_mapping(rows, lambda pr, prop=prop: pr[prop])
                    if mapping is None or len(mapping) == len(rows):
                        continue
                    cost = 3 + len(mapping) + PANEL_COST.get(prop, 0)

                def predict(t, prop=prop, mapping=mapping, transpose=transpose, scale=scale):
                    g, ps = panels(t, bg)
                    if g is None:
                        return None
                    out = np.zeros((len(g['rows']), len(g['cols'])), dtype=int)
                    for o in ps:
                        v = o.props['color'] if mapping is None else mapping.get(o.props[prop])
                        if v is None:
                            return None
                        out[o.props['index']] = v
                    out = out.T if transpose else out
                    return np.kron(out, np.ones(scale, dtype=int))
                found.append(Pattern('J', f'panel summary: {"own colour" if mapping is None else prop}'
                                     f'{" transposed" if transpose else ""} x{scale}', cost, predict))
    # 3. Each panel is kept, filled, or has its background filled, by a property
    if all(a.shape == b.shape for a, b in pairs):
        rows = []
        for (a, b), (g, ps) in zip(pairs, grids):
            inside = np.zeros(a.shape, dtype=bool)
            for o in ps:
                r0, c0, r1, c1 = o.bbox
                inside[r0:r1 + 1, c0:c1 + 1] = True
            if not np.array_equal(a[~inside], b[~inside]):
                return found
            for o in ps:
                acts = _panel_actions(_region(a, o), _region(b, o), o.props)
                if acts is None:
                    return found
                rows.append((o.props, acts))
        if all(acts == {'keep'} or 'keep' in acts for _, acts in rows):
            return found
        for prop in PANEL_PROPERTIES:
            mapping = _learn_mapping(rows, lambda pr, prop=prop: pr[prop])
            if mapping is None:
                continue

            def predict(t, prop=prop, mapping=mapping):
                g, ps = panels(t, bg)
                if g is None:
                    return None
                out = t.copy()
                for o in ps:
                    act = mapping.get(o.props[prop])
                    if act is None:
                        return None
                    r0, c0, r1, c1 = o.bbox
                    out[r0:r1 + 1, c0:c1 + 1] = _apply_panel(_region(t, o), act, o.props)
                return out
            found.append(Pattern('J', f'panels by {prop}: {len(mapping)} rules',
                                 2 + len(mapping) + PANEL_COST.get(prop, 0), predict))
    return found


def _gap(o1: Obj, o2: Obj) -> int:
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in o1.cells for r2, c2 in o2.cells)


def _overlap(lo1, hi1, lo2, hi2) -> bool:
    return lo1 <= hi2 and lo2 <= hi1


def _nearest_color(o: Obj, others: List[Obj]) -> Optional[int]:
    if not others:
        return None
    d = [_gap(o, x) for x in others]
    colors = {x.props['color'] for x, v in zip(others, d) if v == min(d)}
    return colors.pop() if len(colors) == 1 else None


def _reference_rules():
    """Ways a target object finds the element whose colour it takes"""
    def same(prop):
        def rule(o, refs):
            colors = {x.props['color'] for x in refs if x.props[prop] == o.props[prop]}
            return colors.pop() if len(colors) == 1 else None
        return rule

    def column(o, refs):
        return _nearest_color(o, [x for x in refs if _overlap(o.bbox[1], o.bbox[3], x.bbox[1], x.bbox[3])])

    def row(o, refs):
        return _nearest_color(o, [x for x in refs if _overlap(o.bbox[0], o.bbox[2], x.bbox[0], x.bbox[2])])

    def aligned(o, refs):
        return _nearest_color(o, [x for x in refs if _overlap(o.bbox[1], o.bbox[3], x.bbox[1], x.bbox[3])
                                  or _overlap(o.bbox[0], o.bbox[2], x.bbox[0], x.bbox[2])])

    def inside(o, refs):
        r0, c0, r1, c1 = o.bbox
        colors = {x.props['color'] for x in refs
                  if r0 <= x.bbox[0] and x.bbox[2] <= r1 and c0 <= x.bbox[1] and x.bbox[3] <= c1}
        return colors.pop() if len(colors) == 1 else None

    return {'same shape': same('shape'), 'same shape up to rotation': same('shape_canon'),
            'same size': same('size'), 'marker in its column': column, 'marker in its row': row,
            'marker in its row or column': aligned, 'marker inside it': inside,
            'nearest element': lambda o, refs: _nearest_color(o, refs)}


def marker_patterns(pairs, bg) -> List[Pattern]:
    """Objects of one colour take the colour of a reference element (a key, a marker)"""
    found = []
    if any(a.shape != b.shape for a, b in pairs):
        return found
    for mode in ('color4', 'color8'):
        seg = [segment(a, bg, mode) for a, _ in pairs]
        targets = set()
        ok = True
        for (a, b), objs in zip(pairs, seg):
            covered = np.zeros(a.shape, dtype=bool)
            for o in objs:
                cells = tuple(zip(*o.cells))
                covered[cells] = True
                after = set(b[cells].tolist())
                if after != {o.props['color']}:
                    if len(after) != 1:
                        ok = False
                    targets.add(o.props['color'])
            if not np.array_equal(a[~covered], b[~covered]):
                ok = False
        if not ok or len(targets) != 1:
            continue
        target = targets.pop()
        for rname, rule in _reference_rules().items():
            def predict(t, rule=rule, mode=mode, target=target):
                objs = segment(t, bg, mode)
                refs = [x for x in objs if x.props['color'] != target]
                out = t.copy()
                for o in objs:
                    if o.props['color'] == target:
                        c = rule(o, refs)
                        if c is None:
                            return None
                        out[tuple(zip(*o.cells))] = c
                return out
            if all((lambda o: o is not None and np.array_equal(o, b))(_safe(predict, a)) for a, b in pairs):
                found.append(Pattern('J', f'objects[{mode}] of colour {target} take the colour of the {rname}',
                                     3, predict))
    return found


def _legend_map(a: Array, bg: int, reverse: bool) -> Tuple[Optional[Dict], set]:
    """Colour map read from the grid's key: two-cell pieces of two colours (first -> second)"""
    mapping, cells = {}, set()
    for o in segment(a, bg, 'multi4'):
        if o.props['size'] == 2 and o.props['n_colors'] == 2:
            (p1, p2) = sorted(o.cells)
            x, y = int(a[p1]), int(a[p2])
            if reverse:
                x, y = y, x
            if mapping.get(x, y) != y:
                return None, set()
            mapping[x] = y
            cells |= set(o.cells)
    return (mapping or None), cells


def legend_patterns(pairs, bg) -> List[Pattern]:
    found = []
    if any(a.shape != b.shape for a, b in pairs):
        return found
    for reverse in (False, True):
        def predict(t, reverse=reverse):
            mapping, cells = _legend_map(t, bg, reverse)
            if mapping is None:
                return None
            out = t.copy()
            for r in range(t.shape[0]):
                for c in range(t.shape[1]):
                    if (r, c) not in cells:
                        out[r, c] = mapping.get(int(t[r, c]), int(t[r, c]))
            return out
        if all((lambda o: o is not None and np.array_equal(o, b) and not np.array_equal(a, b))(_safe(predict, a))
               for a, b in pairs):
            found.append(Pattern('J', f'colour map read from the key{" (reversed)" if reverse else ""}', 3, predict))
    # The region bounded by the cells of one marker colour
    colors = [int(c) for c in np.unique(pairs[0][0]) if c != bg]
    for color in colors:
        for inner in (False, True):
            def crop_to(t, color=color, inner=inner):
                rr, cc = np.nonzero(t == color)
                if rr.size < 2:
                    return None
                r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
                if inner:
                    r0, r1, c0, c1 = r0 + 1, r1 - 1, c0 + 1, c1 - 1
                return t[r0:r1 + 1, c0:c1 + 1] if r0 <= r1 and c0 <= c1 else None
            if all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(crop_to, a))
                   for a, b in pairs):
                found.append(Pattern('J', f'region bounded by markers of {color}{" (inside)" if inner else ""}',
                                     2, crop_to))
    return found


def role_patterns(pairs, bg) -> List[Pattern]:
    return panel_patterns(pairs, bg) + marker_patterns(pairs, bg) + legend_patterns(pairs, bg)


# ------------------------------------------------------------- family K

DIRS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1),
        'NE': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (1, -1)}
DIR_SETS = dict({d: (d,) for d in DIRS}, **{
    'N+S': ('N', 'S'), 'E+W': ('E', 'W'), 'orthogonal': ('N', 'S', 'E', 'W'),
    'diagonal': ('NE', 'NW', 'SE', 'SW'), 'all 8': tuple(DIRS)})
ON_OBSTACLE = ('stop', 'pass', 'turn left', 'turn right', 'step around left', 'step around right', 'reverse')
RAY_SELECT = ('any', 'hit', 'free')


def _turn(d: Tuple[int, int], left: bool) -> Tuple[int, int]:
    dr, dc = d
    return (-dc, dr) if left else (dc, -dr)


def _first_obstacle(a: Array, bg: int, r: int, c: int, d: Tuple[int, int]) -> bool:
    h, w = a.shape
    r, c = r + d[0], c + d[1]
    while 0 <= r < h and 0 <= c < w:
        if a[r, c] != bg:
            return True
        r, c = r + d[0], c + d[1]
    return False


def cast_rays(a: Array, bg: int, seed: Optional[int], dirs: Sequence[str], color: Optional[int], on_obstacle: str,
              bounce: Optional[str], select: str, cross: Optional[int]) -> Array:
    """
    Every cell of the seed colour (every non-background cell if seed is
    None) casts a ray in each direction. Rays paint background cells;
    obstacles are the input's non-background cells. Diagonal rays may
    bounce off the top / bottom edges ('rows'), the side edges ('cols') or
    all edges ('all').
    """
    h, w = a.shape
    out = a.copy()
    owner = np.full(a.shape, -1, dtype=int)        # which ray painted a cell
    limit = 2 * (h + w) + 4
    ray_id = 0
    for r, c in zip(*np.nonzero(a != bg if seed is None else a == seed)):
        col = int(a[r, c]) if color is None else color
        for name in dirs:
            d = DIRS[name]
            ray_id += 1
            if select != 'any' and _first_obstacle(a, bg, r, c, d) != (select == 'hit'):
                continue
            y, x, steps, turns = r, c, 0, 0
            while steps < limit and turns < 4:
                ny, nx = y + d[0], x + d[1]
                if not (0 <= ny < h and 0 <= nx < w):
                    out_r, out_c = not 0 <= ny < h, not 0 <= nx < w
                    if bounce and d[0] and d[1] and (bounce == 'all' or (out_r, out_c) == (bounce == 'rows', bounce == 'cols')):
                        d = (-d[0] if out_r else d[0], -d[1] if out_c else d[1])
                        steps += 1
                        continue
                    break
                if a[ny, nx] != bg:
                    if on_obstacle == 'stop':
                        break
                    if on_obstacle == 'pass':
                        y, x = ny, nx
                        steps += 1
                        continue
                    turns += 1
                    if on_obstacle in ('turn left', 'turn right'):
                        d = _turn(d, on_obstacle == 'turn left')
                        continue
                    if on_obstacle == 'reverse':
                        d = (-d[0], -d[1])
                        continue
                    side = _turn(d, on_obstacle == 'step around left')
                    sy, sx = y + side[0], x + side[1]
                    if not (0 <= sy < h and 0 <= sx < w) or a[sy, sx] != bg:
                        break
                    ny, nx = sy, sx
                if out[ny, nx] == bg:
                    out[ny, nx] = col
                elif cross is not None and owner[ny, nx] not in (-1, ray_id) and out[ny, nx] != col:
                    out[ny, nx] = cross
                owner[ny, nx] = ray_id
                y, x = ny, nx
                steps += 1
    return out


def ray_patterns(pairs, bg, accept: Optional[Callable] = None) -> List[Pattern]:
    """Ray patterns that fit exactly, or (with `accept`) every one that `accept` takes"""
    found = []
    if any(a.shape != b.shape for a, b in pairs):
        return found
    # rays only paint background cells (when only part of the change is wanted, some of it must be additions)
    if accept is None and not all(np.all(a[a != b] == bg) for a, b in pairs) or \
            not any(np.any((a != b) & (a == bg)) for a, b in pairs):
        return found
    seeds = [None] + sorted(set.intersection(*[set(np.unique(a).tolist()) - {bg} for a, _ in pairs]))
    added = sorted({int(v) for a, b in pairs for v in np.unique(b[a != b])})
    a0, b0 = pairs[0]
    for seed in seeds:
        for color in [None] + added:
            # crossings differ only between rays of different colours (rays in their seeds' colours)
            crosses = [None] + (added if color is None else [])
            for dname, dirs in DIR_SETS.items():
                for on_obstacle in ON_OBSTACLE:
                    for select in RAY_SELECT:
                        diagonal = any(DIRS[x][0] and DIRS[x][1] for x in dirs)
                        for bounce in ((None, 'rows', 'cols', 'all') if diagonal else (None,)):
                            for cross in crosses:
                                args = (seed, dirs, color, on_obstacle, bounce, select, cross)
                                if accept is not None:
                                    if not accept(lambda t, args=args: cast_rays(t, bg, *args)):
                                        continue
                                elif not np.array_equal(cast_rays(a0, bg, *args), b0) or \
                                        not all(np.array_equal(cast_rays(a, bg, *args), b) for a, b in pairs[1:]):
                                    continue
                                cost = 3 + (on_obstacle != 'stop') + (select != 'any') + (bounce is not None) \
                                    + (cross is not None)
                                found.append(Pattern(
                                    'K', f'rays from {"every element" if seed is None else seed} {dname} in {"own colour" if color is None else color},'
                                         f' at obstacle {on_obstacle}, {select} rays'
                                         f'{f", bounce off {bounce} edges" if bounce else ""}'
                                         f'{f", crossings {cross}" if cross is not None else ""}',
                                    cost, lambda t, args=args: cast_rays(t, bg, *args)))
    return found


# ------------------------------------------------------------- family N

# Symmetry maps about a centre given in doubled coordinates (so it may fall between cells):
# each takes doubled positions (R, C) and the centre (cy, cx) and gives the image's doubled position
_SYM_MAPS = {
    'lr': lambda R, C, cy, cx: (R, 2 * cx - C),
    'ud': lambda R, C, cy, cx: (2 * cy - R, C),
    'rot180': lambda R, C, cy, cx: (2 * cy - R, 2 * cx - C),
    'rot90': lambda R, C, cy, cx: (cy + (C - cx), cx - (R - cy)),
    'rot270': lambda R, C, cy, cx: (cy - (C - cx), cx + (R - cy)),
    'transpose': lambda R, C, cy, cx: (cy + (C - cx), cx + (R - cy)),
    'anti_transpose': lambda R, C, cy, cx: (cy - (C - cx), cx - (R - cy)),
}
SYMMETRY_GROUPS = {
    'lr': ('lr',), 'ud': ('ud',), 'rot180': ('rot180',), 'both mirrors': ('lr', 'ud', 'rot180'),
    'diagonal': ('transpose',), 'rot90': ('rot90', 'rot180', 'rot270'), 'all 8': tuple(_SYM_MAPS),
}


@functools.lru_cache(maxsize=8192)
def _image_index(shape, op: str, centre: Tuple[int, int]):
    """For every cell, the cell its image falls on (and whether it lies in the grid)"""
    h, w = shape
    R, C = np.mgrid[0:h, 0:w] * 2
    R2, C2 = _SYM_MAPS[op](R, C, *centre)
    valid = (R2 % 2 == 0) & (C2 % 2 == 0) & (R2 >= 0) & (C2 >= 0) & (R2 < 2 * h) & (C2 < 2 * w)
    return valid, np.where(valid, R2 // 2, 0), np.where(valid, C2 // 2, 0)


def symmetric_fill(a: Array, known: Array, ops: Sequence[str], centre) -> Tuple[Array, Array]:
    """Unknown cells take the value of a known image cell; returns the grid and the cells still unknown"""
    out, filled = a.copy(), known.copy()
    for op in ops:
        valid, rr, cc = _image_index(a.shape, op, tuple(centre))
        take = ~filled & valid & known[rr, cc]
        out[take] = a[rr[take], cc[take]]
        filled |= take
    return out, ~filled


def _agreement(a: Array, known: Array, ops: Sequence[str], centre) -> Tuple[int, int]:
    """(matching, conflicting) known cell pairs under the symmetry"""
    match = conflict = 0
    for op in ops:
        valid, rr, cc = _image_index(a.shape, op, tuple(centre))
        both = valid & known & known[rr, cc]
        same = a[both] == a[rr[both], cc[both]]
        match += int(same.sum())
        conflict += int((~same).sum())
    return match, conflict


def find_centre(a: Array, known: Array, ops: Sequence[str], near, radius: int = 6):
    """The centre near `near` where most known cells agree with their images and none conflict"""
    best, best_match = None, 0
    # a left-right mirror does not depend on the centre's row, a top-bottom mirror not on its column
    rows = [near[0]] if set(ops) == {'lr'} else range(near[0] - radius, near[0] + radius + 1)
    cols = [near[1]] if set(ops) == {'ud'} else range(near[1] - radius, near[1] + radius + 1)
    for cy in rows:
        for cx in cols:
            match, conflict = _agreement(a, known, ops, (cy, cx))
            if conflict == 0 and match > best_match:
                best, best_match = (cy, cx), match
    return best


def _bbox_centre(mask: Array):
    rr, cc = np.nonzero(mask)
    return None if rr.size == 0 else (int(rr.min() + rr.max()), int(cc.min() + cc.max()))


def _centre_finders(pairs, bg):
    """Ways to find a symmetry centre in a grid (doubled coordinates)"""
    finders = [('grid centre', lambda a, ops: (a.shape[0] - 1, a.shape[1] - 1)),
               ('foreground centre', lambda a, ops: _bbox_centre(a != bg))]
    for c in sorted(set.intersection(*[set(np.unique(a).tolist()) for a, _ in pairs]) - {bg}):
        finders.append((f'centre of colour {c}', lambda a, ops, c=c: _bbox_centre(a == c)))
    for prop, how in (('size', 'max'), ('size', 'min'), ('unique_color', ('eq', True))):
        def obj_centre(a, ops, prop=prop, how=how):
            sel = _select(segment(a, bg, 'multi8'), prop, how)
            return None if sel is None else (sel.bbox[0] + sel.bbox[2], sel.bbox[1] + sel.bbox[3])
        finders.append((f'centre of the {prop} {_how(how)} object', obj_centre))
    finders.append(('best-fitting centre',
                    lambda a, ops: (lambda near: None if near is None else find_centre(a, a != bg, ops, near))(
                        _bbox_centre(a != bg))))
    return finders


def symmetry_patterns(pairs, bg, accept: Optional[Callable] = None) -> List[Pattern]:
    """Symmetry patterns that fit exactly, or (with `accept`) the completions that `accept` takes"""
    found = []
    same = all(a.shape == b.shape for a, b in pairs)
    additive = same and (all(np.all(a[a != b] == bg) for a, b in pairs) if accept is None else
                         any(np.any((a != b) & (a == bg)) for a, b in pairs))
    # 1. Complete a symmetry about a centre found in the grid
    if same and additive and not all(np.array_equal(a, b) for a, b in pairs):
        for fname, finder in _centre_finders(pairs, bg):
            for gname, ops in SYMMETRY_GROUPS.items():
                def predict(t, finder=finder, ops=ops):
                    centre = finder(t, ops)
                    return None if centre is None else symmetric_fill(t, t != bg, ops, centre)[0]
                if (_fits(predict, pairs) if accept is None else accept(predict)):
                    found.append(Pattern('N', f'complete {gname} symmetry about the {fname}',
                                         3 + (fname != 'grid centre'), predict))
    if accept is not None:
        return found
    # 2. Repair the cells under a mask colour from their mirror images
    masks = sorted(set.intersection(*[set(np.unique(a).tolist()) for a, _ in pairs]))
    for m in masks:
        if same and not all(np.all(b[a == m] != m) and np.array_equal(a[a != m], b[a != m]) for a, b in pairs):
            continue
        if not same and not all(np.any(a == m) and (lambda rr, cc: (rr.max() - rr.min() + 1, cc.max() - cc.min() + 1))(
                *np.nonzero(a == m)) == b.shape for a, b in pairs):
            continue
        for gname, ops in SYMMETRY_GROUPS.items():
            def repaired(t, m=m, ops=ops):
                known = t != m
                if known.all():
                    return None
                centre = find_centre(t, known, ops, (t.shape[0] - 1, t.shape[1] - 1))
                if centre is None:
                    return None
                out, unknown = symmetric_fill(t, known, ops, centre)
                return None if unknown.any() else out

            def patch(t, m=m, repaired=repaired):
                out = repaired(t)
                if out is None:
                    return None
                rr, cc = np.nonzero(t == m)
                return out[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            fn = repaired if same else patch
            if _fits(fn, pairs):
                found.append(Pattern('N', f'repair cells of {m} by {gname} symmetry'
                                     f'{"" if same else ", output the repaired patch"}', 3, fn))
    return found


# ------------------------------------------------------------- family O

def _assign_to_anchors(a: Array, b: Array, anchors: List[Obj]) -> Optional[List[List[Tuple[int, int, int]]]]:
    """Every added cell, given to its nearest anchor"""
    out = [[] for _ in anchors]
    if not anchors:
        return None
    for r, c in zip(*np.nonzero(a != b)):
        d = [min(max(abs(r - y), abs(c - x)) for y, x in o.cells) for o in anchors]
        out[int(np.argmin(d))].append((int(r), int(c), int(b[r, c])))
    return out


def _template_paint(t: Array, objs: List[Obj], key, templates, scaled: bool, bg: int) -> Array:
    out = t.copy()
    h, w = t.shape
    for o in objs:
        r0, c0, r1, c1 = o.bbox
        uh, uw = (r1 - r0 + 1, c1 - c0 + 1) if scaled else (1, 1)
        for dr, dc, v in templates.get(key(o.props), ()):
            for y in range(r0 + dr * uh, r0 + dr * uh + uh):
                for x in range(c0 + dc * uw, c0 + dc * uw + uw):
                    if 0 <= y < h and 0 <= x < w and out[y, x] == bg:
                        out[y, x] = v
    return out


def relative_patterns(pairs, bg) -> List[Pattern]:
    found = []
    if any(a.shape != b.shape for a, b in pairs) or all(np.array_equal(a, b) for a, b in pairs):
        return found
    additive = all(np.all(a[a != b] == bg) for a, b in pairs)
    for mode in ('color4', 'color8', 'multi8'):
        seg = [segment(a, bg, mode) for a, _ in pairs]
        if any(not objs for objs in seg):
            continue
        # 1. The same pattern of added cells around every anchor (in cells, or in units of the anchor's size)
        if additive:
            for kname, key in (('every object', lambda pr: 0), ('colour', lambda pr: pr['color']),
                               ('shape', lambda pr: pr['shape_canon'])):
                for scaled in (False, True):
                    templates: Dict[object, set] = {}
                    seen: Dict[object, List[frozenset]] = {}
                    for (a, b), objs in zip(pairs, seg):
                        given = _assign_to_anchors(a, b, objs)
                        for o, cells in zip(objs, given):
                            r0, c0, r1, c1 = o.bbox
                            uh, uw = (r1 - r0 + 1, c1 - c0 + 1) if scaled else (1, 1)
                            rel = frozenset(((r - r0) // uh, (c - c0) // uw, v) for r, c, v in cells)
                            templates.setdefault(key(o.props), set()).update(rel)
                            seen.setdefault(key(o.props), []).append(rel)
                    # A template generalises only if at least two anchors show all of it
                    # (one anchor per key would just record that anchor's answer)
                    if not any(templates.values()) or any(
                            tpl and sum(rel == tpl for rel in seen[k]) < 2 for k, tpl in templates.items()):
                        continue
                    size = sum(len(v) for v in templates.values())

                    def predict(t, key=key, templates=templates, scaled=scaled, mode=mode):
                        return _template_paint(t, segment(t, bg, mode), key, templates, scaled, bg)
                    if _fits(predict, pairs):
                        found.append(Pattern('O', f'cells placed around each object[{mode}] by {kname}'
                                             f'{" in units of its size" if scaled else ""}: {size} offsets',
                                             3 + min(size, 20) / 4 + (kname != 'every object'), predict))
        # 2. Objects of the moving colours line up with the nearest other object
        moved, still = set(), set()
        for (a, b), objs in zip(pairs, seg):
            for o in objs:
                (still if all(b[p] == a[p] for p in o.cells) else moved).add(o.props['color'])
        if not moved or not still or moved & still:
            continue
        for axis in ('row', 'column', 'both'):
            def predict(t, mode=mode, moved=frozenset(moved), axis=axis):
                objs = segment(t, bg, mode)
                refs = [o for o in objs if o.props['color'] not in moved]
                moving = [o.props['color'] in moved for o in objs]
                vectors = []
                for o, m in zip(objs, moving):
                    if not m:
                        vectors.append((0, 0))
                        continue
                    if not refs:
                        return None
                    ref = min(refs, key=lambda x: _gap(o, x))
                    dy = (ref.bbox[0] + ref.bbox[2]) - (o.bbox[0] + o.bbox[2])
                    dx = (ref.bbox[1] + ref.bbox[3]) - (o.bbox[1] + o.bbox[3])
                    dy, dx = (dy if axis != 'column' else 0), (dx if axis != 'row' else 0)
                    if dy % 2 or dx % 2:
                        return None
                    vectors.append((dy // 2, dx // 2))
                return _shift_objects(t, objs, moving, vectors, bg)
            if _fits(predict, pairs):
                found.append(Pattern('O', f'objects[{mode}] of {sorted(moved)} line up with the nearest '
                                     f'other object ({axis})', 4, predict))
    return found


# ------------------------------------------------------------- partial matches

def _progress(fn, pairs) -> Optional[Tuple[int, int, int]]:
    """(needed, right, wrong) cell counts of a pattern over the examples (None if it fails on one)"""
    needed = right = wrong = 0
    for a, b in pairs:
        out = _safe(fn, a)
        if out is None or out.shape != b.shape:
            return None
        change = a != b
        needed += int(change.sum())
        right += int((change & (out == b)).sum())
        # wrong: differs from the output, unless it is a needed change not yet made
        wrong += int(((out != b) & ~(change & (out == a))).sum())
    return needed, right, wrong


def _near_miss(stats) -> bool:
    """Reaches at least half of the change, with fewer wrong cells than cells to change"""
    needed, right, wrong = stats
    return needed > 0 and 0 < wrong < needed and 2 * right >= needed


def partial_pool(pairs, bg, deadline: float = float('inf')) -> List[Tuple[Pattern, Tuple[int, int, int]]]:
    """
    Patterns produced without an exact fit that are useful partial matches:
    sound steps (right where they change, wrong nowhere) and near misses
    """
    if any(a.shape != b.shape for a, b in pairs):
        return []
    a0, b0 = pairs[0]
    stats = {}

    import time

    def accept(fn):
        if time.time() > deadline:
            return False
        first = _progress(fn, pairs[:1])
        if first is None or first[1] == 0 or not (first[2] == 0 or _near_miss(first)):
            return False            # judged on the first example before the others
        st = _progress(fn, pairs)
        if st is None or st[1] == 0 or st[1] == st[0] and st[2] == 0:
            return False
        stats[id(fn)] = st
        return st[2] == 0 or _near_miss(st)
    pats = [p for p in transform_patterns(pairs, bg) if accept(p.predict)]
    pats += ray_patterns(pairs, bg, accept) + symmetry_patterns(pairs, bg, accept)
    return [(p, stats[id(p.predict)]) for p in pats if id(p.predict) in stats]


EXCEPTION_PROPERTIES = ('color', 'size', 'height', 'width', 'is_rect', 'holes', 'n_colors', 'shape_canon',
                        'sym_lr', 'sym_ud', 'touches_border', 'is_largest', 'is_smallest', 'unique_color',
                        'unique_shape')
MAX_NEAR_MISSES = 12
EXCEPTION_SECONDS = 7        # share of MULTI_STEP_SECONDS the exception search may use
POOL_SECONDS = 5             # share of MULTI_STEP_SECONDS for collecting partial matches


def _with_only(fn, t: Array, objs: List[Obj], acting: List[bool], bg: int) -> Optional[Array]:
    """Run the pattern with only the acting objects present, then put the others back"""
    removed = [o for o, act in zip(objs, acting) if not act]
    t2 = t.copy()
    for o in removed:
        t2[tuple(zip(*o.cells))] = bg
    out = _safe(fn, t2)
    if out is None or out.shape != t.shape:
        return None
    for o in removed:
        cells = tuple(zip(*o.cells))
        out[cells] = t[cells]
    return out


def exception_patterns(pairs, bg, deadline: float, pool=None) -> List[Pattern]:
    """
    A near-miss pattern made exact by a condition found from its exceptions:
    objects with one property value do not take part, or only they take part
    """
    import time
    found = []
    if any(a.shape != b.shape for a, b in pairs):
        return found
    pool = partial_pool(pairs, bg) if pool is None else pool
    near = sorted(((p, st) for p, st in pool if _near_miss(st)), key=lambda x: (x[1][2], x[0].complexity))
    for base, _ in near[:MAX_NEAR_MISSES]:
        for mode in ('color4', 'multi8'):
            if time.time() > deadline:
                return found
            seg = [segment(a, bg, mode) for a, _ in pairs]
            a0, b0 = pairs[0]
            out0 = _safe(base.predict, a0)
            wrong = np.pad(out0 != b0, 1)
            near_wrong = np.zeros_like(wrong)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near_wrong |= np.roll(np.roll(wrong, dr, 0), dc, 1)
            near_wrong = near_wrong[1:-1, 1:-1]
            exceptions = {i for i, o in enumerate(seg[0]) if any(near_wrong[p] for p in o.cells)}
            if not exceptions:
                continue
            for prop in EXCEPTION_PROPERTIES:
                for v in sorted({o.props[prop] for o in seg[0]}, key=str):
                    chosen = {i for i, o in enumerate(seg[0]) if o.props[prop] == v}
                    for variant in ('without', 'only'):
                        # the excluded objects must be among the exceptions; the acting ones must avoid them
                        if variant == 'without' and not (chosen and chosen <= exceptions):
                            continue
                        if variant == 'only' and not (chosen and not chosen & exceptions):
                            continue
                        acting = lambda o, prop=prop, v=v, variant=variant: (o.props[prop] == v) == (variant == 'only')
                        if not all((lambda o: o is not None and np.array_equal(o, b))(
                                _with_only(base.predict, a, objs, [acting(o) for o in objs], bg))
                                for (a, b), objs in zip(pairs, seg)):
                            continue

                        def predict(t, mode=mode, acting=acting, fn=base.predict):
                            objs = segment(t, bg, mode)
                            return _with_only(fn, t, objs, [acting(o) for o in objs], bg)
                        label = f'{prop}={str(v)[:20]}'
                        found.append(Pattern('X', f'{base.family}:{base.name} '
                                             f'{"except" if variant == "without" else "only for"} objects[{mode}] with {label}',
                                             base.complexity + 2 + PROPERTY_COST.get(prop, 0), predict))
    return found


# ------------------------------------------------------------- family M

NUMERIC_PROPERTIES = ('height', 'width', 'size', 'holes', 'n_colors', 'color_count', 'shape_count')


def _shift_objects(a: Array, objs: List[Obj], moving: List[bool], vectors, bg: int) -> Optional[Array]:
    """Move the chosen objects by their own vectors (moved objects are drawn over still ones)"""
    out = a.copy()
    h, w = a.shape
    for o, m in zip(objs, moving):
        if m:
            out[tuple(zip(*o.cells))] = bg
    for o, m, (dr, dc) in zip(objs, moving, vectors):
        if not m:
            continue
        for r, c in o.cells:
            if not (0 <= r + dr < h and 0 <= c + dc < w):
                return None
            out[r + dr, c + dc] = a[r, c]
    return out


def param_patterns(pairs, bg) -> List[Pattern]:
    """Patterns whose amount (distance, factor) is a property of the element or a count in the grid"""
    found = []
    quantities = _quantities(pairs, bg)
    qvals = [{n: q(a) for n, q in quantities} for a, _ in pairs]
    # 1. Objects move a distance taken from their own property, or from a count in the grid
    if all(a.shape == b.shape for a, b in pairs) and not all(np.array_equal(a, b) for a, b in pairs):
        distances = [(f'{prop}{off:+d}' if off else prop, 'prop', prop, 1, off)
                     for prop in NUMERIC_PROPERTIES for off in (0, 1, -1)]
        distances += [(f'2*{prop}', 'prop', prop, 2, 0) for prop in NUMERIC_PROPERTIES]
        distances += [(name, 'count', name, 1, 0) for name, _ in quantities]
        for mode in ('color4', 'color8', 'multi8'):
            seg = [segment(a, bg, mode) for a, _ in pairs]
            moved, still = set(), set()
            for (a, b), objs in zip(pairs, seg):
                for o in objs:
                    (still if all(b[p] == a[p] for p in o.cells) else moved).add(o.props['color'])
            movers = [('all objects', None)]
            if moved and still and not moved & still:
                movers.append((f'objects of {sorted(moved)}', frozenset(moved)))
            for mname, colors in movers:
                for dname, kind, key, mult, off in distances:
                    for direction, (dr, dc) in DIRS.items():
                        def predict(t, objs=None, qv=None, mode=mode, colors=colors, kind=kind, key=key,
                                    mult=mult, off=off, dr=dr, dc=dc):
                            objs = segment(t, bg, mode) if objs is None else objs
                            if kind == 'count':
                                qv = qv if qv is not None else {key: dict(quantities)[key](t)}
                            moving = [colors is None or o.props['color'] in colors for o in objs]
                            dist = [(o.props[key] if kind == 'prop' else qv[key]) * mult + off for o in objs]
                            return _shift_objects(t, objs, moving, [(dr * k, dc * k) for k in dist], bg)
                        if all((lambda o: o is not None and np.array_equal(o, b))(predict(a, objs, qv))
                               for (a, b), objs, qv in zip(pairs, seg, qvals)):
                            found.append(Pattern('M', f'{mname}[{mode}] move {direction} by {dname}', 4,
                                                 lambda t, f=predict: f(t)))
    # 2. Scale or tile by a count that differs between the examples
    factors = []
    for a, b in pairs:
        if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return found
        factors.append((b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]))
    if len(set(factors)) > 1 and all(k == m for k, m in factors):
        for name, q in quantities:
            if all(qv[name] == k for qv, (k, _) in zip(qvals, factors)):
                for oname, op in (('upscale', lambda a, k: np.kron(a, np.ones((k, k), dtype=a.dtype))),
                                  ('tile', lambda a, k: np.tile(a, (k, k)))):
                    fn = lambda t, q=q, op=op: (lambda k: op(t, k) if 0 < k <= 30 else None)(q(t))
                    if all((lambda o: o is not None and np.array_equal(o, b))(_safe(fn, a)) for a, b in pairs):
                        found.append(Pattern('M', f'{oname} by {name}', 3, fn))
    return found


# ------------------------------------------------------------- family L

MULTI_STEP_SECONDS = 20      # time budget for multi-step patterns, used only when no single pattern fits
MAX_STEPS = 3
# Limits on the chains formed (their number is the product of starts and finals at every step)
MAX_PARTIAL_STARTS = 30      # partial steps tried first, those reaching the most cells
MAX_STARTS_PER_RESULT = 3    # cheapest starts kept for each distinct intermediate result
MAX_FINALS = 5               # cheapest next patterns kept for each intermediate result
MAX_CHAINS = 300             # cheapest chains returned


def _fits(pred, pairs) -> bool:
    return all((lambda o: o is not None and o.shape == b.shape and np.array_equal(o, b))(_safe(pred, a))
               for a, b in pairs)


def _subgoals(pairs, bg) -> List[Tuple[str, List[Tuple[Array, Array]]]]:
    """Intermediate targets: each output with only part of its change applied"""
    if any(a.shape != b.shape for a, b in pairs):
        return []
    colors = sorted({int(v) for a, b in pairs for v in np.unique(b[a != b])})
    parts = [(f'changes to {c}', lambda a, b, c=c: (a != b) & (b == c)) for c in colors]
    parts += [('additions', lambda a, b: (a != b) & (a == bg)),
              ('deletions', lambda a, b: (a != b) & (b == bg)),
              ('recolourings', lambda a, b: (a != b) & (a != bg) & (b != bg))]
    out, seen = [], set()
    for name, part in parts:
        targets = [np.where(part(a, b), b, a) for a, b in pairs]
        if all(np.array_equal(t, a) for t, (a, _) in zip(targets, pairs)) or \
                all(np.array_equal(t, b) for t, (_, b) in zip(targets, pairs)):
            continue
        key = b''.join(t.tobytes() for t in targets)
        if key not in seen:
            seen.add(key)
            out.append((name, [(a, t) for (a, _), t in zip(pairs, targets)]))
    return out


def _shape_steps(pairs) -> List[Tuple[str, Callable, float]]:
    """First steps that give the input the output's size"""
    ratios = {(b.shape[0] / a.shape[0], b.shape[1] / a.shape[1]) for a, b in pairs}
    if len(ratios) != 1:
        return []
    kr, kc = ratios.pop()
    steps = []
    if kr >= 1 and kc >= 1 and kr == int(kr) and kc == int(kc) and (kr, kc) != (1, 1):
        k, m = int(kr), int(kc)
        steps.append((f'upscale {k}x{m}', lambda a, k=k, m=m: np.kron(a, np.ones((k, m), dtype=a.dtype)), 1))
        steps.append((f'tile {k}x{m}', lambda a, k=k, m=m: np.tile(a, (k, m)), 1))
    elif kr <= 1 and kc <= 1 and (kr, kc) != (1, 1) and (1 / kr) == int(1 / kr) and (1 / kc) == int(1 / kc):
        k, m = int(1 / kr), int(1 / kc)
        steps.append((f'downscale {k}x{m}',
                      lambda a, k=k, m=m: a[::k, ::m] if a.shape[0] % k == 0 and a.shape[1] % m == 0 else None, 1))
    return steps


def _step_families():
    """Patterns that can be one step of a chain (every single family)"""
    return list(FAMILIES.values())


def multi_step_patterns(pairs, bg, deadline: float, depth: int = MAX_STEPS, pool=None) -> List[Pattern]:
    """
    Chains of patterns: a first step that reaches part of the change (or
    the output's size), then a final pattern, or another chain, from there
    """
    import time
    found = []
    if depth < 2:
        return found
    starts = list(_shape_steps(pairs))
    # steps that reach part of the change without getting any cell wrong
    pool = partial_pool(pairs, bg, min(deadline, time.time() + POOL_SECONDS)) if pool is None else pool
    sound = sorted(((p, st) for p, st in pool if st[2] == 0), key=lambda x: (-x[1][1], x[0].complexity))
    starts += [(f'[partial] {p.family}:{p.name}', p.predict, p.complexity + 1)
               for p, _ in sound[:MAX_PARTIAL_STARTS]]
    for gname, sub in _subgoals(pairs, bg):
        for family in _step_families():
            if time.time() > deadline:
                break
            starts += [(f'[{gname}] {p.family}:{p.name}', p.predict, p.complexity)
                       for p in family(sub, bg) if _fits(p.predict, sub)]
    # The next step is searched once per distinct intermediate result; every start that
    # reaches it is kept (starts that agree on the examples may differ on the test)
    distinct: Dict[bytes, Tuple[List[Array], List]] = {}
    for name, fn, cost in starts:
        mids = [_safe(fn, a) for a, _ in pairs]
        if any(m is None for m in mids):
            continue
        key = b''.join(m.tobytes() + bytes(str(m.shape), 'ascii') for m in mids)
        distinct.setdefault(key, (mids, []))[1].append((name, fn, cost))
    for mids, group in sorted(distinct.values(), key=lambda x: min(c for _, _, c in x[1])):
        if time.time() > deadline:
            break
        rest = [(m, b) for m, (_, b) in zip(mids, pairs)]
        finals = []
        for family in _step_families():
            if time.time() > deadline:
                break
            finals += [p for p in family(rest, bg) if _fits(p.predict, rest)]
        if not finals:
            finals = multi_step_patterns(rest, bg, deadline, depth - 1)
        finals = sorted(finals, key=lambda p: p.complexity)[:MAX_FINALS]
        for name, fn, cost in sorted(group, key=lambda x: x[2])[:MAX_STARTS_PER_RESULT]:
            for p2 in finals:
                def predict(t, f1=fn, f2=p2.predict):
                    m = _safe(f1, t)
                    return None if m is None else f2(m)
                found.append(Pattern('L', f'{name} then {p2.family}:{p2.name}', cost + p2.complexity + 1, predict))
    return sorted(found, key=lambda p: p.complexity)[:MAX_CHAINS]


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
              resize_patterns, relation_patterns, count_patterns, build_patterns,
              role_patterns, ray_patterns)
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
            'H': count_patterns, 'I': build_patterns, 'J': role_patterns, 'K': ray_patterns,
            'M': param_patterns, 'N': symmetry_patterns, 'O': relative_patterns}
COMBINATION_SECONDS = 60     # time budget for combinations, used only when no single pattern fits


def find_patterns(task: Dict, families: Sequence[str] = tuple(FAMILIES),
                  combinations: bool = True) -> Tuple[List[Dict], int]:
    """
    Patterns consistent with every training pair, with their test predictions.

    Exceptions (family X) and multi-step patterns (family L), together
    within MULTI_STEP_SECONDS, and then
    combinations (family G, within COMBINATION_SECONDS) are searched only
    when no single pattern fits.
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
        deadline = time.time() + MULTI_STEP_SECONDS
        pool = partial_pool(pairs, bg, time.time() + POOL_SECONDS)
        collect(exception_patterns(pairs, bg, min(deadline, time.time() + EXCEPTION_SECONDS), pool))
        if not consistent:
            collect(multi_step_patterns(pairs, bg, deadline, pool=pool))
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
