#!/usr/bin/env python3
"""
ARC Elements - find every element of a puzzle grid and describe it

A grid is broken into objects in several ways at once (no single
segmentation fits every puzzle):

- 'color4' / 'color8': connected cells of one colour (4- or 8-connected)
- 'multi4' / 'multi8': connected non-background cells of any colours

Every object gets the same set of properties, computed with the grid
vocabulary; properties that compare an object with the others in its grid
(ranks, how many objects share its shape or colour) make rules such as
"the largest object" or "the object with a unique shape" expressible.

Properties (all hashable values):
  color, colors, n_colors, size, height, width, is_rect, holes,
  shape, shape_canon, pattern, sym_lr, sym_ud, touches_border,
  top, left, size_rank_desc, size_rank_asc, is_largest, is_smallest,
  shape_count, color_count, unique_shape, unique_color, row_rank,
  col_rank, n_objects_same_size
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from grid_vocabulary import Grid, components, enclosed

SEGMENTATIONS = ('color4', 'color8', 'multi4', 'multi8')
Pos = Tuple[int, int]


@dataclass
class Obj:
    cells: FrozenSet[Pos]
    bbox: Tuple[int, int, int, int]            # r0, c0, r1, c1
    props: Dict[str, object] = field(default_factory=dict)

    def mask(self) -> np.ndarray:
        r0, c0, r1, c1 = self.bbox
        m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in self.cells:
            m[r - r0, c - c0] = True
        return m


def _canonical(mask: np.ndarray) -> bytes:
    """Same value for a shape and its rotations / reflections"""
    forms = []
    for k in range(4):
        m = np.rot90(mask, k)
        for f in (m, np.fliplr(m)):
            forms.append(bytes(str(f.shape), 'ascii') + np.packbits(f).tobytes())
    return min(forms)


def _rank(values: List[int], descending: bool) -> List[int]:
    distinct = sorted(set(values), reverse=descending)
    return [distinct.index(v) for v in values]


def segment(a: np.ndarray, bg: int, mode: str) -> List[Obj]:
    """Objects of a grid for one segmentation mode"""
    grid = Grid(a, empty=bg)
    diagonal = mode.endswith('8')
    if mode.startswith('color'):
        values = [v for v in np.unique(a).tolist() if v != bg]
        regions = components(grid, values=values, diagonal=diagonal)
    else:
        regions = components(grid, multicolor=True, background=bg, diagonal=diagonal)
    objs = [Obj(r.cells, r.bbox) for r in regions]
    describe_objects(objs, a, bg)
    return objs


def describe_objects(objs: List[Obj], a: np.ndarray, bg: int):
    h, w = a.shape
    for o in objs:
        r0, c0, r1, c1 = o.bbox
        mask = o.mask()
        colors = Counter(int(a[p]) for p in o.cells)
        crop = np.where(mask, a[r0:r1 + 1, c0:c1 + 1], -1)
        holes = 0
        if mask.shape[0] > 2 and mask.shape[1] > 2:
            inner = Grid(mask.astype(int), empty=0)
            holes = sum(len(reg.cells) for reg in enclosed(inner, walls=[1]))
        o.props.update({
            'color': colors.most_common(1)[0][0],
            'colors': frozenset(colors),
            'n_colors': len(colors),
            'size': len(o.cells),
            'height': r1 - r0 + 1,
            'width': c1 - c0 + 1,
            'is_rect': len(o.cells) == mask.size,
            'holes': holes,
            'shape': bytes(str(mask.shape), 'ascii') + np.packbits(mask).tobytes(),
            'shape_canon': _canonical(mask),
            'pattern': bytes(str(crop.shape), 'ascii') + crop.astype(np.int16).tobytes(),
            'sym_lr': bool(np.array_equal(crop, np.fliplr(crop))),
            'sym_ud': bool(np.array_equal(crop, np.flipud(crop))),
            'touches_border': r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
            'top': r0,
            'left': c0,
        })
    sizes = [o.props['size'] for o in objs]
    desc, asc = _rank(sizes, True), _rank(sizes, False)
    rows, cols = _rank([o.bbox[0] for o in objs], False), _rank([o.bbox[1] for o in objs], False)
    shape_counts = Counter(o.props['shape_canon'] for o in objs)
    color_counts = Counter(o.props['color'] for o in objs)
    size_counts = Counter(sizes)
    for i, o in enumerate(objs):
        o.props.update({
            'size_rank_desc': desc[i],
            'size_rank_asc': asc[i],
            'is_largest': desc[i] == 0,
            'is_smallest': asc[i] == 0,
            'shape_count': shape_counts[o.props['shape_canon']],
            'color_count': color_counts[o.props['color']],
            'unique_shape': shape_counts[o.props['shape_canon']] == 1,
            'unique_color': color_counts[o.props['color']] == 1,
            'row_rank': rows[i],
            'col_rank': cols[i],
            'n_objects_same_size': size_counts[sizes[i]],
        })


# Properties a rule may condition on (absolute positions excluded: they rarely carry over)
RULE_PROPERTIES = ('color', 'colors', 'n_colors', 'size', 'height', 'width', 'is_rect', 'holes',
                   'shape_canon', 'sym_lr', 'sym_ud', 'touches_border', 'size_rank_desc',
                   'size_rank_asc', 'is_largest', 'is_smallest', 'shape_count', 'color_count',
                   'unique_shape', 'unique_color', 'row_rank', 'col_rank', 'n_objects_same_size',
                   'pattern', 'shape')


def background(grids: List[np.ndarray]) -> int:
    """Most common colour over the given grids"""
    counts = Counter()
    for g in grids:
        counts.update(g.ravel().tolist())
    return counts.most_common(1)[0][0]
