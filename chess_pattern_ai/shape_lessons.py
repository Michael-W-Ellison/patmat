#!/usr/bin/env python3
"""
Shape Lessons - teaching the program the names of shapes from worksheets

As for digits and letters (digit_lessons, letter_lessons): lessons read from
rendered K5 Learning kindergarten shape worksheets; where each exercise is on
a page is given per sheet type here, and what is in it is found by the
program. What is known beforehand: the names of shapes as words
(symbol_learning.SHAPES), as the alphabet was for letters; the names on the
sheets are read with the letters it has learned (SymbolMemory.read_word).
Shapes are compared by silhouette (an outline and a filled shape alike).

Lessons, in teaching order:
- a worksheetfun shapes chart (https://www.worksheetfun.com): ten shapes,
  each with its name printed under it (read); the shape above a name is
  learned under it (heart, pentagon, hexagon, octagon are taught only here).
- K5 trace-color-shapes b-g (one sheet per shape): the title names the
  shape; the small coloured shapes on the sheet are it, turned and sized.
- tracing-drawing circles, diamonds, ovals, rectangles, squares, triangles
  (one sheet per shape): the title names the shape (read); the dotted
  outlines on the sheet, joined and filled, are that shape.
- names-of-shapes a, b ("write the names of the shapes"): a word bank of
  names, and under each shape its name's first letter and one dash per
  letter: the name in the bank of that length (of several, those the first
  letter may be).
- names-of-shapes c and match-shapes-to-names ("match the shapes with their
  names"): a column of shapes and a column of their names, shuffled
  (matching lessons, once some shapes are recognised).
- shapes 1, 2 ("identify the correct shape"): a name and four shapes, one of
  them the named shape (choice lessons).
- shapes 4 (the same with a coloured shape in place of the name) and
  match-similar-shapes (a column of coloured shapes and one of outlines,
  shuffled): once the coloured shape is recognised, as choice and matching
  lessons.

Test (children_worksheets): shapes 3 (name and four shapes) and shapes in
patterns 1-4 (draw the 2 shapes that come next), with cells cut from the pages.

Results (as committed): 102 shape examples of 9 names (circle 13, oval 12,
square 12, rectangle 19, triangle 18, diamond 14, star 10, heart 1, hexagon
3), all labelled right (checked by eye). Of the K5 sheets the tracing sheets,
the trace-color sheets, the captions and the chart teach; the name and
shape matching sheets confirm; of the choice rows most confirm and two end
in contradictions (a pentagon read as a hexagon; an oval as a rectangle).
Not taught (no sheet among these shows them named): pentagon, octagon,
parallelogram, cross, trapezoid. Shapes no lesson taught are read as the
nearest taught shape: at SIZE x SIZE a hexagon is near a circle.

Run: python shape_lessons.py [cache directory]  (adds shapes to the memory saved by the earlier lessons, LEARNED)
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np

from digit_lessons import LEARNED, fetch
from symbol_learning import SHAPES, SymbolMemory, colourful, components, ink, silhouette, symbols_of

K5 = 'https://www.k5learning.com/worksheets/kindergarten/{}.pdf'
K5_TOP = 'https://www.k5learning.com/worksheets/{}.pdf'
SCALE = 2.0


def page(path: str) -> np.ndarray:
    import pypdfium2 as pdfium
    return np.array(pdfium.PdfDocument(path)[0].render(scale=SCALE).to_pil().convert('RGB'))


def _runs(flags, least: int = 1):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= least:
                out.append((start, i))
            start = None
    return out


def table(region: np.ndarray, inset: int = 6) -> List[List[np.ndarray]]:
    """
    The cells of the largest ruled table in region, row by row: its lines are the rows and columns of the
    table's own ink (its largest connected piece) that are more than 60% ink; each cell cut inside them
    """
    mask = ink(region)
    parts = sorted(components(mask), key=lambda c: -(c[0].stop - c[0].start) * (c[1].stop - c[1].start))
    if not parts:
        return []
    rs, cs, grid = parts[0]
    hlines = _runs(grid.mean(axis=1) > 0.6)
    vlines = _runs(grid.mean(axis=0) > 0.6)
    cells = []
    for (a0, a1), (b0, b1) in zip(hlines, hlines[1:]):
        row = []
        for (c0, c1), (d0, d1) in zip(vlines, vlines[1:]):
            if b0 - a1 > 2 * inset and d0 - c1 > 2 * inset:
                row.append(region[rs.start + a1 + inset:rs.start + b0 - inset, cs.start + c1 + inset:cs.start + d0 - inset])
        if row:
            cells.append(row)
    return cells


def shapes_in(cell: np.ndarray, least: int = 20) -> List[np.ndarray]:
    """The shapes drawn in a cell, left to right (pieces one above another together; specks dropped)"""
    return [m for m in symbols_of(ink(cell)) if m.shape[0] >= least and m.shape[1] >= least // 2]


TINT = 25               # coloured shapes: ink whose saturation is above this (brown and grey-blue too; text is black)


def tinted(region: np.ndarray) -> np.ndarray:
    rgb = region[..., :3].astype(int)
    return ink(region) & (rgb.max(axis=2) - rgb.min(axis=2) > TINT)


def coloured_shapes(region: np.ndarray, least: int = 30) -> List[np.ndarray]:
    """The coloured shapes in region (the dark details on them, a face, filled in), top to bottom"""
    from scipy import ndimage
    colour = tinted(region)
    filled = ndimage.binary_fill_holes(colour)
    found = [(rs.start, cs.start, m) for rs, cs, m in components(filled) if min(m.shape) >= least]
    return [m for _, _, m in sorted(found, key=lambda e: (e[0], e[1]))]


def words_in(mask: np.ndarray, gap: Optional[int] = None) -> List[np.ndarray]:
    """The words in a band of ink, left to right: runs of symbols less than gap cells apart (default: a third
    of the band's height)"""
    if gap is None:
        gap = max(8, mask.shape[0] // 3)
    cols = mask.any(axis=0)
    out = []
    for c0, c1 in _runs(cols):
        if out and c0 - out[-1][1] < gap:
            out[-1] = (out[-1][0], c1)
        else:
            out.append((c0, c1))
    return [mask[:, c0:c1] for c0, c1 in out]


def _bands(mask: np.ndarray, gap: int = 15):
    rows = mask.any(axis=1)
    out = []
    for r0, r1 in _runs(rows):
        if out and r0 - out[-1][1] < gap:
            out[-1] = (out[-1][0], r1)
        else:
            out.append((r0, r1))
    return out


def _dark(region: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    colour = colourful(region)
    return ink(region) & ~ndimage.binary_dilation(colour, iterations=3)


def word_bank_lessons(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    """names-of-shapes a, b: the word bank above the table, and the captions (first letter and a dash per
    letter) under the shapes"""
    p = page(path)
    grid_top = 500
    bank_mask = _dark(p[260:grid_top])
    bank = []
    for r0, r1 in _bands(bank_mask)[1:]:                     # the first band is the instruction
        for w in words_in(bank_mask[r0:r1]):
            word = memory.read_word(w, SHAPES)
            if word:
                bank.append(word)
    out = []
    for i, cell in enumerate(c for row in table(p[grid_top:]) for c in row):
        shapes = coloured_shapes(cell)
        dark = _dark(cell)
        from scipy import ndimage
        filled = ndimage.binary_fill_holes(tinted(cell))
        rows = np.nonzero(filled.any(axis=1))[0]
        if len(rows):
            dark[:rows.max() + 6] = False                    # the caption is below the shape
        marks = [m for m in symbols_of(dark)]
        letters = [m for m in marks if m.shape[0] >= 20]
        dashes = [m for m in marks if m.shape[0] < 12 and 8 <= m.shape[1] <= 40]
        src = f'K5 {name} card {i + 1}'
        if len(shapes) != 1 or not letters:
            out.append({'kind': 'skipped', 'source': src, 'why': f'{len(shapes)} shapes, {len(letters)} letters'})
            continue
        # the name of the caption's length; of several, the ones whose first letter it may be
        fits = [w for w in bank if len(w) == 1 + len(dashes)]
        if len(fits) > 1:
            initials = {str(a).lower() for a, _ in memory.alternatives(letters[0], 'upper', 3)}
            fits = [w for w in fits if w[0] in initials]
        if len(fits) != 1:
            out.append({'kind': 'skipped', 'source': src, 'why': f'names fitting the caption: {fits} (bank {bank})'})
            continue
        out.append({'kind': 'order', 'symbols': [shapes[0]], 'names': fits, 'source': src})
    return out


def matching_lessons(path: str, memory: SymbolMemory, name: str, coloured: bool) -> List[Dict]:
    """A column of shapes (coloured or drawn) and a column of their names, shuffled"""
    p = page(path)
    p = p[260:]
    half = p.shape[1] // 2
    right = _dark(p[:, half:])
    words, tops = [], []
    for r0, r1 in _bands(right):
        word = memory.read_word(right[r0:r1], SHAPES)
        if word:
            words.append(word)
            tops.append((r0, r1))
    if not words:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'no names read'}]
    lo, hi = tops[0][0] - 40, tops[-1][1] + 40
    left = p[lo:hi, :half]
    shapes = coloured_shapes(left) if coloured else [m for r0, r1 in _bands(ink(left), 10)
                                                     for m in shapes_in(left[r0:r1])]
    if len(shapes) != len(words):
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': f'{len(shapes)} shapes, {len(words)} names'}]
    return [{'kind': 'matching', 'numerals': shapes, 'counts': words, 'domain': 'shape', 'source': f'K5 {name}'}]


def choice_lessons(path: str, memory: SymbolMemory, name: str, named: bool = True) -> List[Dict]:
    """shapes 1-4: a name (or a coloured shape) and four shapes, one of them the same shape"""
    out = []
    for i, row in enumerate(table(page(path)), 1):
        if len(row) != 2:
            continue
        candidates = shapes_in(row[1])
        src = f'K5 {name} row {i}'
        if len(candidates) != 4:
            out.append({'kind': 'skipped', 'source': src, 'why': f'{len(candidates)} shapes'})
            continue
        if named:
            word = memory.read_word(_dark(row[0]), SHAPES)
            if word is None:
                out.append({'kind': 'skipped', 'source': src, 'why': 'name not read'})
                continue
            out.append({'kind': 'choice', 'numerals': candidates, 'count': word, 'domain': 'shape', 'source': src})
        else:
            model = coloured_shapes(row[0])
            if len(model) != 1:
                out.append({'kind': 'skipped', 'source': src, 'why': 'no model shape'})
                continue
            out.append({'kind': 'shape choice', 'model': model[0], 'symbols': candidates, 'source': src})
    return out


def similar_lessons(path: str, name: str) -> List[Dict]:
    """match-similar-shapes: a column of coloured shapes and one of drawn shapes, shuffled"""
    p = page(path)[260:]
    half = p.shape[1] // 2
    models = coloured_shapes(p[:, :half])
    if not models:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'no coloured shapes'}]
    rows = [(rs.start, rs.stop) for rs, cs, m in components(tinted(p[:, :half])) if min(m.shape) >= 30]
    lo, hi = min(r[0] for r in rows) - 30, max(r[1] for r in rows) + 30
    right = p[lo:hi, half:]
    drawn = [m for r0, r1 in _bands(ink(right), 10) for m in shapes_in(right[r0:r1])]
    if len(drawn) != len(models):
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': f'{len(models)} coloured, {len(drawn)} drawn'}]
    return [{'kind': 'shape matching', 'models': models, 'symbols': drawn, 'source': f'K5 {name}'}]


TRACED = 225            # tracing sheets: the dotted outlines are light grey, ink darker than this
JOIN = 12               # tracing sheets: dashes this far apart are joined into one outline


def tracing_lessons(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    """
    "Tracing and drawing circles" (one sheet per shape): the title names the shape; the dotted outlines on
    the sheet, joined and filled, are that shape at several sizes
    """
    from scipy import ndimage
    p = page(path)
    colour = colourful(p) & ink(p)
    shape = None
    for r0, r1 in _bands(colour[100:300], 6):                 # the title, below the logo
        for w in words_in(colour[100 + r0:100 + r1]):
            shape = shape or memory.read_word(w, SHAPES)
    if shape is None:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'title not read'}]
    rgb = p[..., :3].astype(int)
    light = (rgb.mean(axis=2) < TRACED) & ~ndimage.binary_dilation(colourful(p), iterations=3)
    # below the instructions: the traced word, then the outlines, then "Draw some ... below" (bands of ink
    # at least 20 rows tall, 20 rows apart)
    parts = [(a, b) for a, b in _bands(light[300:], 20) if b - a >= 20]
    if len(parts) < 2:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'no outlines found'}]
    region = light[300 + parts[1][0]:300 + parts[1][1]]
    # dashes joined: widened, filled, shrunk back
    solid = ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_dilation(region, iterations=JOIN)),
                                   iterations=JOIN)
    found = [m for rs, cs, m in components(solid)
             if min(m.shape) > 60 and max(m.shape) < 3 * min(m.shape) and m.mean() > 0.4]
    if found:   # copies of one shape fill their boxes alike; one that does not is likely two touching
        middle = float(np.median([m.mean() for m in found]))
        found = [m for m in found if abs(m.mean() - middle) <= 0.06]
    if not found:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'no outlines found'}]
    return [{'kind': 'order', 'symbols': found, 'names': [shape] * len(found), 'source': f'K5 {name}'}]


def _joined(mask: np.ndarray, join: Optional[int] = None) -> np.ndarray:
    """Dotted outlines made solid: the ink widened by join (default JOIN), its insides filled, shrunk back"""
    from scipy import ndimage
    join = JOIN if join is None else join
    return ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=join)),
                                  iterations=join)


def chart_lessons(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    """
    A chart of shapes with their names printed under them (worksheetfun "Shapes" tracing chart): each
    name read, and the shape just above it (a dotted outline, joined and filled) learned under it
    """
    from scipy import ndimage
    from symbol_learning import is_frame
    p = page(path)
    mask = ink(p) & ~colourful(p)
    for rs, cs, m in components(mask):          # the page's border
        if is_frame(m):
            mask[rs, cs] &= ~m
    solid = _joined(mask, CHART_JOIN)
    shapes = [(rs, cs, m) for rs, cs, m in components(solid) if min(m.shape) > 80 and m.mean() > 0.3]
    rest = mask.copy()
    for rs, cs, m in shapes:
        rest[rs, cs] &= ~ndimage.binary_dilation(m, iterations=8)
    out = []
    for r0, r1 in _bands(rest, 10):
        if r1 - r0 < 15:
            continue
        for w_start, w_end in _word_spans(rest[r0:r1]):
            word = memory.read_word(rest[r0:r1, w_start:w_end], SHAPES)
            if word is None:
                continue
            centre = (w_start + w_end) / 2
            above = [(r0 - rs.stop, m) for rs, cs, m in shapes
                     if cs.start <= centre <= cs.stop and 0 <= r0 - rs.stop <= 80]
            if len(above) == 1:
                out.append({'kind': 'order', 'symbols': [above[0][1]], 'names': [word],
                            'source': f'{name} {word}'})
    return out or [{'kind': 'skipped', 'source': name, 'why': 'no named shapes found'}]


def _word_spans(mask: np.ndarray):
    gap = max(8, mask.shape[0] // 3)
    out = []
    for c0, c1 in _runs(mask.any(axis=0)):
        if out and c0 - out[-1][1] < gap:
            out[-1] = (out[-1][0], c1)
        else:
            out.append((c0, c1))
    return out


def trace_color_lessons(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    """
    "Trace and color in the stars" (one sheet per shape): the title names the shape (read); the small
    coloured shapes scattered on the sheet are that shape, turned and sized differently. A sheet with fewer
    than 3 coloured shapes (a leaf on the circles sheet) is not used
    """
    p = page(path)
    shape = None
    colour = colourful(p) & ink(p)
    dark = ink(p) & ~colourful(p)
    for mask in (colour, dark):                       # the title, or else the instruction under it
        for r0, r1 in _bands(mask[100:300], 6):
            for w in words_in(mask[100 + r0:100 + r1]):
                shape = shape or memory.read_word(w, SHAPES)
    if shape is None:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'title not read'}]
    found = coloured_shapes(p[400:], least=20)
    if len(found) < 3:
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': f'{len(found)} coloured shapes'}]
    return [{'kind': 'order', 'symbols': found, 'names': [shape] * len(found), 'source': f'K5 {name}'}]


CHART_JOIN = 6          # the chart's dashes are close: joining wider merges the names into the shapes
WSF_CHART = 'https://www.worksheetfun.com/wp-content/uploads/2019/10/wfun19_tracing_shapes_T100_3.pdf'


def lessons(cache: str, memory: SymbolMemory) -> List[Dict]:
    out = []
    out += chart_lessons(fetch(WSF_CHART, cache), memory, 'worksheetfun shapes chart')
    for s in 'bcdefg':
        out += trace_color_lessons(fetch(K5.format(f'trace-color-shapes-{s}'), cache), memory,
                                   f'trace-color-shapes-{s}')
    for s in ('circles', 'diamonds', 'ovals', 'rectangles', 'squares', 'triangles'):
        out += tracing_lessons(fetch(K5_TOP.format(f'kindergarten-tracing-drawing-{s}'), cache), memory,
                               f'tracing-drawing-{s}')
    for sheet in ('names-of-shapes-a', 'names-of-shapes-b'):
        out += word_bank_lessons(fetch(K5.format(sheet), cache), memory, sheet)
    out += matching_lessons(fetch(K5.format('names-of-shapes-c'), cache), memory, 'names-of-shapes-c', True)
    out += matching_lessons(fetch(K5_TOP.format('kindergarten-match-shapes-to-names'), cache), memory,
                            'match-shapes-to-names', False)
    for sheet in ('shapes-1', 'shapes-2'):
        out += choice_lessons(fetch(K5.format(sheet), cache), memory, sheet)
    out += choice_lessons(fetch(K5.format('shapes-4'), cache), memory, 'shapes-4', named=False)
    out += similar_lessons(fetch(K5_TOP.format('kindergarten-match-similar-shapes'), cache), 'match-similar-shapes')
    return out


def teach(cache: str, path: str = LEARNED, verbose: bool = True) -> SymbolMemory:
    memory = SymbolMemory(path if os.path.exists(path) else None)
    memory.examples = [e for e in memory.examples if e['meaning'] not in SHAPES]          # shapes anew
    plan = lessons(cache, memory)
    taught = [l for l in plan if l['kind'] != 'skipped']
    outcomes = memory.teach(taught)
    if verbose:
        for l in plan:
            if l['kind'] == 'skipped':
                print(f'{l["source"]:40s} skipped: {l["why"]}')
        for l, r in zip(taught, outcomes):
            print(f'{l["source"]:40s} {l["kind"]:14s} {r["outcome"]} {r.get("learned", r.get("why", ""))}')
        print('known:', {k: v for k, v in memory.meanings().items() if k in SHAPES})
    memory.path = path
    memory.save(path)
    return memory


if __name__ == '__main__':
    teach(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_worksheets'))
