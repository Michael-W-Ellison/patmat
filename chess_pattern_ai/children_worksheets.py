#!/usr/bin/env python3
"""
Children's Worksheets - pattern worksheets for young children, transcribed as grids

The items of free "complete the pattern" worksheets for kindergarten,
written as sequence puzzles (sequence_puzzles) with the publishers' answers,
and a runner that solves them and reports the score.

Sources (downloaded 2026-11-05; each item names its sheet and number):
- K5 Learning, kindergarten pattern worksheets (each with an answer page):
  https://www.k5learning.com/free-preschool-kindergarten-worksheets/patterns
  shapes 1-4, letters a-c, objects a-c, cut and paste a-e
- Math Worksheets 4 Kids, pattern worksheets:
  https://www.mathworksheets4kids.com/patterns.php
  the 6 sheets that are complete downloads (the other 36 are previews with
  most items covered): color-easy1, repeating-circle1, repeating-cut-paste1,
  size-circle1, growing-type1-1, repeating-growing-mixed1. These have no
  answer pages; their answers here are mine (each is the only answer a
  repeat or a steady growth gives).

Transcription (done by hand, from the rendered pages): each picture is drawn
as a small grid, the same picture always the same grid:
- geometric shapes and letters: drawn as pixel glyphs (outlines in colour 1
  for K5's black outlines, filled in their colour for coloured shapes);
- pictures (fruit, animals, toys): each distinct picture is a distinct fixed
  8 x 8 pattern of 3 colours (PICTURE_SIZE), a stand-in that keeps what a
  child compares (same picture or not) and nothing else;
- the same picture at two sizes: its grid at 1x and 2x (each cell 2 x 2);
- colour variants of one picture (a train car in green or orange, a
  dinosaur in purple or yellow): the same grid with its main colour changed;
- growing pictures (a caterpillar with 1, 3, 5 segments, a tree with 1, 2,
  3 layers): built from part grids placed side by side or stacked.
Colours are numbered per sheet (names in COLOURS); 0 is the paper.

What a child does that is not transcribed: reading the instruction, and for
"draw the picture" items, drawing; here every answer is a whole panel. Items
with several blanks are scored right only if every blank is right. Choices
(cut-outs, "circle the answer") are given as panels.

Results (sequence_puzzles, symbol_learning and learned_symbols.json as committed with this file):
- ITEMS, the 21 sheets seen while writing sequence_puzzles: 105/105 right.
- HELD_OUT, 33 worksheetfun.com sheets transcribed after sequence_puzzles
  was frozen: first 173/180 (the 7 wrong: the number sequences, numbers
  read as pictures). Then the program was taught digits from counting
  worksheets (digit_lessons) and given the numbers family, and the number
  sequences' panels replaced by the sheet's own cells cut from the page
  (number_cells): 180/180. The number sheet is no longer held out: it was
  examined while the digit reading was developed (the cell cutting was
  fixed on it: slivers of ruled lines left in cells were read as 1s). The
  digits on it are in a type style the program was never shown; what it
  reads them with is learned from other sheets only.
- Without learned symbols (run(..., symbols=None)) the 7 number items are
  wrong, as before.

Run: python children_worksheets.py  (per sheet and per kind of item; -v for every item)
"""

import json
import os
import sys
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

PICTURE_SIZE = 8
COLOURS = {'black': 1, 'red': 2, 'green': 3, 'yellow': 4, 'grey': 5, 'pink': 6, 'orange': 7, 'blue': 8,
           'brown': 9, 'purple': 10, 'teal': 11, 'lavender': 12, 'redorange': 13, 'darkred': 14,
           'lightgreen': 15, 'white': 16}


def _art(text: str, colour: int = 1) -> np.ndarray:
    rows = [r for r in text.strip('\n').split('\n')]
    w = max(len(r) for r in rows)
    return np.array([[colour if ch == '#' else 0 for ch in r.ljust(w)] for r in rows], dtype=int)


# ------------------------------------------------------------------ outline shapes (K5: black outlines)

OUTLINES = {
    'circle': """
..###..
.#...#.
#.....#
#.....#
#.....#
.#...#.
..###..""",
    'rect': """
#####
#...#
#...#
#...#
#...#
#...#
#####""",
    'tri': """
...#...
..#.#..
..#.#..
.#...#.
.#...#.
#.....#
#######""",
    'ttri': """
...#...
...#...
..#.#..
..#.#..
..#.#..
.#...#.
.#...#.
.#...#.
#######""",
    'oval': """
.###.
#...#
#...#
#...#
#...#
#...#
.###.""",
    'diamond': """
...#...
..#.#..
.#...#.
#.....#
.#...#.
..#.#..
...#...""",
    'star': """
...#...
...#...
#######
.#...#.
..#.#..
.#.#.#.
#.....#""",
}
OUTLINES['tri_r'] = np.rot90(_art(OUTLINES['tri']), -1)
OUTLINES['tri_l'] = np.rot90(_art(OUTLINES['tri']), 1)

FILLED = {
    'square': """
#####
#####
#####
#####
#####""",
    'star': """
..#..
#####
.###.
.#.#.
#...#""",
    'circle': """
.###.
#####
#####
#####
.###.""",
    'para': """
..####
.####.
####..""",
    'rect': """
###
###
###
###
###
###""",
    'pent': """
..#..
.###.
#####
.###.
.###.""",
    'window': """
.##.
####
####
.##.""",
}

# 5 x 7 capital letters (K5 letter patterns)
LETTERS = {
    'A': '.###.|#...#|#...#|#####|#...#|#...#|#...#', 'B': '####.|#...#|#...#|####.|#...#|#...#|####.',
    'C': '.###.|#...#|#....|#....|#....|#...#|.###.', 'D': '####.|#...#|#...#|#...#|#...#|#...#|####.',
    'E': '#####|#....|#....|####.|#....|#....|#####', 'F': '#####|#....|#....|####.|#....|#....|#....',
    'G': '.###.|#...#|#....|#.###|#...#|#...#|.###.', 'H': '#...#|#...#|#...#|#####|#...#|#...#|#...#',
    'I': '.###.|..#..|..#..|..#..|..#..|..#..|.###.', 'J': '..###|...#.|...#.|...#.|#..#.|#..#.|.##..',
    'K': '#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#', 'L': '#....|#....|#....|#....|#....|#....|#####',
    'M': '#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#', 'N': '#...#|##..#|#.#.#|#..##|#...#|#...#|#...#',
    'O': '.###.|#...#|#...#|#...#|#...#|#...#|.###.', 'P': '####.|#...#|#...#|####.|#....|#....|#....',
    'Q': '.###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#', 'R': '####.|#...#|#...#|####.|#.#..|#..#.|#...#',
    'S': '.####|#....|#....|.###.|....#|....#|####.', 'T': '#####|..#..|..#..|..#..|..#..|..#..|..#..',
    'U': '#...#|#...#|#...#|#...#|#...#|#...#|.###.', 'W': '#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#',
    'X': '#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#', 'Y': '#...#|#...#|.#.#.|..#..|..#..|..#..|..#..',
    'Z': '#####|....#|...#.|..#..|.#...|#....|#####',
}


LOWER = {'a': '.....|.....|.###.|....#|.####|#...#|.####', 'b': '#....|#....|#.##.|##..#|#...#|#...#|####.',
         'c': '.....|.....|.###.|#....|#....|#...#|.###.'}
DIGITS = {'0': '.###.|#...#|#..##|#.#.#|##..#|#...#|.###.', '1': '..#..|.##..|..#..|..#..|..#..|..#..|.###.',
          '2': '.###.|#...#|....#|...#.|..#..|.#...|#####', '3': '#####|...#.|..#..|...#.|....#|#...#|.###.',
          '4': '...#.|..##.|.#.#.|#..#.|#####|...#.|...#.', '5': '#####|#....|####.|....#|....#|#...#|.###.',
          '6': '..##.|.#...|#....|####.|#...#|#...#|.###.', '7': '#####|....#|...#.|..#..|.#...|.#...|.#...',
          '8': '.###.|#...#|#...#|.###.|#...#|#...#|.###.', '9': '.###.|#...#|#...#|.####|....#|...#.|.##..'}

# worksheetfun shapes (outlines) and the shape variants of its "what comes next" sheets
OUTLINES.update({
    'square': '#######|#.....#|#.....#|#.....#|#.....#|#.....#|#######',
    'heart': '.##.##.|#..#..#|#.....#|#.....#|.#...#.|..#.#..|...#...',
    'hexagon': '..####..|.#....#.|#......#|.#....#.|..####..',
    'ellipse': '..#####..|.#.....#.|#.......#|.#.....#.|..#####..',
    'star6': '....#....|...#.#...|####.####|.#.....#.|..#...#..|.#.....#.|####.####|...#.#...|....#....',
    'wrect': '#########|#.......#|#.......#|#.......#|#########',
    'pentagon': '...#...|..#.#..|.#...#.|#.....#|.#...#.|.#...#.|.#####.',
    'rhombus': '....#....|..##.##..|##.....##|..##.##..|....#....',
    'octagon': '.#####.|#.....#|#.....#|#.....#|#.....#|#.....#|.#####.',
    'tri_split': '...#...|..###..|..###..|.#.#.#.|.#.#.#.|#..#..#|#######',
    'tri_half': '...#...|..###..|..###..|.#.###.|.#.###.|#..####|#######',
    'tri_top': '...#...|..###..|..###..|.#####.|.#...#.|#.....#|#######',
    'sq_half': '#######|####..#|####..#|####..#|####..#|####..#|#######',
    'sq_half_h': '#######|#######|#######|#######|#.....#|#.....#|#######',
    'sq_corner': '#######|#...###|#...###|#.....#|#.....#|#.....#|#######',
    'sq_x': '#######|##...##|#.#.#.#|#..#..#|#.#.#.#|##...##|#######',
    'ring': '..#####..|.#.....#.|#..###..#|#..#.#..#|#..###..#|.#.....#.|..#####..',
    'thickring': '..#####..|.#######.|###...###|###...###|###...###|.#######.|..#####..',
})
for _name in list(OUTLINES):
    if isinstance(OUTLINES[_name], str) and '|' in OUTLINES[_name]:
        OUTLINES[_name] = '\n'.join(OUTLINES[_name].split('|'))
OUTLINES['tri_down'] = np.flipud(_art(OUTLINES['tri_top']))


def picture(name: str, main: Optional[int] = None) -> np.ndarray:
    """
    A stand-in for a picture: a fixed 8 x 8 pattern of 3 colours chosen from the name (the same name, the same
    grid); main: the colour of its main part (for colour variants of one picture)
    """
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    cells = rng.choice(3, size=(PICTURE_SIZE, PICTURE_SIZE), p=[0.3, 0.45, 0.25])
    palette = [0, int(rng.integers(20, 60)), int(rng.integers(60, 99))]
    if main is not None:
        palette[1] = main
    return np.array(palette)[cells]


# ------------------------------------------------------------------ growing pictures

def _hstack(parts, align='bottom'):
    h = max(p.shape[0] for p in parts)
    padded = []
    for p in parts:
        pad = h - p.shape[0]
        top = pad if align == 'bottom' else pad // 2 if align == 'middle' else 0
        padded.append(np.pad(p, ((top, pad - top), (0, 0))))
    return np.hstack(padded)


def _vstack(parts):
    w = max(p.shape[1] for p in parts)
    return np.vstack([np.pad(p, ((0, 0), ((w - p.shape[1]) // 2, w - p.shape[1] - (w - p.shape[1]) // 2)))
                      for p in parts])


def growing(kind: str, n: int) -> np.ndarray:
    C = COLOURS
    if kind == 'caterpillar':           # a head and n body segments to its right
        head = picture('caterpillar head')
        segment = _art('.##.\n####\n####\n####\n.##.', C['green'])
        return _hstack([head] + [segment] * n)
    if kind == 'tree':                  # n layers of leaves on a trunk
        layer = _art('...#...\n..###..\n.#####.\n#######', C['green'])
        trunk = _art('#\n#\n#', C['brown'])
        return _vstack([layer] * n + [trunk])
    if kind == 'train':                 # an engine and n cars to its right
        return _hstack([picture('locomotive')] + [_art('####\n####\n####\n####', C['purple'])] * n)
    if kind == 'building':              # a ground floor with a door and n - 1 floors on top
        floor = _art('#######\n#.###.#\n#######', C['pink'])
        ground = _art('#######\n#.#.#.#\n###.###', C['pink'])
        return np.vstack([floor] * (n - 1) + [ground])
    if kind == 'bar':                   # a tall bar and n squares to its right
        bar = _art('##\n##\n##\n##\n##\n##', C['green'])
        square = _art('##\n##', C['purple'])
        return _hstack([bar] + [square] * n, align='middle')
    if kind == 'ell':                   # an L of 2n - 1 circles: n - 1 up and n - 1 right of the corner
        circle = _art('.#.\n###\n.#.', C['teal'])
        g = np.zeros((3 * n, 3 * n), dtype=int)
        for i in range(n):
            g[3 * i:3 * i + 3, 0:3] = circle
            g[3 * (n - 1):3 * n, 3 * i:3 * i + 3] = circle
        return g
    raise KeyError(kind)


# ------------------------------------------------------------------ tokens

def panel(token: str) -> np.ndarray:
    """
    A panel from its token:
      circle, rect, tri, ttri, oval, diamond, star, tri_r, tri_l   outline shapes (colour 1)
      letter:X                                                    a capital letter
      fill:<shape>:<colour>                                       a filled shape (FILLED) in a colour
      pic:<name>                                                  a picture (stand-in)
      pic:<name>@<colour>                                         a picture with its main part in a colour
      big:<token> / small:<token>                                 a panel at 2x / 1x
      grow:<kind>:<n>                                             a growing picture
    """
    if token.startswith('big:'):
        return np.kron(panel(token[4:]), np.ones((2, 2), int))
    if token.startswith('small:'):
        return panel(token[6:])
    if token.startswith('letter:'):
        c = token[7:]
        return _art('\n'.join({**LETTERS, **LOWER, **DIGITS}[c].split('|')))
    if token.startswith('num:'):            # a number: its digits side by side, one column apart
        digits = [_art('\n'.join(DIGITS[d].split('|'))) for d in token[4:]]
        out = digits[0]
        for d in digits[1:]:
            out = np.hstack([out, np.zeros((out.shape[0], 1), int), d])
        return out
    if token.startswith('fill:'):
        _, shape, colour = token.split(':')
        return _art(FILLED[shape], COLOURS[colour])
    if token.startswith('pic:'):
        name = token[4:]
        if '@' in name:
            name, colour = name.split('@')
            return picture(name, COLOURS[colour])
        return picture(name)
    if token.startswith('grow:'):
        _, kind, n = token.split(':')
        return growing(kind, int(n))
    if token.startswith('cell:'):
        _, r, c = token.split(':')
        return _number_cells()[int(r)][int(c)]
    shape = OUTLINES[token]
    return shape.copy() if isinstance(shape, np.ndarray) else _art(shape)


# ------------------------------------------------------------------ cells cut from a rendered page

NUMBER_CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worksheet_number_cells.json')
NUMBER_SHEET = 'https://www.worksheetfun.com/wp-content/uploads/2016/02/wfun16_number_patterns_7.pdf'
_cells_cache: List = []


def number_cells(pdf_path: str, upright_turns: int = -1) -> List[List[Optional[np.ndarray]]]:
    """
    The cells of a sheet printed as a grid of boxes, row by row: the page rendered (scale 2), turned upright
    (upright_turns quarter turns), its ruled lines found (rows / columns of the page more than half ink), the
    boxes between them cut out (a band counts as a row of boxes where the vertical lines cross it), the ink
    touching a box's edge (pieces of the lines) dropped, and each box cropped to its ink (None: empty)
    """
    import pypdfium2 as pdfium
    from symbol_learning import components
    page = np.array(pdfium.PdfDocument(pdf_path)[0].render(scale=2.0).to_pil().convert('L'))
    dark = np.rot90(page, upright_turns) < 128

    def runs(index):
        out = []
        for i in index:
            if out and i - out[-1][1] <= 2:
                out[-1][1] = i
            else:
                out.append([i, i])
        return out

    h = runs(np.nonzero(dark.mean(axis=1) > 0.5)[0])
    v = runs(np.nonzero(dark.mean(axis=0) > 0.5)[0])
    col_bands = [(a[1] + 1, b[0]) for a, b in zip(v, v[1:]) if b[0] - a[1] > 40]
    rows = []
    for a, b in zip(h, h[1:]):
        r0, r1 = a[1] + 1, b[0]
        if r1 - r0 <= 40 or not all(dark[r0:r1, c0 - 1].mean() > 0.9 for c0, _ in col_bands):
            continue
        row = []
        for c0, c1 in col_bands:
            cell = dark[r0:r1, c0:c1].copy()
            for rs, cs, m in components(cell):
                if rs.start == 0 or cs.start == 0 or rs.stop == cell.shape[0] or cs.stop == cell.shape[1]:
                    cell[rs, cs] &= ~m
            if not cell.any():
                row.append(None)
                continue
            ys, xs = np.nonzero(cell)
            row.append(cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(int))
        rows.append(row)
    return rows


def save_number_cells(pdf_path: str, path: str = NUMBER_CELLS) -> None:
    from symbol_learning import pack
    rows = number_cells(pdf_path)
    with open(path, 'w') as f:
        json.dump({'source': NUMBER_SHEET, 'how': 'children_worksheets.number_cells',
                   'rows': [[None if c is None else pack(c) for c in row] for row in rows]}, f)


def _number_cells() -> List[List[Optional[np.ndarray]]]:
    from symbol_learning import unpack
    if not _cells_cache:
        with open(NUMBER_CELLS) as f:
            data = json.load(f)
        _cells_cache.extend([[None if c is None else unpack(c) for c in row] for row in data['rows']])
    return _cells_cache


# ------------------------------------------------------------------ the items

@dataclass
class Item:
    sheet: str
    number: int
    kind: str                       # repeat, size, growing, growing repeat
    given: List[str]
    answer: List[str]
    choices: List[str] = field(default_factory=list)


def _items(sheet, kind, rows, choices=()):
    out = []
    for i, row in enumerate(rows, 1):
        given, answer = row.split('|')
        out.append(Item(sheet, i, kind, given.split(), answer.split(), list(choices)))
    return out


def _letters(sheet, rows):
    return _items(sheet, 'repeat', [' '.join('letter:' + c for c in g) + ' | ' + ' '.join('letter:' + c for c in a)
                                    for g, a in (r.split('|') for r in rows)])


def _pics(sheet, rows, choices=(), kind='repeat'):
    return _items(sheet, kind, [' '.join('pic:' + t for t in r.split('|')[0].split()) + ' | ' +
                                ' '.join('pic:' + t for t in r.split('|')[1].split()) for r in rows],
                  ['pic:' + c for c in choices])


ITEMS: List[Item] = []
# K5 Learning: shapes (outlines), several blanks each
ITEMS += _items('K5 shapes 1', 'repeat', [
    'circle rect circle rect circle rect | circle rect',
    'circle tri circle tri circle tri | circle tri',
    'oval diamond oval diamond oval diamond | oval diamond oval diamond',
    'tri_r star tri_r star tri_r star | tri_r star',
    'rect circle rect | circle rect circle rect circle',
    'tri_r tri_l tri_r tri_l | tri_r tri_l tri_r tri_l tri_r'])
ITEMS += _items('K5 shapes 2', 'repeat', [
    'tri rect tri rect tri rect | tri rect',
    'star tri star tri star tri | star tri',
    'diamond diamond oval oval diamond diamond | oval oval diamond diamond',
    'tri_r tri_r star star tri_r tri_r | star star tri_r',
    'rect circle rect | circle rect circle rect circle',
    'tri_r tri_r tri_l tri_l | tri_r tri_r tri_l tri_l tri_r'])
ITEMS += _items('K5 shapes 3', 'repeat', [
    'tri oval tri oval tri | oval tri oval',
    'star star tri tri star star | tri tri',
    'diamond diamond rect rect diamond diamond | rect rect diamond diamond',
    'tri_r tri_r star tri_r tri_r star | tri_r tri_r star',
    'rect rect circle rect rect circle | rect rect circle',
    'tri_r tri_r tri_l tri_r tri_r | tri_l tri_r tri_r tri_l'])
ITEMS += _items('K5 shapes 4', 'repeat', [
    'star oval star oval star | oval star oval',
    'oval oval tri tri oval oval | tri tri oval',
    'diamond diamond rect diamond diamond rect | diamond diamond rect diamond',
    'rect rect star rect rect star | rect rect star',
    'rect ttri circle rect ttri circle | rect ttri circle',
    'tri_r ttri tri_l tri_r ttri tri_l | tri_r ttri tri_l'])
# K5 Learning: letters
ITEMS += _letters('K5 letters a', ['XXYYXX|Y', 'AOAOAO|A', 'SSRSSR|S', 'PPPQQQ|P', 'BDDBD|DB', 'LGMLG|ML'])
ITEMS += _letters('K5 letters b', ['RPPRPP|R', 'QZYQZY|Q', 'CDCDCD|C', 'ABBABB|A', 'MMNNM|MN', 'HHKHH|KH'])
ITEMS += _letters('K5 letters c', ['EEFEEF|E', 'UWXUWX|U', 'HIIHII|H', 'SSMSSM|S', 'JKJKJ|KJ', 'TTUUT|TU'])
# K5 Learning: pictures
ITEMS += _pics('K5 objects a', ['apple grapes apple grapes apple | grapes',
                                'cherry cherry strawberry strawberry cherry | cherry',
                                'apple apple cherry apple apple | cherry',
                                'strawberry grapes grapes strawberry grapes | grapes strawberry',
                                'cherry cherry apple apple cherry | cherry apple'])
ITEMS += _pics('K5 objects b', ['squid shrimp squid shrimp squid | shrimp',
                                'crab crab squid crab crab | squid',
                                'fish fish shrimp shrimp fish | fish',
                                'shrimp squid squid shrimp squid | squid shrimp',
                                'fish crab fish crab | fish crab'])
ITEMS += _pics('K5 objects c', ['pizza pizza fries fries pizza | pizza',
                                'pizza burger burger pizza burger | burger',
                                'drumstick fries drumstick fries drumstick | fries',
                                'burger burger drumstick burger burger | drumstick burger',
                                'fries pizza fries pizza | fries pizza'])
# K5 Learning: cut and paste (the cut-outs are the choices)
ITEMS += _pics('K5 cut and paste a', ['engine@green car@orange car@green car@orange | car@green',
                                      'engine@red car@teal car@red car@teal | car@red',
                                      'engine@lavender car@redorange car@lavender car@redorange | car@lavender'],
               choices=['car@green', 'car@red', 'car@lavender', 'car@darkred'])
ITEMS += _pics('K5 cut and paste b', ['donut_pink donut_chocolate_stripes donut_pink | donut_chocolate_stripes',
                                      'donut_chocolate_sprinkles donut_vanilla donut_chocolate_sprinkles | donut_vanilla',
                                      'donut_vanilla donut_green donut_vanilla | donut_green',
                                      'donut_blue donut_pink donut_blue | donut_pink'],
               choices=['donut_chocolate_stripes', 'donut_pink', 'donut_vanilla', 'donut_green'])
ITEMS += _pics('K5 cut and paste c', ['crested@purple crested@yellow crested@purple crested@yellow | crested@purple',
                                      'sailback@yellow sailback@green sailback@yellow sailback@green | sailback@yellow',
                                      'longneck@green longneck@pink longneck@green longneck@pink | longneck@green',
                                      'brachio@teal brachio@lightgreen brachio@teal brachio@lightgreen | brachio@teal'],
               choices=['brachio@teal', 'crested@purple', 'sailback@yellow', 'longneck@green'])
ITEMS += _pics('K5 cut and paste d', ['melon tricolour tricolour melon tricolour tricolour melon tricolour | tricolour',
                                      'twist block twist block twist block twist block | twist',
                                      'lime chocolate lime lime chocolate lime lime lime | chocolate',
                                      'rainbow rainbow block block rainbow rainbow block block | rainbow'],
               choices=['chocolate', 'rainbow', 'tricolour', 'twist'])
ITEMS[-2].kind = 'growing repeat'          # lime chocolate, lime lime chocolate, lime lime lime ...
ITEMS += _pics('K5 cut and paste e', ['soccer basketball tennis soccer | basketball',
                                      'racket paddle paddle racket | paddle',
                                      'target sled target sled | target',
                                      'slide bike slide bike | slide'],
               choices=['paddle', 'slide', 'basketball', 'target'])
# Math Worksheets 4 Kids
ITEMS += _items('MW4K color-easy1', 'repeat', [
    ' '.join('fill:window:' + c for c in row.split('|')[0].split()) + ' | fill:window:' + row.split('|')[1].strip()
    for row in ['green orange green orange green orange green orange green | orange',
                'blue blue yellow yellow blue blue yellow yellow blue | blue',
                'red red red brown brown red red red brown | brown',
                'purple orange purple green purple orange purple green purple | orange',
                'blue pink pink pink yellow blue pink pink pink | yellow']])
for i, (row, choices) in enumerate([('duck parrot duck parrot duck | parrot', ['duck', 'parrot']),
                                    ('flower flower balloon flower flower | balloon', ['balloon', 'flower']),
                                    ('hat hat bucket bucket hat | hat', ['bucket', 'hat']),
                                    ('ball bag ball bag ball | bag', ['bag', 'ball']),
                                    ('purse dress dress purse dress | dress', ['dress', 'purse'])], 1):
    item = _pics('MW4K repeating-circle1', [row], choices)[0]
    item.number = i
    ITEMS.append(item)
ITEMS += _pics('MW4K repeating-cut-paste1', ['boat chicken boat chicken boat | chicken',
                                             'drum car car drum car | car',
                                             'car elephant elephant car elephant | elephant',
                                             'chicken chicken drum chicken chicken | drum',
                                             'boat boat drum drum boat | boat'],
               choices=['car', 'drum', 'elephant', 'chicken', 'boat'])
for i, (thing, row) in enumerate([('guitar', 'big small big small big | small'),
                                  ('drum', 'small small big small small | big'),
                                  ('trumpet', 'small big big small big | big'),
                                  ('piano', 'big small small big small | small'),
                                  ('saxophone', 'big big small small big | big')], 1):
    given, answer = row.split('|')
    ITEMS.append(Item('MW4K size-circle1', i, 'size', [f'{s}:pic:{thing}' for s in given.split()],
                      [f'{answer.strip()}:pic:{thing}'], [f'big:pic:{thing}', f'small:pic:{thing}']))
ITEMS += _items('MW4K growing-type1-1', 'growing', [
    'grow:caterpillar:1 grow:caterpillar:3 grow:caterpillar:5 | grow:caterpillar:7',
    'grow:tree:1 grow:tree:2 grow:tree:3 | grow:tree:4',
    'grow:train:0 grow:train:2 grow:train:4 | grow:train:6',
    'grow:building:1 grow:building:2 grow:building:3 | grow:building:4'])
ITEMS += _items('MW4K repeating-growing-mixed1', 'repeat', [
    'fill:square:blue fill:star:red fill:square:blue fill:star:red fill:square:blue | fill:star:red',
    'grow:bar:1 grow:bar:2 grow:bar:3 | grow:bar:4',
    'fill:circle:orange fill:para:pink fill:para:pink fill:circle:orange fill:para:pink | fill:para:pink',
    'grow:ell:1 grow:ell:2 grow:ell:3 | grow:ell:4',
    'fill:rect:red fill:rect:red fill:pent:blue fill:rect:red fill:rect:red | fill:pent:blue'])
for item in ITEMS[-5:]:
    if item.given[0].startswith('grow:'):
        item.kind = 'growing'


# ------------------------------------------------------------------ held out: worksheetfun.com

# Transcribed after sequence_puzzles was written and frozen, as a test on sheets not seen while writing it.
# Source: https://www.worksheetfun.com/category/critical-thinking/pattern/ (the PDFs linked from its pattern
# posts, downloaded 2026-11-05). No answer pages: the answers are mine (each is what the repeat gives). Worked
# examples (the first item of a sheet, answered on the sheet) are left out. Sheets printed sideways are read
# from the bottom up (the page's left to right), and their shapes named as they stand on the upright page.

def _shapes(sheet, rows, kind='repeat'):
    """rows: 'given ... | answer [| choice choice ...]'"""
    out = []
    for i, row in enumerate(rows, 1):
        parts = [p.split() for p in row.split('|')]
        out.append(Item(sheet, i, kind, parts[0], parts[1], parts[2] if len(parts) > 2 else []))
    return out


HELD_OUT: List[Item] = []
_TRACE_1 = ['circle square circle square | circle | square circle',
            'heart tri heart tri | heart | heart tri',
            'ellipse hexagon ellipse hexagon | ellipse | hexagon ellipse',
            'wrect star6 wrect star6 | wrect | star6 wrect',
            'pentagon rhombus pentagon rhombus | pentagon | rhombus pentagon',
            'octagon circle octagon circle | octagon | circle octagon']
_TRACE_2 = ['rhombus square rhombus square | rhombus | square rhombus',
            'star6 tri star6 tri | star6 | star6 tri',
            'ellipse heart ellipse heart | ellipse | heart ellipse',
            'wrect circle wrect circle | wrect | circle wrect',
            'pentagon rhombus pentagon rhombus | pentagon | rhombus pentagon',
            'octagon heart octagon heart | octagon | heart octagon']
HELD_OUT += _shapes('WSF what comes next 1 (trace)', _TRACE_1)
HELD_OUT += _shapes('WSF what comes next 2 (trace)', _TRACE_2)
HELD_OUT += _shapes('WSF what comes next circle 1', _TRACE_1)
HELD_OUT += _shapes('WSF what comes next circle 2', _TRACE_2)
# The number sequences are not transcribed: their panels are the sheet's own cells, cut from the rendered page
# by number_cells() and kept in NUMBER_CELLS; the program reads them with what it learned (digit_lessons). The
# answers are values (value:<n>), compared with the value the program gives.
HELD_OUT += _shapes('WSF number patterns 7', [
    ' '.join(f'cell:{r}:{c}' for c in range(5)) + f' | value:{v}'
    for r, v in enumerate([11, 12, 16, 12, 27, 16, 20])], kind='number')
HELD_OUT += _shapes('WSF pattern T200 1', [
    'pic:face_sad pic:face_happy pic:face_sad pic:face_happy pic:face_sad | pic:face_happy',
    'pic:face_zigzag pic:face_o pic:face_zigzag pic:face_o pic:face_zigzag | pic:face_o',
    'pic:moon pic:sun pic:moon pic:sun pic:moon | pic:sun',
    'star pic:moon star pic:moon star | pic:moon',
    'pic:face_happy pic:face_zigzag pic:face_happy pic:face_zigzag pic:face_happy | pic:face_zigzag',
    'pic:sun star pic:sun star pic:sun | star'])
HELD_OUT += _shapes('WSF pattern T200 2', [' '.join('letter:' + c for c in g) + ' | letter:' + a for g, a in [
    ('12312312', '3'), ('abcabcab', 'c'), ('45454545', '4'), ('ABBABBA', 'B'), ('56756756', '7')]])
HELD_OUT += _shapes('WSF pattern T200 3', [
    'pic:face_sad pic:face_happy pic:face_sad pic:face_happy pic:face_sad | pic:face_happy',
    'pic:flower heart pic:flower heart pic:flower | heart',
    'pic:moon pic:sun pic:moon pic:sun pic:moon | pic:sun',
    'pic:face_zigzag pic:face_o pic:face_zigzag pic:face_o pic:face_zigzag | pic:face_o',
    'star pic:moon star pic:moon star | pic:moon'])
HELD_OUT += _shapes('WSF what comes next 16-1 (circle)', [
    'star tri star tri star tri | star | star tri',
    'hexagon hexagon square hexagon hexagon square | hexagon | hexagon square',
    'pentagon rhombus pentagon rhombus pentagon rhombus | pentagon | rhombus pentagon',
    'heart star6 heart star6 heart star6 | heart | star6 heart',
    'circle wrect wrect circle wrect wrect | circle | circle wrect',
    'ellipse hexagon hexagon ellipse hexagon hexagon | ellipse | hexagon ellipse'])
HELD_OUT += _shapes('WSF what comes next 16-2', [
    'tri star6 tri star6 tri star6 | tri', 'hexagon hexagon square hexagon hexagon square | hexagon',
    'pentagon rhombus pentagon rhombus pentagon rhombus | pentagon', 'heart star6 heart star6 heart star6 | heart',
    'circle wrect wrect circle wrect wrect | circle', 'ellipse hexagon hexagon ellipse hexagon hexagon | ellipse'])
HELD_OUT += _shapes('WSF what comes next 16-3', [
    'tri star6 star6 tri star6 star6 tri star6 star6 | tri',
    'hexagon ellipse hexagon ellipse hexagon ellipse hexagon ellipse hexagon | ellipse',
    'square pentagon pentagon square pentagon pentagon square pentagon pentagon | square',
    'rhombus hexagon heart rhombus hexagon heart rhombus hexagon heart | rhombus',
    'circle tri circle tri circle tri circle tri circle | tri',
    'ellipse rhombus rhombus ellipse rhombus rhombus ellipse rhombus rhombus | ellipse',
    'hexagon circle star6 hexagon circle star6 hexagon circle star6 | hexagon',
    'tri square pentagon tri square pentagon tri square pentagon | tri',
    'square heart heart square heart heart square heart heart | square'])
HELD_OUT += _shapes('WSF trace T1000 1', [
    'pentagon heart pentagon heart | pentagon | heart pentagon', 'octagon circle octagon circle | octagon | octagon circle',
    'pentagon tri pentagon tri | pentagon | tri pentagon', 'star6 heart star6 heart | star6 | heart star6',
    'ellipse heart ellipse heart | ellipse | ellipse heart', 'octagon heart octagon heart | octagon | heart octagon'])
HELD_OUT += _shapes('WSF trace T1000 2', [
    'ellipse square ellipse square | ellipse | square ellipse', 'pentagon heart pentagon heart | pentagon | pentagon heart',
    'octagon circle octagon circle | octagon | circle octagon', 'star6 heart star6 heart | star6 | star6 heart',
    'pentagon tri pentagon tri | pentagon | tri pentagon', 'octagon heart octagon heart | octagon | heart octagon'])
HELD_OUT += _shapes('WSF trace T35 10', [
    'star6 square star6 square | star6 | star6 square', 'octagon rhombus octagon rhombus | octagon | rhombus octagon',
    'pentagon circle pentagon circle | pentagon | circle pentagon', 'wrect heart wrect heart | wrect | wrect heart'])
HELD_OUT += _shapes('WSF trace T35 11', [
    'octagon rhombus octagon rhombus | octagon | rhombus octagon', 'star6 tri star6 tri | star6 | star6 tri',
    'wrect heart wrect heart | wrect | wrect heart', 'pentagon circle pentagon circle | pentagon | circle pentagon'])
HELD_OUT += _shapes('WSF trace T35 2', [
    'rhombus heart rhombus heart | rhombus | heart rhombus', 'star6 square star6 square | star6 | star6 square',
    'ellipse tri ellipse tri | ellipse | ellipse tri', 'wrect heart wrect heart | wrect | wrect heart',
    'pentagon circle pentagon circle | pentagon | circle pentagon',
    'octagon rhombus octagon rhombus | octagon | rhombus octagon'])
HELD_OUT += _shapes('WSF trace T35 3', [
    'star6 square star6 square | star6 | square star6', 'octagon heart octagon heart | octagon | octagon heart',
    'wrect circle wrect circle | wrect | circle wrect', 'star6 tri star6 tri | star6 | star6 tri',
    'octagon rhombus octagon rhombus | octagon | rhombus octagon', 'wrect heart wrect heart | wrect | wrect heart'])
HELD_OUT += _shapes('WSF trace T35 7', [
    'rhombus square rhombus square | rhombus | square rhombus', 'star6 tri star6 tri | star6 | star6 tri',
    'ellipse heart ellipse heart | ellipse | heart ellipse', 'wrect circle wrect circle | wrect | circle wrect'])
HELD_OUT += _shapes('WSF trace T35 8', [
    'ellipse tri ellipse tri | ellipse | tri ellipse', 'wrect heart wrect heart | wrect | wrect heart',
    'pentagon circle pentagon circle | pentagon | circle pentagon',
    'octagon rhombus octagon rhombus | octagon | rhombus octagon'])
_BUGS = ['ladybug', 'snail', 'sunflower', 'butterfly', 'flower', 'bee', 'spider']
_TOYS = ['ball', 'drum', 'zebra', 'van', 'turtle', 'top', 'rocket', 'car']
HELD_OUT += _pics('WSF picture cut and paste 1', ['butterfly flower butterfly flower butterfly | flower',
                                                  'sunflower ladybug sunflower ladybug sunflower | ladybug',
                                                  'sunflower bee sunflower bee sunflower | bee',
                                                  'snail spider snail spider snail | spider'], _BUGS)
HELD_OUT += _pics('WSF picture cut and paste 2', ['sunflower ladybug ladybug sunflower ladybug | ladybug',
                                                  'snail flower snail flower snail | flower',
                                                  'bee bee sunflower bee bee | sunflower',
                                                  'butterfly spider butterfly spider butterfly | spider'], _BUGS)
HELD_OUT += _pics('WSF picture cut and paste 5', ['turtle zebra turtle zebra turtle | zebra',
                                                  'ball car ball car ball | car', 'drum top drum top drum | top',
                                                  'rocket van rocket van rocket | van'], _TOYS)
HELD_OUT += _pics('WSF picture cut and paste 6', ['turtle zebra zebra turtle zebra | zebra',
                                                  'van top van top van | top', 'ball ball car ball ball | car',
                                                  'rocket drum rocket drum rocket | drum'], _TOYS)
HELD_OUT += _pics('WSF picture cut and paste 7', ['ball car ball car ball | car', 'drum top top drum top | top',
                                                  'turtle zebra turtle zebra turtle | zebra',
                                                  'rocket rocket van rocket rocket | van'], _TOYS)
HELD_OUT += _shapes('WSF what shape comes next T25 1', [
    'ellipse hexagon ellipse hexagon ellipse | hexagon', 'octagon tri tri octagon tri | tri',
    'pentagon hexagon octagon pentagon hexagon | octagon', 'circle circle square circle circle | square',
    'star6 heart star6 heart star6 | heart', 'pentagon rhombus circle pentagon rhombus | circle'])
HELD_OUT += _shapes('WSF what shape comes next T25 2', [
    'ellipse hexagon hexagon ellipse hexagon | hexagon', 'star6 heart star6 heart star6 | heart',
    'pentagon hexagon pentagon hexagon pentagon | hexagon', 'circle circle square circle circle | square',
    'octagon tri tri octagon tri | tri', 'pentagon rhombus circle pentagon rhombus | circle'])
HELD_OUT += _shapes('WSF what shape comes next T25 3', [
    'ellipse ellipse hexagon hexagon ellipse | ellipse', 'octagon tri tri octagon tri | tri',
    'pentagon hexagon heart pentagon hexagon | heart', 'circle circle square circle circle | square',
    'star6 heart star6 heart star6 | heart', 'pentagon rhombus circle pentagon rhombus | circle'])
HELD_OUT += _shapes('WSF what comes next T25 4', [
    'tri tri_split tri tri_split tri | tri_split', 'square sq_half square sq_half square | sq_half',
    'circle ring circle ring circle | ring', 'square sq_corner square sq_corner square | sq_corner',
    'square sq_x square sq_x square | sq_x', 'tri_top tri_down tri_top tri_down tri_top | tri_down'])
HELD_OUT += _shapes('WSF what comes next T25 5', [
    'tri tri_split tri_split tri tri_split | tri_split', 'square sq_half square sq_half square | sq_half',
    'ring ring circle ring ring | circle', 'square sq_corner sq_half square sq_corner | sq_half',
    'square sq_x square sq_x square | sq_x', 'tri tri_top tri_down tri tri_top | tri_down'])
HELD_OUT += _shapes('WSF what comes next T25 6', [
    'tri tri_split tri_half tri tri_split | tri_half', 'square sq_half sq_half_h square sq_half | sq_half_h',
    'circle ring thickring circle ring | thickring', 'square sq_corner sq_half square sq_corner | sq_half',
    'square sq_x square sq_x square | sq_x', 'tri tri_top tri_down tri tri_top | tri_down'])
_CUT_A = 'hexagon heart tri pentagon rhombus square ellipse circle'
_CUT_B = 'hexagon octagon heart pentagon circle star6 square tri'
HELD_OUT += _shapes('WSF shape cut and paste 5', [r + ' | ' + _CUT_A for r in [
    'circle square circle square circle | square', 'heart tri heart tri heart | tri',
    'ellipse hexagon ellipse hexagon ellipse | hexagon', 'pentagon rhombus pentagon rhombus pentagon | rhombus']])
HELD_OUT += _shapes('WSF shape cut and paste 6', [r + ' | ' + _CUT_B for r in [
    'pentagon star6 pentagon star6 pentagon | star6', 'octagon tri octagon tri octagon | tri',
    'heart hexagon heart hexagon heart | hexagon', 'circle square circle square circle | square']])
HELD_OUT += _shapes('WSF shape cut and paste 7', [r + ' | ' + _CUT_B for r in [
    'pentagon star6 star6 pentagon star6 | star6', 'octagon square octagon square octagon | square',
    'heart heart hexagon heart heart | hexagon', 'circle tri circle tri circle | tri']])
HELD_OUT += _shapes('WSF shape cut and paste 8', [r + ' | ' + _CUT_B for r in [
    'heart star6 heart star6 heart | star6', 'octagon tri tri octagon tri | tri',
    'pentagon hexagon pentagon hexagon pentagon | hexagon', 'circle circle square circle circle | square']])


# ------------------------------------------------------------------ running

def run(items: List[Item] = ITEMS, verbose: bool = False, symbols='learned', **solve_kwargs) -> Dict:
    """symbols: the SymbolMemory to read symbols with ('learned': the one digit_lessons saved, None: none)"""
    from sequence_puzzles import SequencePuzzle, solve
    if symbols == 'learned':
        from digit_lessons import LEARNED
        from symbol_learning import SymbolMemory
        symbols = SymbolMemory(LEARNED) if os.path.exists(LEARNED) else None
    results = []
    for item in items:
        values = [int(t[6:]) for t in item.answer if t.startswith('value:')]
        truth = [] if values else [panel(t) for t in item.answer]
        choices = [panel(c) for c in item.choices]
        puzzle = SequencePuzzle([panel(t) for t in item.given], len(item.answer), choices)
        answer = solve(puzzle, symbols=symbols, **solve_kwargs)
        if answer is None:
            outcome = 'no answer'
        elif values:
            outcome = 'right' if answer.get('values') == values else 'wrong'
        elif all(a.shape == t.shape and np.array_equal(a, t) for a, t in zip(answer['panels'], truth)):
            outcome = 'right'
        else:
            outcome = 'wrong'
        results.append((item, outcome, answer))
        if verbose:
            print(f'{item.sheet:30s} {item.number}  {item.kind:15s} {outcome:9s} '
                  f'{answer["family"] + ": " + answer["rule"] if answer else ""}')
    return {'results': results,
            'right': sum(o == 'right' for _, o, _ in results), 'wrong': sum(o == 'wrong' for _, o, _ in results),
            'unanswered': sum(o == 'no answer' for _, o, _ in results), 'items': len(results)}


def report(summary: Dict):
    def table(key):
        groups: Dict[str, List[str]] = {}
        for item, outcome, _ in summary['results']:
            groups.setdefault(key(item), []).append(outcome)
        for name, outs in groups.items():
            print(f'  {name:32s} right {outs.count("right")}/{len(outs)}, wrong {outs.count("wrong")}, '
                  f'no answer {outs.count("no answer")}')
    print('by sheet:')
    table(lambda i: i.sheet)
    print('by kind:')
    table(lambda i: i.kind)
    print(f'total: right {summary["right"]}/{summary["items"]}, wrong {summary["wrong"]}, '
          f'no answer {summary["unanswered"]}')


if __name__ == '__main__':
    for title, items in (('Sheets seen while writing sequence_puzzles (K5 Learning, Math Worksheets 4 Kids)', ITEMS),
                         ('Held out (worksheetfun.com)', HELD_OUT)):
        print(f'== {title}')
        report(run(items, verbose='-v' in sys.argv))
