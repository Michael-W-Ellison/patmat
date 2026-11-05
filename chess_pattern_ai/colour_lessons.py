#!/usr/bin/env python3
"""
Colour Lessons - teaching the program the names of colours from worksheets

As for digits, letters and shapes (digit_lessons, letter_lessons,
shape_lessons): lessons read from rendered K5 Learning kindergarten colour
worksheets; where each exercise is on a page is given per sheet type, and
what is in it is found by the program. What is known beforehand: the names
of colours as words (symbol_learning.COLOURS); the names on the sheets are
read with the letters the program has learned. A colour is the median colour
of an object (in CIELAB), so details on it do not count.

Lessons, in teaching order:
- "What color is each dinosaur / truck / egg / drawing tool?" (match-write
  a, b, c, e): beside each coloured object, its colour's first letter and one
  dash per letter; the colour of that first letter and length (if only one
  name fits), learned from the object the caption is beside: of the objects
  on its line, the one nearest across, if clearly nearer than the next.
- "What color is each bird?" (match-write d): the captions are not beside
  their birds; the sheet's objects and captions as a matching lesson.
- "Real-world colors" (colors 1-3): colour words and pictures of things in
  those colours, shuffled (matching lessons).
- "Matching colors to names" (match-colors-to-names 1): swatches and names,
  shuffled (a matching lesson). Sheets 2 and 3 are not used: their names are
  dotted, to trace, and are not read (joining the dots does not make them
  readable to the program).
Matching lessons left undecided are taken two at a time (learn_across): the
same colour on two sheets is the one name both have left.

Test (children_worksheets): colors 4 and match-colors-to-names 4 (words and
pictures to pair), sorting colors 1-4 (pictures to sort under 3 colour
words), and the Math Worksheets 4 Kids airplane windows (colour patterns).

Results (as committed): 40 colour examples of 10 names (red 4, orange 4,
yellow 6, green 8, blue 6, purple 4, pink 4, brown 1, black 2, grey 1), all
labelled right (checked by eye, as swatches). The captions teach red,
orange, yellow, green, blue, purple, pink; the bird sheet (matching) another
orange; colors-1 brown; black from colors-2 with match-colors-to-names-1
across the two sheets (the tyre and the black swatch); then colors-2 grey.
White is not taught: on colors-3 it is undecided with a lighter brown.
COLOUR_CONFIDENT (20, CIELAB) was chosen on the objects of colors 1-3,
recognised among each sheet's names: 13 of 18 recognised, none wrongly (at
12, 6; at 28, one wrong).

Run: python colour_lessons.py [cache directory]  (adds colours to the memory saved by the earlier lessons, LEARNED)
"""

import os
import sys
from typing import Dict, List

import numpy as np

from digit_lessons import LEARNED, fetch
from shape_lessons import _bands, _dark, _runs, words_in
from symbol_learning import COLOURS, SymbolMemory, components, symbols_of

K5 = 'https://www.k5learning.com/worksheets/kindergarten/{}.pdf'
SCALE = 2.0
HEADER = 300            # pages at SCALE 2: the title and instructions lie above this row
PAPER = 235             # objects: pixels darker than this in some channel (white paper is not)
SMALL_PART = 0.35       # objects: less ink than this share of the median object is a piece, not an object
NEAREST = 1.3           # captions: the nearest object must be this many times nearer than the next


def page(path: str) -> np.ndarray:
    import pypdfium2 as pdfium
    return np.array(pdfium.PdfDocument(path)[0].render(scale=SCALE).to_pil().convert('RGB'))


def objects_in(region: np.ndarray, least: int = 40) -> List[Dict]:
    """
    The objects in region: connected parts that are not paper, their insides filled, at least least cells
    in height and width: {'sample' (RGBA: the object's pixels, alpha its mask), 'top', 'left', 'centre'}
    """
    from scipy import ndimage
    marked = region[..., :3].min(axis=2) < PAPER
    filled = ndimage.binary_fill_holes(marked)
    out = []
    for rs, cs, m in components(filled):
        if min(m.shape) < least:
            continue
        rgba = np.dstack([region[rs, cs, :3], (m * 255).astype(np.uint8)])
        out.append({'sample': rgba, 'top': rs.start, 'left': cs.start,
                    'centre': ((rs.start + rs.stop) / 2, (cs.start + cs.stop) / 2), 'mask': m})
    if out:      # pieces much smaller than the rest (a stem, a handle) are not objects of their own
        middle = float(np.median([o['mask'].sum() for o in out]))
        out = [o for o in out if o['mask'].sum() >= SMALL_PART * middle]
    return sorted(out, key=lambda o: (o['top'], o['left']))


def captions(region: np.ndarray, memory: SymbolMemory) -> List[Dict]:
    """
    The captions in region: a letter followed on its line by dashes (one per further letter); the colour
    name of that length, of several the ones beginning with the likeliest reading of the letter that begins
    any, if only one fits: {'name', 'centre'}
    """
    dark = _dark(region)
    marks = []
    for rs, cs, m in components(dark):
        h, w = rs.stop - rs.start, cs.stop - cs.start
        if h >= 20 and w <= 2 * h:
            marks.append(('letter', rs, cs, m))
        elif h < 12 and 8 <= w <= 60:
            marks.append(('dash', rs, cs, m))
    letters = [k for k in marks if k[0] == 'letter']
    dashes = [k for k in marks if k[0] == 'dash']
    out = []
    for _, rs, cs, m in letters:
        line = []
        x = cs.stop
        for _, drs, dcs, _ in sorted(dashes, key=lambda d: d[2].start):
            if dcs.start >= x and dcs.start - x < 40 and drs.start >= rs.start and drs.stop <= rs.stop + 12:
                line.append(dcs)
                x = dcs.stop
        if not line:
            continue
        length = 1 + len(line)
        fits = [c for c in COLOURS if len(c) == length]
        if len(fits) > 1:       # the likeliest reading of the letter that begins one of them
            for letter, _ in memory.alternatives(m, 'upper', 3):
                begun = [c for c in fits if c[0] == str(letter).lower()]
                if begun:
                    fits = begun
                    break
        if len(fits) == 1:
            out.append({'name': fits[0], 'centre': ((rs.start + rs.stop) / 2, (cs.start + x) / 2)})
    return out


def caption_lessons(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    p = page(path)[HEADER:]
    objects = objects_in(p)
    out = []
    for i, cap in enumerate(captions(p, memory), 1):
        src = f'K5 {name} caption {i}'
        # the objects on the caption's line (their rows reach it), by distance across
        y = cap['centre'][0]
        dist = sorted((abs(o['centre'][1] - cap['centre'][1]), k) for k, o in enumerate(objects)
                      if o['top'] - 20 <= y <= o['top'] + o['mask'].shape[0] + 20)
        if not dist or (len(dist) > 1 and dist[1][0] < NEAREST * dist[0][0]):
            out.append({'kind': 'skipped', 'source': src, 'why': 'no object clearly nearest'})
            continue
        out.append({'kind': 'order', 'symbols': [objects[dist[0][1]]['sample']], 'names': [cap['name']],
                    'source': src})
    return out or [{'kind': 'skipped', 'source': f'K5 {name}', 'why': 'no captions read'}]


def caption_matching_lesson(path: str, memory: SymbolMemory, name: str) -> List[Dict]:
    p = page(path)[HEADER:]
    objects = objects_in(p)
    names = [c['name'] for c in captions(p, memory)]
    if not objects or len(objects) != len(names):
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': f'{len(objects)} objects, {len(names)} captions'}]
    return [{'kind': 'matching', 'numerals': [o['sample'] for o in objects], 'counts': names, 'domain': 'colour',
             'source': f'K5 {name}'}]


def words_and_objects(path: str, memory: SymbolMemory, words_left: bool = True, dotted: bool = False):
    """
    A sheet of colour words in one half and objects in the other: (names read, objects). dotted: the words are
    dotted, to trace, on ruled lines: the lines are dropped and the dots joined before reading
    """
    from scipy import ndimage
    p = page(path)[HEADER:]
    half = p.shape[1] // 2
    words_side = p[:, :half] if words_left else p[:, half:]
    objects_side = p[:, half:] if words_left else p[:, :half]
    if dotted:
        rgb = words_side[..., :3].astype(int)
        traced = rgb.mean(axis=2) < 235
        traced[traced.mean(axis=1) > 0.3] = False             # the ruled lines
        dark = ndimage.binary_closing(traced, structure=np.ones((5, 5)))
        objects_side = objects_side.copy()
    else:
        dark = _dark(words_side)
    names = []
    for r0, r1 in _bands(dark):
        for w in words_in(dark[r0:r1]):
            word = memory.read_word(w, COLOURS)
            if word:
                names.append(word)
    return names, objects_in(objects_side)


def matching_lesson(path: str, memory: SymbolMemory, name: str, words_left: bool = True,
                    dotted: bool = False) -> List[Dict]:
    names, objects = words_and_objects(path, memory, words_left, dotted)
    if not names or len(names) != len(objects):
        return [{'kind': 'skipped', 'source': f'K5 {name}', 'why': f'{len(objects)} objects, {len(names)} names'}]
    return [{'kind': 'matching', 'numerals': [o['sample'] for o in objects], 'counts': names, 'domain': 'colour',
             'source': f'K5 {name}'}]


def lessons(cache: str, memory: SymbolMemory) -> List[Dict]:
    out = []
    for s in 'abce':
        out += caption_lessons(fetch(K5.format(f'match-write-colors-{s}'), cache), memory, f'match-write-colors-{s}')
    out += caption_matching_lesson(fetch(K5.format('match-write-colors-d'), cache), memory, 'match-write-colors-d')
    for k in (1, 2, 3):
        out += matching_lesson(fetch(K5.format(f'colors-{k}'), cache), memory, f'colors-{k}')
    out += matching_lesson(fetch(K5.format('match-colors-to-names-1'), cache), memory, 'match-colors-to-names-1',
                           words_left=False)

    return out


def teach(cache: str, path: str = LEARNED, verbose: bool = True) -> SymbolMemory:
    memory = SymbolMemory(path if os.path.exists(path) else None)
    memory.examples = [e for e in memory.examples if e['meaning'] not in COLOURS]          # colours anew
    plan = lessons(cache, memory)
    taught = [l for l in plan if l['kind'] != 'skipped']
    outcomes = memory.teach(taught)
    if verbose:
        for l in plan:
            if l['kind'] == 'skipped':
                print(f'{l["source"]:40s} skipped: {l["why"]}')
        for l, r in zip(taught, outcomes):
            print(f'{l["source"]:40s} {l["kind"]:9s} {r["outcome"]} {r.get("learned", r.get("why", ""))}')
        print('known:', {k: v for k, v in memory.meanings().items() if k in COLOURS})
    memory.path = path
    memory.save(path)
    return memory


if __name__ == '__main__':
    teach(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_worksheets'))
