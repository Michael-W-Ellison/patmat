#!/usr/bin/env python3
"""
Digit Lessons - teaching the program digits from children's counting worksheets

The lessons come from free kindergarten counting worksheets, read from their
rendered pages (no hand transcription): the program finds the pictures and
the numerals in each scene, counts the pictures, and learns what it can
(symbol_learning.SymbolMemory.teach). Where each scene is on a page (a band
between empty rows, or a framed card) is given per sheet type here, as a
teacher points at the next exercise; what is in the scene is the program's.

Sheets, in teaching order:
- worksheetfun number tracing 1-10: one numeral and that many line-drawn
  pictures (counting lessons; 10 teaches 0 by place value once 1 is known).
- K5 Learning "count the objects" (count-match a, b; count-match-10 a, b, d):
  a group of coloured pictures and 2 or 5 numerals, one of them the count
  (choice lessons), in three more type styles.
- K5 Learning "count and match" (count-match c; count-match-10 e): five
  groups and five numerals on a sheet, to be paired (matching lessons).

Results (as committed): the tracing sheets teach 0-9 in one type style
(0 by place value from the 10 sheet); the K5 sheets confirm 2nd styles of
2-6 and 8 and teach one 4 by elimination; the handwritten-style cards stay
undecided (nothing in them is recognised strictly enough to rule it out),
and scenes whose counts may be wrong (one picture, uneven pictures) are
skipped. Read with the learned examples, the 35 numbers of the
worksheetfun number-sequence sheet (a type style not among the lessons) are
all right; with the tracing sheets' examples alone, 26/35.

A scene is not taught from when its count is unsure (pictures of uneven
size: two touching pictures read as one, or one picture in two parts; a
single picture blob, which may be touching copies) or
it has an unexpected number of numerals.

Run: python digit_lessons.py [cache directory]  (downloads the sheets once, teaches a new memory, saves it to
LEARNED, prints each lesson's outcome)
"""

import os
import sys
import urllib.request
from typing import Dict, List

import numpy as np

from symbol_learning import SymbolMemory, blobs, counting_scene, ink, pictures_and_numerals

HERE = os.path.dirname(os.path.abspath(__file__))
LEARNED = os.path.join(HERE, 'learned_symbols.json')
WSF = 'https://www.worksheetfun.com/letter%20and%20numbers/fun-numbertracing-{}.pdf'
K5 = 'https://www.k5learning.com/worksheets/kindergarten/{}.pdf'
TOP, BOTTOM = 140, 750          # K5 pages at scale 1: the exercises lie between the instructions and the footer
MIN_GAP = 10                    # bands are separated by at least this many empty rows
UNEVEN = 2.0                    # a scene whose pictures differ in ink by more than this factor is not counted


def fetch(url: str, cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, url.rsplit('/', 1)[1].replace('%20', '_'))
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


def page(path: str, scale: float = 1.0) -> np.ndarray:
    import pypdfium2 as pdfium
    return np.array(pdfium.PdfDocument(path)[0].render(scale=scale).to_pil().convert('RGB'))


def bands(region: np.ndarray) -> List[np.ndarray]:
    """The parts of region between runs of at least MIN_GAP rows without ink"""
    rows = ink(region).any(axis=1)
    out, start, empty = [], None, 0
    for y, r in enumerate(list(rows) + [False] * MIN_GAP):
        if r:
            if start is None:
                start = y
            empty = 0
        elif start is not None:
            empty += 1
            if empty >= MIN_GAP:
                out.append(region[start:y - empty + 1])
                start, empty = None, 0
    return [b for b in out if b.shape[0] >= 20]


def cards(region: np.ndarray) -> List[np.ndarray]:
    """The insides of the large frames (cards) in region, top to bottom, left to right"""
    from symbol_learning import components
    found = []
    for rs, cs, m in components(ink(region)):
        if rs.stop - rs.start > 100 and cs.stop - cs.start > 150:
            border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
            if border.mean() > 0.5 and m[8:-8, 8:-8].mean() < 0.05:
                found.append((rs.start, cs.start, region[rs.start + 8:rs.stop - 8, cs.start + 8:cs.stop - 8]))
    return [c for _, _, c in sorted(found, key=lambda e: (e[0] // 50, e[1]))]


def _doubt(pictures, numerals, numerals_wanted) -> str:
    """Why a scene's count or numerals are not to be trusted ('' if they are): copies of one picture are of
    about one size, so a blob much bigger than another is likely two touching pictures (or a part of one)"""
    if len(numerals) != numerals_wanted:
        return f'{len(numerals)} numerals, not {numerals_wanted}'
    if not pictures:
        return 'no pictures'
    if len(pictures) == 1:
        return 'one picture: one, or touching copies of it, cannot be told apart'
    sizes = [int(p.sum()) for p in pictures]
    if max(sizes) > UNEVEN * min(sizes):
        return f'pictures of uneven size ({min(sizes)} to {max(sizes)} cells of ink): count unsure'
    if numerals_wanted > 1 and len(pictures) > 9:
        return f'count {len(pictures)} but the numerals are single symbols'
    return ''


def lessons(cache: str) -> List[Dict]:
    out: List[Dict] = []
    for n in range(1, 11):
        p = page(fetch(WSF.format(n), cache))
        pictures, numerals = counting_scene(p[55:385, 55:560])
        out.append({'kind': 'counting', 'numerals': numerals, 'count': len(pictures),
                    'source': f'worksheetfun number tracing {n}'})
    for sheet, layout, numerals_wanted in [('count-match-a', 'rows', 2), ('count-match-b', 'rows', 2),
                                           ('count-match-10-b', 'rows', 2), ('count-match-10-a', 'cards', 2),
                                           ('count-match-10-d', 'cards', 5)]:
        p = page(fetch(K5.format(sheet), cache))[TOP:BOTTOM]
        for i, scene in enumerate(bands(p) if layout == 'rows' else cards(p), 1):
            pictures, numerals = pictures_and_numerals(scene)
            why = _doubt(pictures, numerals, numerals_wanted)
            if why:
                out.append({'kind': 'skipped', 'source': f'K5 {sheet} #{i}', 'why': why})
                continue
            out.append({'kind': 'choice', 'numerals': numerals, 'count': len(pictures),
                        'source': f'K5 {sheet} #{i}'})
    for sheet in ('count-match-c', 'count-match-10-e'):
        p = page(fetch(K5.format(sheet), cache))[TOP:BOTTOM]
        numerals, counts = [], []
        doubts = []
        for scene in bands(p):
            pictures, nums = pictures_and_numerals(scene)
            why = _doubt(pictures, nums, 1)
            if why:
                doubts.append(why)
            numerals += nums[:1]
            counts.append(len(pictures))
        if doubts:
            out.append({'kind': 'skipped', 'source': f'K5 {sheet}', 'why': '; '.join(doubts)})
        else:
            out.append({'kind': 'matching', 'numerals': numerals, 'counts': counts, 'source': f'K5 {sheet}'})
    return out


def teach(cache: str, path: str = LEARNED, verbose: bool = True) -> SymbolMemory:
    memory = SymbolMemory()
    plan = lessons(cache)
    taught = [l for l in plan if l['kind'] != 'skipped']
    outcomes = memory.teach(taught)
    if verbose:
        for l in plan:
            if l['kind'] == 'skipped':
                print(f'{l["source"]:40s} skipped: {l["why"]}')
        for l, r in zip(taught, outcomes):
            what = l.get('count', l.get('counts'))
            print(f'{l["source"]:40s} {l["kind"]:9s} count {what}: {r["outcome"]} {r.get("learned", r.get("why", ""))}')
        print('known:', memory.meanings())
    memory.save(path)
    return memory


if __name__ == '__main__':
    teach(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_worksheets'))
