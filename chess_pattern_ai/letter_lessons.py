#!/usr/bin/env python3
"""
Letter Lessons - teaching the program letters from children's alphabet worksheets

As digit_lessons does for digits: the lessons are read from rendered
worksheet pages; where each exercise is on a page is given per sheet type
here, and what is in it is found by the program (symbol_learning).

What is known beforehand, as counting was for digits: the alphabet as an
order of 26 names, A to Z, and a to z for the lower case, and which lower
case name goes with which capital.

Sheets, in teaching order (all K5 Learning, kindergarten):
- "Printing the letter A" ... "Z" (printing-letters-a ... z): each sheet
  shows its letter in green, as a capital and in lower case. The 26 sheets
  are an order lesson: the n-th sheet's letters are the n-th letter. Type
  style: a round sans serif (single-storey a and g).
- "Uppercase and lowercase letters" (uppercase-lowercase-letters-c): a column
  of capitals and a column of their lower case letters, shuffled; case
  matching lessons. Type style: a grey sans serif.
- "Matching upper and lower case letters" (uppercase-lowercase-letters-a):
  a capital and three lower case letters per row, one of them its lower
  case; case choice lessons. Type style: Arial-like (double-storey a and g).
On both case sheets the capitals run through the alphabet in order (page
after page, row after row): an anchored order lesson each, which places
them by the capitals already recognised. A case lesson waits until its
capitals are recognised with confidence; it then rules out lower case
letters recognised as other letters.

Results (as committed): the printing sheets teach A-Z and a-z in the round
style (one sheet's letters each); the case choice sheet's capitals are
placed in order by 13 recognised anchors (D E J K L N P Q R S learned), and
its rows then teach or confirm the lower case letters of that style except
b and f (each ruled out as h and t: recognised as those with confidence,
the lesson ends in a contradiction and teaches nothing); the case matching
sheet adds B C F. Every stored example was checked by eye (139 with the
digits): none is mislabelled. Letters read on the K5 order of letters
sheets: see children_worksheets.

Run: python letter_lessons.py [cache directory] [--no-a] [--no-c]  (adds the letters to the memory saved by
digit_lessons, LEARNED, or to a new one; prints each lesson's outcome)
"""

import os
import sys
from typing import Dict, List

import numpy as np

from digit_lessons import LEARNED, fetch, page
from symbol_learning import UPPER, LOWER, SymbolMemory, colourful, components, ink, is_frame, symbols_of

K5 = 'https://www.k5learning.com/worksheets/kindergarten/{}.pdf'
SCALE = 2.0
HEADER = 200            # pages at SCALE 2: the title and logo lie above this row


def clean(region: np.ndarray, dark_only: bool = False) -> np.ndarray:
    """region's ink without frames (boxes drawn around letters); dark_only: without coloured ink either"""
    mask = ink(region)
    if dark_only:
        colour = colourful(region)
        if colour is not None:
            from scipy import ndimage
            mask &= ~ndimage.binary_dilation(colour, iterations=3)   # with the coloured lines' blurred edges
    for rs, cs, m in components(mask):
        if is_frame(m):
            mask[rs, cs] &= ~m
    return mask


def _symbols_in_bands(mask: np.ndarray, min_gap: int) -> List[List[np.ndarray]]:
    """The symbols of each band of mask between runs of at least min_gap empty rows, top to bottom"""
    return [symbols_of(mask[r0:r1]) for r0, r1 in _row_runs(mask, min_gap)]


def _row_runs(mask: np.ndarray, min_gap: int):
    rows = mask.any(axis=1)
    out, start, empty = [], None, 0
    for y, r in enumerate(list(rows) + [False] * min_gap):
        if r:
            if start is None:
                start = y
            empty = 0
        elif start is not None:
            empty += 1
            if empty >= min_gap:
                out.append((start, y - empty + 1))
                start, empty = None, 0
    return out


def printing_sheet(path: str):
    """The green capital and lower case letter of a printing-letters sheet (None if not found)"""
    p = page(path, SCALE)
    mask = ink(p) & colourful(p)
    mask[:HEADER] = False
    found = [s for s in _letters(_symbols_in_bands(mask, 30)) if s]
    if len(found) < 2 or len(found[0]) != 1 or len(found[1]) != 1:
        return None
    return found[0][0], found[1][0]


def _letters(rows: List[List[np.ndarray]], least: int = 30) -> List[List[np.ndarray]]:
    """Of each row, the symbols shaped like letters: at least least cells tall (pieces of rules, specks and
    the fringes of coloured circles are not), and not much wider than tall (rules)"""
    return [[m for m in row if m.shape[0] >= least and m.shape[1] <= 2.5 * m.shape[0]] for row in rows]


def lessons(cache: str, use_a: bool = True, use_c: bool = True) -> List[Dict]:
    out: List[Dict] = []
    capitals, small = [], []
    for c in LOWER:
        found = printing_sheet(fetch(K5.format(f'printing-letters-{c}'), cache))
        if found is None:
            out.append({'kind': 'skipped', 'source': f'printing-letters-{c}', 'why': 'letters not found'})
            capitals.append(None)
            small.append(None)
            continue
        capitals.append(found[0])
        small.append(found[1])
    for names, symbols in ((UPPER, capitals), (LOWER, small)):
        keep = [(n, s) for n, s in zip(names, symbols) if s is not None]
        out.append({'kind': 'order', 'symbols': [s for _, s in keep], 'names': [n for n, _ in keep],
                    'source': 'K5 printing-letters a-z'})
    if use_c:
        import pypdfium2 as pdfium
        path = fetch(K5.format('uppercase-lowercase-letters-c'), cache)
        for i in range(len(pdfium.PdfDocument(path))):
            p = page_n(path, i)
            mask = clean(p, dark_only=True)
            mask[:HEADER] = False
            half = mask.shape[1] // 2
            left = [s for row in _letters(_symbols_in_bands(mask[:, :half], 20)) for s in row]
            right = [s for row in _letters(_symbols_in_bands(mask[:, half:], 20)) for s in row]
            if len(left) == len(right) and left:
                out.append({'kind': 'anchored order', 'symbols': left, 'names': UPPER, 'domain': 'upper',
                            'source': f'K5 uppercase-lowercase-letters-c p{i + 1} capitals'})
                out.append({'kind': 'case matching', 'capitals': left, 'symbols': right,
                            'source': f'K5 uppercase-lowercase-letters-c p{i + 1}'})
            else:
                out.append({'kind': 'skipped', 'source': f'K5 uppercase-lowercase-letters-c p{i + 1}',
                            'why': f'{len(left)} capitals, {len(right)} lower case'})
    if use_a:
        import pypdfium2 as pdfium
        path = fetch(K5.format('uppercase-lowercase-letters-a'), cache)
        rows_found = []
        for i in range(len(pdfium.PdfDocument(path))):
            p = page_n(path, i)
            mask = clean(p)
            mask[:HEADER + 100] = False                 # also the instruction line
            for j, row in enumerate([r for r in _letters(_symbols_in_bands(mask, 20)) if r], 1):
                src = f'K5 uppercase-lowercase-letters-a p{i + 1} row {j}'
                if len(row) != 4:
                    out.append({'kind': 'skipped', 'source': src, 'why': f'{len(row)} letters, not 4'})
                    continue
                out.append({'kind': 'case choice', 'capital': row[0], 'symbols': row[1:], 'source': src})
                rows_found.append(row[0])
        # the rows' capitals run through the alphabet in order, page after page
        out.insert(len(out) - len(rows_found), {'kind': 'anchored order', 'symbols': rows_found, 'names': UPPER,
                                                'domain': 'upper',
                                                'source': 'K5 uppercase-lowercase-letters-a capitals'})
    return out


def page_n(path: str, n: int) -> np.ndarray:
    import pypdfium2 as pdfium
    return np.array(pdfium.PdfDocument(path)[n].render(scale=SCALE).to_pil().convert('RGB'))


def teach(cache: str, path: str = LEARNED, verbose: bool = True, **kwargs) -> SymbolMemory:
    memory = SymbolMemory(path if os.path.exists(path) else None)
    memory.examples = [e for e in memory.examples if not isinstance(e['meaning'], str)]   # letters anew
    plan = lessons(cache, **kwargs)
    taught = [l for l in plan if l['kind'] != 'skipped']
    outcomes = memory.teach(taught)
    if verbose:
        for l in plan:
            if l['kind'] == 'skipped':
                print(f'{l["source"]:48s} skipped: {l["why"]}')
        for l, r in zip(taught, outcomes):
            learned = r.get('learned', r.get('why', ''))
            print(f'{l["source"]:48s} {l["kind"]:13s} {r["outcome"]} {"".join(map(str, learned)) if isinstance(learned, list) else learned}')
        print('known:', {k: v for k, v in memory.meanings().items() if isinstance(k, str)})
    memory.path = path
    memory.save(path)
    return memory


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    teach(args[0] if args else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_worksheets'),
          use_a='--no-a' not in sys.argv, use_c='--no-c' not in sys.argv)
