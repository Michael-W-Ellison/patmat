#!/usr/bin/env python3
"""
Symbol Learning - learning what written symbols mean, from examples

A SymbolMemory holds examples of symbols (ink images) with their meanings
(any JSON value: 7, 'A', ...). It grows by being taught and keeps what it
has learned in a JSON file, so knowledge accumulates across runs, and new
kinds of symbols (letters, signs) are added the same way.

Lessons (each returns what it learned, or why it learned nothing):
- learn(image, meaning): one labelled example (a teacher pointing at a
  symbol and naming it).
- counting lesson (learn_by_counting): numeral symbols beside a group of
  pictures, as on a number tracing sheet: the pictures are counted and the
  numeral means the count. A numeral of several symbols (10) teaches its one
  unknown symbol when the others are known, by decimal place value (the one
  rule built in rather than learned).
- choice lesson (learn_by_choice): a group of pictures and several numerals,
  one of which is the count ("count and circle the number"). Numerals
  recognised with confidence as some other number are ruled out; if one
  numeral is left, it is learned as the count (and a numeral recognised as
  the count is learned again, as one more example of its style).
- matching lesson (learn_by_matching): several groups and as many numerals,
  to be paired ("count and match"). Confidently recognised numerals are
  paired with their counts; when one numeral and one count are left, the
  numeral is learned as that count.
Lessons that cannot yet be decided are kept (teach) and tried again once
other lessons have taught more, so knowledge spreads from symbols known in
one style to the same symbols in other styles.

Recognition: an ink image is cropped to its ink, scaled into a SIZE x SIZE
square keeping its proportions, and blurred; the nearest stored example
(mean absolute difference) gives the meaning when nearer than REJECT, else
the symbol is unknown (None). Lessons rule a numeral out only when its
nearest example is nearer than CONFIDENT.

Reading: read(panel) splits a panel into symbols (connected ink, left to
right) and recognises each; read_number() makes a number of digits by
place value.

Blobs (blobs): ink is dark (luminance below DARK) or coloured (saturation
above COLOURED); connected ink is grouped into blobs (parts whose boxes
touch after widening by GAP), coloured parts with coloured, dark parts with
dark; a dark part inside a coloured blob's box joins it (a picture's
details). A part that is a rectangular frame (a box drawn around a numeral
or a card) is left out. A blob is coloured when most of its ink is: on
these worksheets pictures are coloured and numerals are black or grey.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SIZE = 16               # recognition: symbols compared at SIZE x SIZE
REJECT = 0.22           # recognition: nearest example further than this -> unknown
CONFIDENT = 0.10        # lessons: a numeral is ruled out only when recognised nearer than this
GAP = 3                 # blobs: parts of one blob may be this many cells apart
DARK = 160              # ink: luminance below this (0..255)
COLOURED = 60           # ink: or saturation (max - min channel) above this
FRAME = 0.6             # a part whose box border is this much ink, and is nearly empty inside, is a frame
SMALL = 0.25            # pictures with less ink than this share of the largest are noise
SHORT = 0.6             # numerals less tall than this share of the tallest are not numerals ("OR")


# ------------------------------------------------------------------ images

def ink(image) -> np.ndarray:
    """A boolean ink mask: from an RGB or grey array 0..255, or a 0/1 or boolean array"""
    a = np.asarray(image)
    if a.dtype == bool:
        return a
    if a.ndim == 3:
        rgb = a[..., :3].astype(int)
        return (rgb.mean(axis=2) < DARK) | (rgb.max(axis=2) - rgb.min(axis=2) > COLOURED)
    if a.max() > 1:
        return a < DARK
    return a > 0


def colourful(image) -> Optional[np.ndarray]:
    """Per cell: whether it is coloured (None for an image without colour channels)"""
    a = np.asarray(image)
    if a.ndim != 3:
        return None
    rgb = a[..., :3].astype(int)
    return rgb.max(axis=2) - rgb.min(axis=2) > COLOURED


def components(mask: np.ndarray) -> List[Tuple[slice, slice, np.ndarray]]:
    """8-connected components: (row slice, column slice, the component's mask within that box)"""
    from scipy import ndimage
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), int))
    return [(box[0], box[1], labels[box] == i) for i, box in enumerate(ndimage.find_objects(labels), 1)]


def is_frame(m: np.ndarray) -> bool:
    """A rectangular outline (possibly with rounded corners or a shadow): most of its box border is ink and
    the inside of the box (a margin of 15% in from each side) is nearly empty"""
    if min(m.shape) < 12:
        return False
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    dy, dx = max(2, m.shape[0] * 15 // 100), max(2, m.shape[1] * 15 // 100)
    return border.mean() >= FRAME and m[dy:-dy, dx:-dx].mean() < 0.05


def _union_find(n):
    group = list(range(n))

    def find(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    def union(i, j):
        group[find(i)] = find(j)
    return find, union


def blobs(image) -> List[Dict]:
    """Blobs of ink (see the module docstring): {'mask', 'top', 'left', 'parts', 'coloured'}"""
    mask = ink(image)
    colour = colourful(image)
    parts = [p for p in components(mask) if not is_frame(p[2])]
    tone = [bool(colour is not None and colour[rs, cs][m].mean() > 0.5) for rs, cs, m in parts]
    box = [(p[0].start, p[1].start, p[0].stop, p[1].stop) for p in parts]
    find, union = _union_find(len(parts))

    def near(a, b):
        return a[0] - GAP <= b[2] and b[0] - GAP <= a[2] and a[1] - GAP <= b[3] and b[1] - GAP <= a[3]

    # parts of one kind (coloured, or dark) that lie close together form one blob
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if tone[i] == tone[j] and near(box[i], box[j]):
                union(i, j)
    # a dark part inside a coloured blob's box is a detail of that picture (an eye, an outline, dots)
    groups: Dict[int, List[int]] = {}
    for i in range(len(parts)):
        groups.setdefault(find(i), []).append(i)
    extent = {g: (min(box[i][0] for i in ids), min(box[i][1] for i in ids), max(box[i][2] for i in ids),
                  max(box[i][3] for i in ids)) for g, ids in groups.items()}
    for g, ids in list(groups.items()):
        if tone[ids[0]]:
            continue
        e = extent[g]
        for h, f in extent.items():
            if h != g and tone[groups[h][0]] and f[0] <= e[0] and f[1] <= e[1] and e[2] <= f[2] and e[3] <= f[3]:
                union(ids[0], groups[h][0])
                break
    members: Dict[int, List[int]] = {}
    for i in range(len(parts)):
        members.setdefault(find(i), []).append(i)
    out = []
    for ids in members.values():
        r0 = min(box[i][0] for i in ids)
        c0 = min(box[i][1] for i in ids)
        r1 = max(box[i][2] for i in ids)
        c1 = max(box[i][3] for i in ids)
        m = np.zeros((r1 - r0, c1 - c0), bool)
        for i in ids:
            rs, cs, pm = parts[i]
            m[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] |= pm
        coloured = bool(colour is not None and colour[r0:r1, c0:c1][m].mean() > 0.5)
        out.append({'mask': m, 'top': r0, 'left': c0, 'parts': len(ids), 'coloured': coloured})
    return out


def pictures_and_numerals(image) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    A worksheet scene's pictures (coloured blobs, small ones dropped as noise) and numeral symbols (the other
    blobs, tall ones only, left to right)
    """
    found = blobs(image)
    pics = [b for b in found if b['coloured']]
    if pics:
        largest = max(b['mask'].sum() for b in pics)
        pics = [b for b in pics if b['mask'].sum() >= SMALL * largest]
    nums = [b for b in found if not b['coloured']]
    if nums:
        tallest = max(b['mask'].shape[0] for b in nums)
        nums = [b for b in nums if b['mask'].shape[0] >= SHORT * tallest]
    return [b['mask'] for b in pics], [b['mask'] for b in sorted(nums, key=lambda b: (b['left'], b['top']))]


ALIKE = 0.08            # counting scenes: blobs this alike (normalised, mean difference) are copies of one picture
SIZE_RATIO = 1.5        # counting scenes: ... with ink within this ratio
NOISE = 0.3             # counting scenes: blobs less tall than this share of the tallest are noise


def counting_scene(image) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    A scene of a numeral and copies of one picture in black ink (a number tracing sheet): the pictures are the
    largest group of blobs alike in look and size; with no two alike, the single picture is the blob of
    several parts (an outline with details); the other blobs are the numeral's symbols, left to right
    """
    found = blobs(image)
    tallest = max((b['mask'].shape[0] for b in found), default=0)
    found = [b for b in found if b['mask'].shape[0] >= NOISE * tallest]
    looks = [normalise(b['mask']) for b in found]
    best: List[int] = []
    for i in range(len(found)):
        alike = [j for j in range(len(found))
                 if np.abs(looks[i] - looks[j]).mean() <= ALIKE
                 and 1 / SIZE_RATIO <= found[i]['mask'].sum() / found[j]['mask'].sum() <= SIZE_RATIO]
        if len(alike) > len(best):
            best = alike
    if len(best) < 2:
        best = [i for i, b in enumerate(found) if b['parts'] > 1][:1]
    numerals = sorted((b for i, b in enumerate(found) if i not in best), key=lambda b: b['left'])
    return [found[i]['mask'] for i in best], [b['mask'] for b in numerals]


def normalise(mask: np.ndarray) -> np.ndarray:
    """A symbol's ink cropped, centred in a square, scaled to SIZE x SIZE (fraction of ink per cell), blurred"""
    from scipy import ndimage
    rows, cols = np.nonzero(mask)
    m = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(float)
    side = max(m.shape)
    square = np.zeros((side, side))
    r0, c0 = (side - m.shape[0]) // 2, (side - m.shape[1]) // 2
    square[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m
    edges = np.linspace(0, side, SIZE + 1)
    out = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            r_a, r_b = int(edges[i]), max(int(edges[i]) + 1, int(np.ceil(edges[i + 1])))
            c_a, c_b = int(edges[j]), max(int(edges[j]) + 1, int(np.ceil(edges[j + 1])))
            out[i, j] = square[r_a:r_b, c_a:c_b].mean()
    return ndimage.gaussian_filter(out, 0.8)


def pack(mask) -> Dict:
    """A 0/1 mask as {'width', 'rows': one hex string per row} (for JSON files)"""
    m = np.asarray(mask).astype(bool)
    padded = np.pad(m, ((0, 0), (0, (-m.shape[1]) % 4)))
    rows = [''.join('%x' % int(''.join('1' if b else '0' for b in r[i:i + 4]), 2) for i in range(0, len(r), 4))
            for r in padded]
    return {'width': int(m.shape[1]), 'rows': rows}


def unpack(packed: Dict) -> np.ndarray:
    w = packed['width']
    return np.array([[int(b) for ch in r for b in format(int(ch, 16), '04b')][:w] for r in packed['rows']],
                    dtype=int).reshape(len(packed['rows']), w)


# ------------------------------------------------------------------ the memory

class SymbolMemory:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.examples: List[Dict] = []          # {'meaning', 'image' (normalised), 'ink' (the mask), 'source'}
        if path and os.path.exists(path):
            with open(path) as f:
                self.examples = json.load(f)['examples']
        self._cache = None

    def save(self, path: Optional[str] = None) -> None:
        with open(path or self.path, 'w') as f:
            json.dump({'about': 'symbol examples learned by symbol_learning.SymbolMemory', 'size': SIZE,
                       'examples': self.examples}, f)

    def meanings(self) -> Dict:
        out: Dict = {}
        for e in self.examples:
            out[e['meaning']] = out.get(e['meaning'], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    # -------------------------------------------------------------- learning

    def learn(self, image, meaning, source: str = '') -> None:
        mask = ink(image)
        self.examples.append({'meaning': meaning, 'image': normalise(mask).round(3).tolist(),
                              'ink': pack(mask), 'source': source})
        self._cache = None

    def learn_by_counting(self, numerals: Sequence[np.ndarray], count: int, source: str = '') -> Dict:
        """numerals (left to right) mean count; see the module docstring"""
        digits = str(count)
        if count < 0 or len(digits) != len(numerals):
            return {'outcome': 'not a lesson', 'why': f'{len(numerals)} numeral symbol(s) for {count}'}
        if len(numerals) == 1:
            self.learn(numerals[0], count, source)
            return {'outcome': 'learned', 'learned': [count]}
        known = [self.confident(s) for s in numerals]
        if any(k is not None and str(k) != d for k, d in zip(known, digits)):
            return {'outcome': 'contradiction', 'why': f'read {known} for {count}'}
        unknown = [i for i, k in enumerate(known) if k is None]
        if len(unknown) != 1:
            return {'outcome': 'pending', 'why': f'{len(unknown)} unknown symbols'}
        i = unknown[0]
        self.learn(numerals[i], int(digits[i]), source)
        return {'outcome': 'learned', 'learned': [int(digits[i])]}

    def learn_by_choice(self, numerals: Sequence[np.ndarray], count: int, source: str = '') -> Dict:
        """One of numerals means count; see the module docstring"""
        known = [self.confident(s) for s in numerals]
        possible = [i for i, k in enumerate(known) if k is None or k == count]
        if not possible:
            return {'outcome': 'contradiction', 'why': f'read {known}, count {count}'}
        if len(possible) > 1:
            return {'outcome': 'pending', 'why': f'{len(possible)} numerals possible'}
        self.learn(numerals[possible[0]], count, source)
        return {'outcome': 'learned' if known[possible[0]] is None else 'confirmed', 'learned': [count]}

    def learn_by_matching(self, numerals: Sequence[np.ndarray], counts: Sequence[int], source: str = '') -> Dict:
        """numerals and counts pair up one to one; see the module docstring"""
        if len(numerals) != len(counts):
            return {'outcome': 'not a lesson', 'why': f'{len(numerals)} numerals, {len(counts)} counts'}
        left = list(counts)
        unknown = []
        for s in numerals:
            k = self.confident(s)
            if k is not None and k in left:
                left.remove(k)
            elif k is not None:
                return {'outcome': 'contradiction', 'why': f'read {k}, counts {list(counts)}'}
            else:
                unknown.append(s)
        if len(unknown) != 1:
            return {'outcome': 'pending' if unknown else 'confirmed', 'why': f'{len(unknown)} unknown'}
        self.learn(unknown[0], left[0], source)
        return {'outcome': 'learned', 'learned': [left[0]]}

    def teach(self, lessons: Sequence[Dict], passes: int = 4) -> List[Dict]:
        """
        Run lessons ({'kind': 'counting' | 'choice' | 'matching', 'numerals', 'count' or 'counts', 'source'})
        until none teaches more (at most passes times over those still pending); returns each lesson's
        final outcome
        """
        outcomes: List[Optional[Dict]] = [None] * len(lessons)
        todo = list(range(len(lessons)))
        for _ in range(passes):
            progress = False
            still = []
            for i in todo:
                l = lessons[i]
                if l['kind'] == 'counting':
                    r = self.learn_by_counting(l['numerals'], l['count'], l.get('source', ''))
                elif l['kind'] == 'choice':
                    r = self.learn_by_choice(l['numerals'], l['count'], l.get('source', ''))
                else:
                    r = self.learn_by_matching(l['numerals'], l['counts'], l.get('source', ''))
                outcomes[i] = r
                if r['outcome'] in ('learned', 'confirmed'):
                    progress = progress or r['outcome'] == 'learned'
                elif r['outcome'] == 'pending':
                    still.append(i)
            todo = still
            if not progress or not todo:
                break
        return outcomes

    # -------------------------------------------------------------- recognition and reading

    def _matrix(self):
        if self._cache is None:
            self._cache = (np.array([e['image'] for e in self.examples]).reshape(len(self.examples), -1)
                           if self.examples else np.zeros((0, SIZE * SIZE)))
        return self._cache

    def nearest(self, image) -> Tuple[Optional[object], float]:
        """(meaning of the nearest example, its distance); (None, inf) with no examples"""
        if not self.examples:
            return None, float('inf')
        d = np.abs(self._matrix() - normalise(ink(image)).reshape(1, -1)).mean(axis=1)
        i = int(d.argmin())
        return self.examples[i]['meaning'], float(d[i])

    def recognise(self, image):
        meaning, distance = self.nearest(image)
        return meaning if distance <= REJECT else None

    def confident(self, image):
        meaning, distance = self.nearest(image)
        return meaning if distance <= CONFIDENT else None

    def read(self, panel) -> Optional[List]:
        """The meanings of a panel's symbols, left to right; None if the panel is empty or a symbol is unknown"""
        mask = ink(panel)
        if not mask.any():
            return None
        out = []
        for _, _, m in sorted(components(mask), key=lambda c: c[1].start):
            meaning = self.recognise(m)
            if meaning is None:
                return None
            out.append(meaning)
        return out

    def read_number(self, panel) -> Optional[int]:
        symbols = self.read(panel)
        if not symbols or not all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s <= 9 for s in symbols):
            return None
        return int(''.join(str(s) for s in symbols))

    def draw_number(self, value: int) -> Optional[np.ndarray]:
        """A panel showing value, written with the first stored example of each digit (bottoms aligned)"""
        glyphs = []
        for ch in str(value):
            example = next((e for e in self.examples if e['meaning'] == int(ch)), None)
            if example is None:
                return None
            glyphs.append(unpack(example['ink']))
        h = max(g.shape[0] for g in glyphs)
        parts = []
        for g in glyphs:
            parts += [np.pad(g, ((h - g.shape[0], 0), (0, 0))), np.zeros((h, 2), int)]
        return np.hstack(parts[:-1])
