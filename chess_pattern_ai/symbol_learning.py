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
- order lesson (learn_in_order): symbols shown in a known order (a series
  of sheets one per letter, A to Z) mean the names in that order.
- anchored order lesson (learn_by_anchored_order): symbols consecutive in a
  known order from an unknown start (capitals down the rows of a sheet set,
  A to Z): the start is fixed by symbols already recognised (at least two
  agreeing, none disagreeing), and the others are learned by their place.
- case lessons: lower case letters to be matched with capitals (one of
  several, or a column of them, shuffled): once the capitals are recognised,
  as choice and matching lessons with the capitals' lower case names.
- across lessons (learn_across): two matching lessons left undecided, each
  with symbols unknown and names left over: a symbol of one that looks the
  same as one of the other (a black tyre, a black swatch) is the same
  thing, so it is the one name both have left, when there is exactly one.
Lessons that cannot yet be decided are kept (teach) and tried again once
other lessons have taught more, so knowledge spreads from symbols known in
one style to the same symbols in other styles.

Recognition: an ink image is cropped to its ink, scaled into a SIZE x SIZE
square keeping its proportions, and blurred; the nearest stored example
(mean absolute difference) gives the meaning when nearer than REJECT, else
the symbol is unknown (None). Lessons rely only on recognitions made with
confidence: the nearest example nearer than CONFIDENT, or nearer by MARGIN
than any example of another meaning (MARGIN was chosen on the capitals of
the case choice sheets, letter_lessons: it keeps 13 of 26, all right).
Ruling a symbol out uses CONFIDENT alone (strict): with the margin, two
handwritten digits were ruled out wrongly and the other numeral taught
under their count.

Reading: read(panel) splits a panel into symbols (connected ink, left to
right) and recognises each; read_number() makes a number of digits by
place value. readings(panel) gives the panel's possible readings from each
symbol's few likeliest meanings (alternatives), for reading in context
(sequence_puzzles). Handwriting is learned the same way as print, from
lessons shaped like practice sheets (handwriting_lessons). read_word(panel,
vocabulary) reads a printed word in context: the vocabulary word its
letters' likeliest readings spell with the fewest changes. Shapes (domain
'shape', names in SHAPES) are compared by silhouette (an outline filled in),
and learned from shape worksheets (shape_lessons). Colours (domain
'colour', names in COLOURS) are samples of objects (RGBA: alpha the object),
compared by their median colour in CIELAB (colour_of), learned from colour
worksheets (colour_lessons); in a matching lesson an object is compared only
with the colours the sheet names.

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
CONFIDENT = 0.10        # lessons: a symbol is recognised with confidence when nearer than this,
MARGIN = 0.06           # ... or (nearer than REJECT and) nearer by this than any example of another meaning
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
    """A rectangular outline (possibly with rounded corners or a shadow): most of its box border (the outer 3
    cells of each side) is ink and
    the inside of the box (a margin of 15% in from each side) is nearly empty"""
    if min(m.shape) < 12:
        return False
    border = np.concatenate([m[:3].any(axis=0), m[-3:].any(axis=0), m[:, :3].any(axis=1), m[:, -3:].any(axis=1)])
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


def _image_code(image: np.ndarray) -> str:
    """A normalised image (values 0..1) as a hex string, one byte per cell (for the JSON file)"""
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8).tobytes().hex()


def _image_values(code) -> np.ndarray:
    if isinstance(code, str):
        return np.frombuffer(bytes.fromhex(code), np.uint8).astype(float).reshape(SIZE, SIZE) / 255
    return np.asarray(code, dtype=float)              # files written before the hex code: lists of values


UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LOWER = UPPER.lower()
# the names of shapes, known beforehand as words (as the alphabet is); a shape is recognised by its silhouette
SHAPES = ('circle', 'oval', 'square', 'rectangle', 'triangle', 'diamond', 'star', 'heart', 'pentagon',
          'hexagon', 'octagon', 'parallelogram', 'cross', 'trapezoid')


# the names of colours, known beforehand as words; a colour sample is an RGBA image (alpha: the object)
COLOURS = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'grey')
COLOUR_REJECT = 40.0    # colours: a sample further than this (CIELAB) from every example is unknown
COLOUR_CONFIDENT = 20.0  # colours: recognised with confidence when nearer than this (chosen on the labelled
                        # objects of colors 1-3, recognised among the sheet's names: 13 of 18, all right),
COLOUR_MARGIN = 15.0    # ... or nearer by this than any example of another colour


def colour_of(sample) -> np.ndarray:
    """The colour of an object, in CIELAB: the median of its pixels (alpha > 0), so details on it (eyes,
    spots, a dark outline) do not count"""
    a = np.asarray(sample).astype(float)
    rgb = a[..., :3][a[..., 3] > 0] if a.shape[-1] == 4 else a[..., :3].reshape(-1, 3)
    return _lab(np.median(rgb, axis=0))


def _lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def silhouette(mask: np.ndarray) -> np.ndarray:
    """A shape's ink with its insides filled: an outline and a filled shape (or one with a face drawn on it)
    look alike"""
    from scipy import ndimage
    return ndimage.binary_fill_holes(ink(mask))


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def in_domain(meaning, domain: Optional[str]) -> bool:
    if domain is None:
        return meaning not in COLOURS              # colour samples are compared only as colours
    if domain == 'digit':
        return isinstance(meaning, int) and not isinstance(meaning, bool) and 0 <= meaning <= 9
    if domain == 'shape':
        return meaning in SHAPES
    if domain == 'colour':
        return meaning in COLOURS
    if not isinstance(meaning, str) or len(meaning) != 1:
        return False
    return {'upper': meaning in UPPER, 'lower': meaning in LOWER, 'letter': meaning in UPPER + LOWER}[domain]


def symbols_of(panel) -> List[np.ndarray]:
    """A panel's symbols, left to right: its connected pieces of ink, pieces overlapping in columns (one above
    another: the dot of an i or j) taken together"""
    mask = ink(panel)
    pieces = sorted(components(mask), key=lambda c: c[1].start)
    groups: List[List] = []
    for rs, cs, m in pieces:
        if groups:
            g0, g1 = min(p[1].start for p in groups[-1]), max(p[1].stop for p in groups[-1])
            overlap = min(g1, cs.stop) - max(g0, cs.start)
            if overlap > 0.5 * min(g1 - g0, cs.stop - cs.start):
                groups[-1].append((rs, cs, m))
                continue
        groups.append([(rs, cs, m)])
    out = []
    for g in groups:
        r0, c0 = min(p[0].start for p in g), min(p[1].start for p in g)
        r1, c1 = max(p[0].stop for p in g), max(p[1].stop for p in g)
        m = np.zeros((r1 - r0, c1 - c0), bool)
        for rs, cs, pm in g:
            m[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] |= pm
        out.append(m)
    return out


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
        if meaning in COLOURS:                        # a colour sample: its colour, and the object's outline
            a = np.asarray(image)
            mask = a[..., 3] > 0 if a.shape[-1] == 4 else np.ones(a.shape[:2], bool)
            self.examples.append({'meaning': meaning, 'lab': [round(float(v), 2) for v in colour_of(image)],
                                  'image': _image_code(np.zeros((SIZE, SIZE))), 'ink': pack(mask), 'source': source})
            self._cache = None
            return
        mask = silhouette(image) if meaning in SHAPES else ink(image)
        self.examples.append({'meaning': meaning, 'image': _image_code(normalise(mask)),
                              'ink': pack(mask), 'source': source})
        self._cache = None

    def learn_in_order(self, symbols: Sequence[np.ndarray], names: Sequence, source: str = '') -> Dict:
        """An order lesson: symbols shown in a known order (the alphabet on a chart, or a series of sheets one
        per letter) mean the names in that order"""
        if len(symbols) != len(names):
            return {'outcome': 'not a lesson', 'why': f'{len(symbols)} symbols, {len(names)} names'}
        for image, name in zip(symbols, names):
            self.learn(image, name, source)
        return {'outcome': 'learned', 'learned': list(names)}

    def learn_by_anchored_order(self, symbols: Sequence[np.ndarray], names: Sequence, source: str = '',
                                domain: Optional[str] = None, anchors: int = 2) -> Dict:
        """
        symbols are consecutive in the order names (letters down a page in alphabetical order), starting
        somewhere unknown: the start is the one where at least anchors symbols recognised with confidence agree
        and none disagree; the others are then learned by their place
        """
        known = [self.confident(s, domain) for s in symbols]
        fits = []
        for start in range(len(names) - len(symbols) + 1):
            agree = sum(k == names[start + i] for i, k in enumerate(known) if k is not None)
            disagree = sum(k != names[start + i] for i, k in enumerate(known) if k is not None)
            if agree >= anchors and not disagree:
                fits.append(start)
        if len(fits) != 1:
            return {'outcome': 'pending' if not fits else 'contradiction',
                    'why': f'{len(fits)} places in the order fit {known}'}
        start = fits[0]
        learned = []
        for i, (s, k) in enumerate(zip(symbols, known)):
            if k is None:
                self.learn(s, names[start + i], source)
                learned.append(names[start + i])
        return {'outcome': 'learned' if learned else 'confirmed', 'learned': learned}

    def learn_by_counting(self, numerals: Sequence[np.ndarray], count: int, source: str = '') -> Dict:
        """numerals (left to right) mean count; see the module docstring"""
        digits = str(count)
        if count < 0 or len(digits) != len(numerals):
            return {'outcome': 'not a lesson', 'why': f'{len(numerals)} numeral symbol(s) for {count}'}
        if len(numerals) == 1:
            self.learn(numerals[0], count, source)
            return {'outcome': 'learned', 'learned': [count]}
        known = [self.confident(s, 'digit', strict=True) for s in numerals]
        if any(k is not None and str(k) != d for k, d in zip(known, digits)):
            return {'outcome': 'contradiction', 'why': f'read {known} for {count}'}
        unknown = [i for i, k in enumerate(known) if k is None]
        if len(unknown) != 1:
            return {'outcome': 'pending', 'why': f'{len(unknown)} unknown symbols'}
        i = unknown[0]
        self.learn(numerals[i], int(digits[i]), source)
        return {'outcome': 'learned', 'learned': [int(digits[i])]}

    def learn_by_choice(self, symbols: Sequence[np.ndarray], meaning, source: str = '', domain: str = 'digit') -> Dict:
        """One of symbols means meaning (a count, or the lower case of a letter); see the module docstring"""
        known = [self.confident(s, domain, strict=True) for s in symbols]
        possible = [i for i, k in enumerate(known) if k is None or k == meaning]
        if not possible:
            return {'outcome': 'contradiction', 'why': f'read {known}, wanted {meaning}'}
        if len(possible) > 1:
            return {'outcome': 'pending', 'why': f'{len(possible)} symbols possible'}
        self.learn(symbols[possible[0]], meaning, source)
        return {'outcome': 'learned' if known[possible[0]] is None else 'confirmed', 'learned': [meaning]}

    def learn_by_matching(self, symbols: Sequence[np.ndarray], meanings: Sequence, source: str = '',
                          domain: str = 'digit') -> Dict:
        """symbols and meanings pair up one to one; see the module docstring"""
        if len(symbols) != len(meanings):
            return {'outcome': 'not a lesson', 'why': f'{len(symbols)} symbols, {len(meanings)} meanings'}
        left = list(meanings)
        unknown = []
        for s in symbols:
            k = self.confident(s, domain, strict=True, among=set(meanings))
            if k is not None and k in left:
                left.remove(k)
            elif k is not None:
                return {'outcome': 'contradiction', 'why': f'read {k}, meanings {list(meanings)}'}
            else:
                unknown.append(s)
        if len(unknown) != 1:
            return {'outcome': 'pending' if unknown else 'confirmed', 'why': f'{len(unknown)} unknown',
                    'unknown': unknown, 'left': left}
        self.learn(unknown[0], left[0], source)
        return {'outcome': 'learned', 'learned': [left[0]]}

    def _alike(self, a, b, domain) -> bool:
        """Whether two samples look the same (nearer than CONFIDENT): the same thing on two sheets"""
        if domain == 'colour':
            return float(np.linalg.norm(colour_of(a) - colour_of(b))) <= COLOUR_CONFIDENT
        f = (lambda x: normalise(silhouette(x))) if domain == 'shape' else (lambda x: normalise(ink(x)))
        return float(np.abs(f(a) - f(b)).mean()) <= CONFIDENT

    def learn_across(self, pending: Sequence[Tuple[Dict, Dict]]) -> List[Dict]:
        """
        Matching lessons left undecided, taken two at a time: an unknown symbol of one that looks the same as
        an unknown symbol of the other is the same thing, so its name is one both lessons have left; when
        exactly one name is left in both, both symbols are learned as it
        """
        learned = []
        for i in range(len(pending)):
            for j in range(i + 1, len(pending)):
                (la, ra), (lb, rb) = pending[i], pending[j]
                domain = la.get('domain', 'digit')
                if lb.get('domain', 'digit') != domain or 'unknown' not in ra or 'unknown' not in rb:
                    continue
                shared = set(ra['left']) & set(rb['left'])
                if len(shared) != 1:
                    continue
                pairs = [(x, y) for x in ra['unknown'] for y in rb['unknown'] if self._alike(x, y, domain)]
                if len(pairs) != 1:
                    continue
                name = shared.pop()
                for sample in pairs[0]:
                    self.learn(sample, name, f'{la.get("source", "")} + {lb.get("source", "")}')
                learned.append({'lessons': (la.get('source'), lb.get('source')), 'learned': name})
        return learned

    def _case_meanings(self, capitals: Sequence[np.ndarray]) -> Optional[List[str]]:
        known = [self.confident(c, 'upper') for c in capitals]
        return None if any(k is None for k in known) else [k.lower() for k in known]

    def teach(self, lessons: Sequence[Dict], passes: int = 4) -> List[Dict]:
        """
        Run lessons until none teaches more (at most passes times over those still pending); returns each
        lesson's final outcome. A lesson is {'kind', 'source', ...}:
          counting       'numerals', 'count'
          choice         'numerals', 'count' (one numeral is the count)
          matching       'numerals', 'counts' (paired one to one)
          order          'symbols', 'names' (in that order)
          anchored order 'symbols', 'names', 'domain' (consecutive in names, from a start to be found)
          set matching   'symbols', 'names', 'domain' (the symbols are the names, shuffled)
          shape choice   'model', 'symbols' (one symbol is the shape the model is)
          shape matching 'models', 'symbols' (each symbol is the shape one model is)
        choice and matching lessons may also give a 'domain' (default 'digit').
          case choice    'capital', 'symbols' (one symbol is the capital's lower case)
          case matching  'capitals', 'symbols' (each symbol is one capital's lower case)
        The case lessons wait until their capitals are recognised with confidence.
        """
        outcomes: List[Optional[Dict]] = [None] * len(lessons)
        todo = list(range(len(lessons)))
        for _ in range(passes):
            progress = False
            still = []
            for i in todo:
                l = lessons[i]
                src = l.get('source', '')
                kind = l['kind']
                if kind == 'counting':
                    r = self.learn_by_counting(l['numerals'], l['count'], src)
                elif kind == 'choice':
                    r = self.learn_by_choice(l['numerals'], l['count'], src, l.get('domain', 'digit'))
                elif kind == 'matching':
                    r = self.learn_by_matching(l['numerals'], l['counts'], src, l.get('domain', 'digit'))
                elif kind in ('shape choice', 'shape matching'):
                    models = [l['model']] if kind == 'shape choice' else l['models']
                    names = [self.confident(m, 'shape') for m in models]
                    if any(n is None for n in names):
                        r = {'outcome': 'pending', 'why': 'a model shape not recognised'}
                    elif kind == 'shape choice':
                        r = self.learn_by_choice(l['symbols'], names[0], src, 'shape')
                    else:
                        r = self.learn_by_matching(l['symbols'], names, src, 'shape')
                elif kind == 'order':
                    r = self.learn_in_order(l['symbols'], l['names'], src)
                elif kind == 'set matching':
                    r = self.learn_by_matching(l['symbols'], l['names'], src, l.get('domain'))
                elif kind == 'anchored order':
                    r = self.learn_by_anchored_order(l['symbols'], l['names'], src, l.get('domain'))
                elif kind in ('case choice', 'case matching'):
                    lower = self._case_meanings([l['capital']] if kind == 'case choice' else l['capitals'])
                    if lower is None:
                        r = {'outcome': 'pending', 'why': 'a capital not recognised'}
                    else:
                        if kind == 'case choice':
                            r = self.learn_by_choice(l['symbols'], lower[0], src, 'lower')
                        else:
                            r = self.learn_by_matching(l['symbols'], lower, src, 'lower')
                        if r['outcome'] in ('learned', 'confirmed') and not l.get('capitals learned'):
                            # the lesson agreed with the capitals as recognised: one more example of each
                            for c, k in zip([l['capital']] if kind == 'case choice' else l['capitals'], lower):
                                self.learn(c, k.upper(), src)
                            l['capitals learned'] = True
                else:
                    raise ValueError(kind)
                outcomes[i] = r
                if r['outcome'] in ('learned', 'confirmed'):
                    progress = progress or r['outcome'] == 'learned'
                elif r['outcome'] == 'pending':
                    still.append(i)
            todo = still
            if not progress and todo:
                # matching lessons still undecided: the same thing on two sheets (learn_across)
                pending = [(lessons[i], outcomes[i]) for i in todo
                           if lessons[i]['kind'] in ('matching', 'set matching')]
                pending = [(dict(l, domain=l.get('domain', 'digit')), r) for l, r in pending]
                if self.learn_across(pending):
                    progress = True
            if not progress or not todo:
                break
        return outcomes

    # -------------------------------------------------------------- recognition and reading

    def _matrix(self):
        if self._cache is None:
            self._cache = (np.array([_image_values(e['image']) for e in self.examples]).reshape(len(self.examples), -1)
                           if self.examples else np.zeros((0, SIZE * SIZE)))
        return self._cache

    def _distances(self, image, domain):
        """(indices of the examples in domain, their distances to image)"""
        keep = [i for i, e in enumerate(self.examples) if in_domain(e['meaning'], domain)]
        if not keep:
            return keep, np.zeros(0)
        if domain == 'colour':
            lab = colour_of(image)
            return keep, np.array([np.linalg.norm(np.array(self.examples[i]['lab']) - lab) for i in keep])
        sample = silhouette(image) if domain == 'shape' else ink(image)
        return keep, np.abs(self._matrix()[keep] - normalise(sample).reshape(1, -1)).mean(axis=1)

    def nearest(self, image, domain: Optional[str] = None) -> Tuple[Optional[object], float]:
        """
        (meaning of the nearest example, its distance), among the examples whose meaning is in domain ('digit',
        'upper', 'lower', 'letter', 'shape', 'colour'; None: all but colours); (None, inf) without such examples
        """
        keep, d = self._distances(image, domain)
        if not keep:
            return None, float('inf')
        i = int(d.argmin())
        return self.examples[keep[i]]['meaning'], float(d[i])

    @staticmethod
    def _limits(domain):
        """(REJECT, CONFIDENT, MARGIN) for domain: colours are compared in CIELAB, the rest as images"""
        return (COLOUR_REJECT, COLOUR_CONFIDENT, COLOUR_MARGIN) if domain == 'colour' else (REJECT, CONFIDENT, MARGIN)

    def recognise(self, image, domain: Optional[str] = None):
        meaning, distance = self.nearest(image, domain)
        return meaning if distance <= self._limits(domain)[0] else None

    def confident(self, image, domain: Optional[str] = None, strict: bool = False, among=None):
        """
        The meaning when it is recognised with confidence: its nearest example nearer than CONFIDENT, or (not
        strict) nearer than REJECT and nearer by MARGIN than the nearest example of any other meaning; else
        None. Ruling a symbol out (choice, matching and place value lessons) is strict: a symbol of a new
        style may be nearer to another meaning's examples by a margin; placing symbols in an order (where
        two or more must agree) and recognising capitals in case lessons are not. among: only these meanings
        are considered (a matching lesson's names: an object is not taken for a colour the sheet does not name)
        """
        reject, confident, margin = self._limits(domain)
        keep, d = self._distances(image, domain)
        if among is not None:                        # only the meanings the lesson names (a matching sheet's list)
            pick = [j for j, i in enumerate(keep) if self.examples[i]['meaning'] in among]
            keep, d = [keep[j] for j in pick], d[pick]
        if not keep:
            return None
        best = int(d.argmin())
        meaning = self.examples[keep[best]]['meaning']
        if d[best] <= confident:
            return meaning
        if strict:
            return None
        others = [d[j] for j, i in enumerate(keep) if self.examples[i]['meaning'] != meaning]
        if d[best] <= reject and (not others or min(others) - d[best] >= margin):
            return meaning
        return None

    def read(self, panel, domain: Optional[str] = None) -> Optional[List]:
        """
        The meanings of a panel's symbols, left to right (pieces one above another, as the dot of an i, are one
        symbol); None if the panel is empty or a symbol is unknown
        """
        symbols = symbols_of(panel)
        if not symbols:
            return None
        out = []
        for m in symbols:
            meaning = self.recognise(m, domain)
            if meaning is None:
                return None
            out.append(meaning)
        return out

    def alternatives(self, image, domain: Optional[str] = None, k: int = 3) -> List[Tuple[object, float]]:
        """The k nearest distinct meanings in domain within REJECT, each with its nearest example's distance"""
        keep, d = self._distances(image, domain)
        out, seen = [], set()
        for j in np.argsort(d):
            if d[j] > self._limits(domain)[0] or len(out) == k:
                break
            meaning = self.examples[keep[j]]['meaning']
            if meaning not in seen:
                seen.add(meaning)
                out.append((meaning, float(d[j])))
        return out

    def readings(self, panel, domain: Optional[str] = None, k: int = 3) -> List[Tuple[List, float]]:
        """
        A panel's possible readings, most likely first: each a list of meanings (one per symbol, left to right)
        with its total distance, from the alternatives of each symbol (at most k ** symbols readings)
        """
        symbols = symbols_of(panel)
        if not symbols:
            return []
        out = [([], 0.0)]
        for m in symbols:
            alts = self.alternatives(m, domain, k)
            if not alts:
                return []
            out = [(r + [a], c + d) for r, c in out for a, d in alts]
        return sorted(out, key=lambda e: e[1])

    def read_word(self, panel, vocabulary: Sequence[str], most: Optional[int] = None) -> Optional[str]:
        """
        A printed word, read in context: each letter's few likeliest readings (capitals or lower case, without
        case), and the word of vocabulary that needs the fewest changes to be spelt by them (a letter counts as
        spelt when it is among its symbol's readings): at most most changes (default: one per 3 letters of the
        word, 1 to 2), and fewer than any other word needs
        """
        symbols = symbols_of(panel)
        if not symbols:
            return None
        alts = []
        for m in symbols:
            options = {str(a).lower() for a, _ in self.alternatives(m, 'letter', 3)}
            alts.append(options)

        def changes(word):
            w = word.lower()
            row = list(range(len(w) + 1))
            for i, options in enumerate(alts, 1):
                prev, row[0] = row[0], i
                for j, ch in enumerate(w, 1):
                    prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ch not in options))
            return row[-1]
        scored = sorted((changes(w), w) for w in vocabulary)
        allowed = most if most is not None else max(1, min(2, len(scored[0][1]) // 3))
        if scored[0][0] > allowed or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][1]

    def read_number(self, panel) -> Optional[int]:
        symbols = self.read(panel, 'digit')
        if not symbols:
            return None
        return int(''.join(str(s) for s in symbols))

    def read_letter(self, panel, domain: str = 'letter') -> Optional[str]:
        symbols = self.read(panel, domain)
        return symbols[0] if symbols and len(symbols) == 1 else None

    def draw(self, meanings: Sequence) -> Optional[np.ndarray]:
        """A panel showing the meanings side by side, each written as its first stored example (bottoms aligned)"""
        glyphs = []
        for meaning in meanings:
            example = next((e for e in self.examples if e['meaning'] == meaning), None)
            if example is None:
                return None
            glyphs.append(unpack(example['ink']))
        h = max(g.shape[0] for g in glyphs)
        parts = []
        for g in glyphs:
            parts += [np.pad(g, ((h - g.shape[0], 0), (0, 0))), np.zeros((h, 2), int)]
        return np.hstack(parts[:-1])

    def draw_number(self, value: int) -> Optional[np.ndarray]:
        """A panel showing value, written with the first stored example of each digit"""
        return self.draw([int(ch) for ch in str(value)])
