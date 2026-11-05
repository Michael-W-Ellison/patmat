#!/usr/bin/env python3
"""
Handwriting Lessons - teaching the program handwritten digits and letters

The program already reads printed digits and letters (digit_lessons,
letter_lessons). Here it is taught handwriting the same way: by lessons
shaped like a child's practice sheets, built from handwriting datasets,
whose labels are used only to put the sheets together (which sample goes
where) and to score the test, never given to the program with a sample.

Sources (downloaded once into a cache directory):
- MNIST (https://storage.googleapis.com/cvdf-datasets/mnist/): handwritten
  digits, 28 x 28; training writers for the lessons, test writers for the
  test.
- EMNIST "byclass" (https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip):
  handwritten digits, capitals and lower case letters (NIST Special Database
  19), 28 x 28, stored transposed (turned upright here); training writers for
  the lessons, test writers for the test.
The files say nothing of who wrote which sample, so one "sheet" here holds
samples of several writers.

Lessons (curriculum, seeded):
- practice sheets (order lessons): 0 1 2 ... 9, A B C ... Z, a b c ... z
  written in order, one sample each; the program learns each by its place.
- runs (anchored order lessons): RUN consecutive symbols from a start not
  told (5 6 7 8 9; k l m n o); placed by at least two symbols the program
  already recognises, the others learned by their place.
- sets (matching lessons): SET different symbols, shuffled, with the names
  of the set given ("these are 2, 5, 7, 8, 9 in some order"); the ones
  recognised strictly are paired off, a single one left is learned.

Test: samples of the test writers, read alone (with the domain known: a
digit, a capital, a lower case letter), and in worksheet-style items made of
them: number sequences going up or down by a fixed step ("2 4 6 8 10 _")
and rows of consecutive letters with 2 or 3 missing ("a b _ d _ f").

Results (as committed: 1 practice sheet and 1000 runs per alphabet; sets
not used: 1000 per alphabet taught 7 and added 13 mislabelled examples):
- learned: 62 examples from the practice sheets, 3788 from runs (of 3000
  runs, 1554 placed by anchors); checked against the dataset labels, 9 of
  the 3850 are mislabelled.
- 500 test writers' samples each, read alone: digits 210 -> 435, capitals
  181 -> 361, lower case 157 -> 304 (print knowledge only -> after).
  With 5 practice sheets and no runs: 317, 281, 223.
- worksheet-style items of test writers' samples (reading in context,
  sequence_puzzles): number sequences 40/50 right, 1 wrong; rows of
  capitals with letters missing 16/25, 2 wrong; lower case 13/25, 3 wrong
  (print knowledge only: 0, 2 and 1 right). The reading in context's
  FIRST_CHOICES was chosen on items of training writers' samples
  (test_items(part='train')), not on these.

Run: python handwriting_lessons.py [cache directory]  (teaches the memory saved by letter_lessons, LEARNED,
prints the lessons' outcomes and the test)
"""

import gzip
import os
import sys
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np

from digit_lessons import LEARNED
from symbol_learning import LOWER, UPPER, SymbolMemory

MNIST = 'https://storage.googleapis.com/cvdf-datasets/mnist/{}'
EMNIST = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
SHEETS = 1              # practice sheets per alphabet
RUNS = 1000             # runs per alphabet
SETS = 0                # sets per alphabet (1000 taught 7 per 1000 and mislabelled 13: not used)
RUN = 5                 # symbols in a run
SET = 5                 # symbols in a set
TEST = 500              # test samples read alone, per alphabet
ITEMS = 50              # test items of each kind
DIGITS = list(range(10))


# ------------------------------------------------------------------ data

def _idx(path: str) -> np.ndarray:
    with gzip.open(path) as f:
        data = f.read()
    n = int.from_bytes(data[4:8], 'big')
    if data[3] == 3:
        rows, cols = int.from_bytes(data[8:12], 'big'), int.from_bytes(data[12:16], 'big')
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, rows, cols)
    return np.frombuffer(data, np.uint8, offset=8)


def _fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(path, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    return path


def load(cache: str) -> Dict[str, Dict[str, Tuple[np.ndarray, list]]]:
    """{'train' / 'test': {'digit' / 'upper' / 'lower': (images as ink masks, names)}}"""
    out: Dict[str, Dict] = {'train': {}, 'test': {}}
    for part, prefix in (('train', 'train'), ('test', 't10k')):
        X = _idx(_fetch(MNIST.format(f'{prefix}-images-idx3-ubyte.gz'), os.path.join(cache, f'mnist-{prefix}-images.gz')))
        y = _idx(_fetch(MNIST.format(f'{prefix}-labels-idx1-ubyte.gz'), os.path.join(cache, f'mnist-{prefix}-labels.gz')))
        out[part]['digit'] = (X > 127, [int(v) for v in y])
    archive = os.path.join(cache, 'emnist.zip')
    for part in ('train', 'test'):
        images = os.path.join(cache, f'emnist-byclass-{part}-images.gz')
        labels = os.path.join(cache, f'emnist-byclass-{part}-labels.gz')
        if not os.path.exists(images):
            with zipfile.ZipFile(_fetch(EMNIST, archive)) as z:
                for name, target in ((f'gzip/emnist-byclass-{part}-images-idx3-ubyte.gz', images),
                                     (f'gzip/emnist-byclass-{part}-labels-idx1-ubyte.gz', labels)):
                    with z.open(name) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
        X, y = _idx(images), _idx(labels)
        upright = np.transpose(X, (0, 2, 1)) > 127
        for domain, first, alphabet in (('upper', 10, UPPER), ('lower', 36, LOWER)):
            keep = np.nonzero((y >= first) & (y < first + 26))[0]
            out[part][domain] = (upright[keep], [alphabet[int(v) - first] for v in y[keep]])
    return out


def _by_name(images, names) -> Dict:
    groups: Dict = {}
    for i, n in enumerate(names):
        groups.setdefault(n, []).append(i)
    return groups


# ------------------------------------------------------------------ lessons

ALPHABETS = {'digit': DIGITS, 'upper': list(UPPER), 'lower': list(LOWER)}


def lessons(data, seed: int = 0, sheets: int = SHEETS, runs: int = RUNS, sets: int = SETS) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []
    for domain, alphabet in ALPHABETS.items():
        images, names = data['train'][domain]
        groups = _by_name(images, names)

        def sample(name):
            return images[int(rng.choice(groups[name]))]
        for k in range(sheets):
            out.append({'kind': 'order', 'symbols': [sample(n) for n in alphabet], 'names': list(alphabet),
                        'source': f'handwriting practice sheet {domain} {k + 1}', 'truth': list(alphabet)})
        for k in range(runs):
            start = int(rng.integers(0, len(alphabet) - RUN + 1))
            out.append({'kind': 'anchored order', 'symbols': [sample(n) for n in alphabet[start:start + RUN]],
                        'names': list(alphabet), 'domain': domain, 'source': f'handwriting run {domain} {k + 1}',
                        'truth': list(alphabet[start:start + RUN])})
        for k in range(sets):
            chosen = [alphabet[int(i)] for i in rng.choice(len(alphabet), SET, replace=False)]
            out.append({'kind': 'set matching', 'symbols': [sample(n) for n in chosen],
                        'names': sorted(chosen, key=str), 'domain': domain,
                        'source': f'handwriting set {domain} {k + 1}', 'truth': chosen})
    return out


# ------------------------------------------------------------------ test

def read_alone(memory: SymbolMemory, data, seed: int = 1, n: int = TEST) -> Dict[str, Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    out = {}
    for domain in ALPHABETS:
        images, names = data['test'][domain]
        pick = rng.choice(len(names), n, replace=False)
        out[domain] = (sum(memory.nearest(images[i], domain)[0] == names[i] for i in pick), n)
    return out


def _number_panel(digits_images: List[np.ndarray]) -> np.ndarray:
    gap = np.zeros((28, 4), bool)
    parts = []
    for d in digits_images:
        parts += [d, gap]
    return np.hstack(parts[:-1]).astype(int)


def test_items(data, seed: int = 2, n: int = ITEMS, part: str = 'test') -> List[Dict]:
    """Worksheet-style items of test writers' samples: {'kind', 'panels' (None: missing), 'blanks', 'values'}"""
    rng = np.random.default_rng(seed)
    digits = _by_name(*data[part]['digit'])
    out = []
    for _ in range(n):
        step = int(rng.choice([-3, -2, -1, 1, 2, 3, 4, 5]))
        start = int(rng.integers(1, 40)) if step > 0 else int(rng.integers(30, 60))
        values = [start + step * i for i in range(6)]
        panels = [_number_panel([data[part]['digit'][0][int(rng.choice(digits[int(ch)]))] for ch in str(v)])
                  for v in values[:5]]
        out.append({'kind': 'handwritten numbers', 'panels': panels, 'blanks': 1, 'values': [values[5]]})
    for k in range(n):
        domain = 'upper' if k % 2 == 0 else 'lower'
        images, names = data[part][domain]
        groups = _by_name(images, names)
        alphabet = ALPHABETS[domain]
        start = int(rng.integers(0, len(alphabet) - 6 + 1))
        row = alphabet[start:start + 6]
        missing = sorted(int(i) for i in rng.choice(6, int(rng.integers(2, 4)), replace=False))
        panels = [None if i in missing else images[int(rng.choice(groups[ch]))].astype(int) for i, ch in enumerate(row)]
        out.append({'kind': f'handwritten letters ({domain})', 'panels': panels, 'blanks': 0,
                    'values': [row[i] for i in missing]})
    return out


def score_items(memory: SymbolMemory, items: List[Dict]) -> Dict[str, List[int]]:
    from sequence_puzzles import SequencePuzzle, solve
    out: Dict[str, List[int]] = {}
    for item in items:
        answer = solve(SequencePuzzle(item['panels'], item['blanks']), symbols=memory, use_slow=False)
        tally = out.setdefault(item['kind'], [0, 0, 0, 0])          # right, wrong, no answer, items
        tally[3] += 1
        if answer is None or 'values' not in answer:
            tally[2] += 1
        elif answer['values'] == item['values']:
            tally[0] += 1
        else:
            tally[1] += 1
    return out


def mislabelled(memory: SymbolMemory, plan: List[Dict]) -> Dict[str, List[int]]:
    """
    Checking the lessons (with the labels the program was not given): per lesson kind, [examples learned,
    of them mislabelled]; an example is traced to its sample by its ink
    """
    from symbol_learning import pack
    truth = {}
    for l in plan:
        for image, name in zip(l['symbols'], l['truth']):
            truth[(l['source'], tuple(pack(image)['rows']))] = name
    kinds = {l['source']: l['kind'] for l in plan}
    out: Dict[str, List[int]] = {}
    for e in memory.examples:
        k = (e['source'], tuple(e['ink']['rows']))
        if k in truth:
            tally = out.setdefault(kinds[e['source']], [0, 0])
            tally[0] += 1
            tally[1] += e['meaning'] != truth[k]
    return out


def report(memory: SymbolMemory, data, title: str) -> None:
    alone = read_alone(memory, data)
    items = score_items(memory, test_items(data))
    print(f'== {title}: ' + ', '.join(f'{d} read {r}/{n}' for d, (r, n) in alone.items()))
    for kind, (r, w, u, n) in items.items():
        print(f'   {kind:28s} right {r}/{n}, wrong {w}, no answer {u}')


def teach(cache: str, path: str = LEARNED, verbose: bool = True, **kwargs) -> SymbolMemory:
    data = load(cache)
    memory = SymbolMemory(path if os.path.exists(path) else None)
    memory.examples = [e for e in memory.examples if not e['source'].startswith('handwriting')]
    if verbose:
        report(memory, data, 'print only')
    plan = lessons(data, **kwargs)
    outcomes = memory.teach(plan)
    if verbose:
        tally: Dict = {}
        for l, r in zip(plan, outcomes):
            key = (l['kind'], l.get('domain', l['source'].split()[3]))
            tally.setdefault(key, {}).setdefault(r['outcome'], 0)
            tally[key][r['outcome']] += 1
        for key, counts in sorted(tally.items(), key=str):
            print(f'{key[0]:15s} {key[1]:6s} {counts}')
        for kind, (n, bad) in mislabelled(memory, plan).items():
            print(f'learned by {kind:15s} {n} examples, {bad} mislabelled')
        report(memory, data, 'after the handwriting lessons')
    memory.path = path
    memory.save(path)
    return memory


if __name__ == '__main__':
    teach(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_worksheets'))
