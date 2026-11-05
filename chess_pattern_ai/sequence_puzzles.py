#!/usr/bin/env python3
"""
Sequence Puzzles - "what comes next" in a row of pictures

A sequence puzzle is a row of panels (integer grids, 0 the background) that
follows a pattern, and one or more blanks after it to be filled in; some
puzzles offer answer choices for a single blank. This is the commonest kind
of pattern worksheet for young children: shapes, letters, colours or
pictures repeated in a cycle (AB, AAB, ABC, ...), the same picture at two
sizes, and growing patterns (a caterpillar that gains segments, a tower that
gains floors).

Solving (solve): the fast families below (repeat, runs, growth) are all
tried, and their predictions ranked by the size of the rule, smallest first
(repeat: prefix + period; runs: 2 per place in the cycle of runs, a symbol
and a length rule; growth: 2 per part added, plus 2); the slow and last-resort
families follow only if needed. The best-ranked prediction fills the blanks
(with answer choices: the best-ranked prediction that is one of them).
For A B AA B AAA the runs rule (size 4) is ranked above a period of 7.
Panels are compared exactly (shape and every cell), so two panels are the
same picture only if they are drawn identically.

Rule families:
- repeat: the panels repeat with a period k: panel i equals panel i + k.
  The first panel may stand apart (a train's engine before its cars): the
  rule may start at panel 1 instead of panel 0 (prefix). The rule must be
  seen to repeat (at least one panel after the first k of it); of those, the
  shortest prefix + period is taken, no prefix first.
- runs: the panels form runs of equal panels (A B AA B AAA ...); the run
  symbols repeat with a period and each symbol's run lengths grow by a fixed
  step (an arithmetic progression, at least two complete runs per symbol).
  The last run may still be under way.
- growth: each panel is the previous one with parts added: the previous
  panel appears unchanged at the same place within the next (anchor), every
  step enlarges the panel by the same amount, and the parts added at each
  step are those of the step before, each moved by its own fixed amount (a
  caterpillar's new segments one segment further along; a tower's new floor
  on top). Needs three panels.
- numbers: each panel read as a number with symbols the program has
  learned (symbol_learning, digit_lessons; pass symbols=SymbolMemory), the
  numbers going up or down by a fixed step; the next numbers are written
  with the learned digits, and given as 'values' too. Rule size 2.
  Read in context: of each symbol's 3 likeliest meanings (within REJECT),
  the reading of the row that goes by a fixed step at the smallest total
  distance (a 7 read as 1 is put right by its neighbours). The rule's size
  is 2 plus that distance, so a row read as letters and as numbers is taken
  as the one it reads as better.
- letters: as numbers, with each panel read as a letter (all capitals or
  all lower case) and its place in the alphabet going up or down by a fixed
  step; the next letters are given as 'values'.
Panels may be missing within the row (None: "fill in the missing
letters"); only numbers and letters fill them in, and the missing panels'
values come first in 'values', then the blanks'.
- shapes: each panel read as the name of its shape (learned: shape_lessons),
  so copies of a shape drawn a little differently (cut from a page) are the
  same; the names repeat as in repeat (rule size + 1 + reading distance).
  Panels of one name must be copies of one drawing (alike in ink, size and
  colours: SAME_DRAWING, SAME_SIZE): names do not merge two pictures, sizes
  or colours that the panels tell apart.
- arc_step: each step from a panel to the next as a training pair for the
  ARC pattern solver (arc_pattern_solver); the last panel is the test input.
  Slow (seconds to minutes), so tried only when the families above predict
  nothing (or nothing among the choices); several blanks are filled one at a
  time.
- repeat_all: the given panels repeated as a whole (a rule not yet seen to
  repeat: "the row so far is one cycle"); the last resort.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
MAX_PREFIX = 1          # repeat: panels before the cycle starts (a train's engine)
SAME_DRAWING = 0.1      # shapes: panels of one name differ in normalised ink by at most this (copies on a page: 0.094)
SAME_SIZE = 0.15        # shapes: ... and in height and width by at most this share
FIRST_CHOICES = 1 / 3   # numbers, letters: share of given panels to be read in context as their likeliest reading
                        # (chosen on validation items of training writers: handwriting_lessons)
FAMILIES = ('repeat', 'runs', 'growth', 'numbers', 'letters', 'shapes', 'arc_step', 'repeat_all')
SLOW = ('arc_step',)


@dataclass
class SequencePuzzle:
    """
    panels: the given panels in order, None for a missing one ("fill in the missing letters"); blanks: how many
    panels come next (may be 0 when panels are missing); choices: candidates for 1 blank
    """
    panels: List[Optional[Array]]
    blanks: int = 1
    choices: List[Array] = field(default_factory=list)

    def __post_init__(self):
        self.panels = [None if p is None else np.asarray(p, dtype=int) for p in self.panels]
        self.choices = [np.asarray(c, dtype=int) for c in self.choices]
        self.missing = [i for i, p in enumerate(self.panels) if p is None]
        if not self.panels or self.blanks < 0 or (self.blanks == 0 and not self.missing):
            raise ValueError('a sequence puzzle has at least one panel and something to fill in')


def _same(a: Optional[Array], b: Optional[Array]) -> bool:
    return a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)


def _symbols(panels: Sequence[Array]) -> List[int]:
    """Each panel as the index of the first equal panel"""
    out = []
    for i, p in enumerate(panels):
        out.append(next(j for j in range(i + 1) if _same(panels[j], p)))
    return out


# ------------------------------------------------------------------ families

def repeat(panels: List[Array], count: int) -> List[Tuple[str, List[Array]]]:
    s = _symbols(panels)
    n = len(s)
    options = []
    for prefix in range(MAX_PREFIX + 1):
        for k in range(1, n - prefix):          # the cycle seen at least once and a bit
            if all(s[i] == s[i + k] for i in range(prefix, n - k)):
                options.append((prefix + k, prefix, k))
                break
    if not options:
        return []
    _, prefix, k = min(options)
    nxt = [panels[prefix + (n - prefix + j) % k] for j in range(count)]
    name = f'repeat every {k}' + (f' after {prefix} panel' + ('s' if prefix > 1 else '') if prefix else '')
    return [(name, [p.copy() for p in nxt], prefix + k)]


def repeat_all(panels: List[Array], count: int) -> List[Tuple[str, List[Array]]]:
    n = len(panels)
    return [(f'the {n} panels repeated', [panels[j % n].copy() for j in range(count)], n)]


def runs(panels: List[Array], count: int) -> List[Tuple[str, List[Array]]]:
    s = _symbols(panels)
    rl: List[List[int]] = []
    for x in s:
        if rl and rl[-1][0] == x:
            rl[-1][1] += 1
        else:
            rl.append([x, 1])
    syms = [r[0] for r in rl]
    m = len(rl)
    n = len(s)
    period = next((k for k in range(1, m) if all(syms[i] == syms[i + k] for i in range(m - k))), None)
    if period is None or period == m:
        return []
    # each place in the cycle: its run lengths (the last run may be unfinished), first value and step
    rules = {}
    for place in range(period):
        lengths = [rl[i][1] for i in range(place, m, period)]
        last_is_open = (m - 1) % period == place
        complete = lengths[:-1] if last_is_open else lengths
        if len(complete) < 2:
            return []
        step = complete[1] - complete[0]
        if any(complete[i + 1] - complete[i] != step for i in range(len(complete) - 1)):
            return []
        expected = [complete[0] + step * i for i in range(len(lengths))]
        if any(e < 1 for e in expected):
            return []
        if last_is_open and lengths[-1] > expected[-1]:
            return []
        rules[place] = (complete[0], step)
    if all(step == 0 for _, step in rules.values()):
        return []                                    # equal runs: a plain repeat
    # expand the runs the rule gives until the blanks are covered; they must reproduce the given panels
    sym_panel = {x: panels[s.index(x)] for x in set(s)}
    expanded: List[int] = []
    t = 0
    while len(expanded) < n + count:
        first, step = rules[t % period]
        expanded += [syms[t % period]] * (first + step * (t // period))
        t += 1
    if expanded[:n] != s:
        return []
    return [('runs growing in a cycle', [sym_panel[x].copy() for x in expanded[n:n + count]], 2 * period)]


def _components(mask: Array) -> List[Array]:
    """4-connected components of a boolean mask, as boolean masks"""
    seen = np.zeros(mask.shape, bool)
    out = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        comp = np.zeros(mask.shape, bool)
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            comp[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u] and not seen[v, u]:
                    seen[v, u] = True
                    stack.append((v, u))
        out.append(comp)
    return out


def _parts(grid: Array, added: Array) -> List[Tuple[Tuple[int, int], Array]]:
    """The added cells of grid as parts: (top-left corner, cropped grid with 0 outside the part)"""
    out = []
    for comp in _components(added):
        rows, cols = np.nonzero(comp)
        r0, c0, r1, c1 = rows.min(), cols.min(), rows.max() + 1, cols.max() + 1
        out.append(((int(r0), int(c0)), np.where(comp[r0:r1, c0:c1], grid[r0:r1, c0:c1], 0)))
    return sorted(out, key=lambda e: (e[0], e[1].shape, e[1].tobytes()))


def _anchors(a: Array, b: Array) -> List[Tuple[int, int]]:
    """Places where a appears unchanged within b"""
    dh, dw = b.shape[0] - a.shape[0], b.shape[1] - a.shape[1]
    if dh < 0 or dw < 0 or (dh == 0 and dw == 0):
        return []
    return [(r, c) for r in range(dh + 1) for c in range(dw + 1)
            if np.array_equal(b[r:r + a.shape[0], c:c + a.shape[1]], a)]


def growth(panels: List[Array], count: int) -> List[Tuple[str, List[Array]]]:
    if len(panels) < 3:
        return []
    grow = (panels[1].shape[0] - panels[0].shape[0], panels[1].shape[1] - panels[0].shape[1])
    if any((b.shape[0] - a.shape[0], b.shape[1] - a.shape[1]) != grow for a, b in zip(panels, panels[1:])):
        return []
    common = set(_anchors(panels[0], panels[1]))
    for a, b in zip(panels[1:], panels[2:]):
        common &= set(_anchors(a, b))
    out = []
    for anchor in sorted(common):
        steps = []
        for a, b in zip(panels, panels[1:]):
            added = np.ones(b.shape, bool)
            added[anchor[0]:anchor[0] + a.shape[0], anchor[1]:anchor[1] + a.shape[1]] = False
            steps.append(_parts(b, added & (b != 0)))
        if any(len(p) != len(steps[0]) or not p for p in steps):
            continue
        # each part keeps its picture and moves by a fixed amount from step to step
        moves = None
        ok = True
        for prev, cur in zip(steps, steps[1:]):
            if any(not _same(x[1], y[1]) for x, y in zip(prev, cur)):
                ok = False
                break
            m = [(y[0][0] - x[0][0], y[0][1] - x[0][1]) for x, y in zip(prev, cur)]
            if moves is not None and m != moves:
                ok = False
                break
            moves = m
        if not ok:
            continue
        last, parts = panels[-1], steps[-1]
        made = []
        for _ in range(count):
            grid = np.zeros((last.shape[0] + grow[0], last.shape[1] + grow[1]), dtype=int)
            grid[anchor[0]:anchor[0] + last.shape[0], anchor[1]:anchor[1] + last.shape[1]] = last
            parts = [((p[0] + d[0], p[1] + d[1]), pic) for (p, pic), d in zip(parts, moves)]
            fits = True
            for (r, c), pic in parts:
                if r < 0 or c < 0 or r + pic.shape[0] > grid.shape[0] or c + pic.shape[1] > grid.shape[1]:
                    fits = False
                    break
                region = grid[r:r + pic.shape[0], c:c + pic.shape[1]]
                region[pic != 0] = pic[pic != 0]
            if not fits:
                break
            made.append(grid)
            last = grid
        if len(made) == count:
            out.append((f'{len(moves)} part(s) added per step, anchored at {anchor}', made,
                        2 * len(moves) + 2))
    return out


def arc_step(panels: List[Array], count: int) -> List[Tuple[str, List[Array]]]:
    from matrix_puzzles import _arc, _arc_task
    seq = list(panels)
    names = []
    for _ in range(count):
        found = _arc(_arc_task(list(zip(seq, seq[1:])), seq[-1]))
        if not found:
            return []
        names.append(found[0][0])
        seq.append(found[0][1])
    return [('ARC: ' + names[0], seq[len(panels):], None)]


def _ordered(panels, count, positions, draw, name):
    """
    positions: each given panel's place in an order (a number's value, a letter's place in the alphabet), None
    for a missing panel; they must go up or down by a fixed step (not 0) from panel to panel. Fills the missing
    panels and the count next ones: (rule, panels drawn, size 2, places)
    """
    known = [(i, v) for i, v in enumerate(positions) if v is not None]
    if len(known) < 2:
        return []
    (i0, v0), (i1, v1) = known[0], known[1]
    if (v1 - v0) % (i1 - i0):
        return []
    step = (v1 - v0) // (i1 - i0)
    if step == 0 or any(v != v0 + step * (i - i0) for i, v in known):
        return []
    wanted = [i for i, v in enumerate(positions) if v is None] + list(range(len(panels), len(panels) + count))
    places = [v0 + step * (i - i0) for i in wanted]
    drawn = [draw(v) for v in places]
    if any(d is None for d in drawn):
        return []
    return [(f'{name} going {"up" if step > 0 else "down"} by {abs(step)}', drawn, 2, places)]


def _in_context(options: List[Optional[List[Tuple[int, float]]]]):
    """
    Reading in context: options holds, for each panel, its possible places in the order with their reading
    distances (None for a missing panel). The start and step (not 0) that give every given panel one of its
    places, at the smallest total distance: the places chosen ([None for missing panels]), or None
    """
    known = [i for i, o in enumerate(options) if o is not None]
    if len(known) < 2 or any(not options[i] for i in known):
        return None
    i0, i1 = known[0], known[1]
    fits = []
    for v0, c0 in options[i0]:
        for v1, c1 in options[i1]:
            if (v1 - v0) % (i1 - i0):
                continue
            step = (v1 - v0) // (i1 - i0)
            if step == 0:
                continue
            total, places = c0 + c1, {i0: v0, i1: v1}
            for i in known[2:]:
                want = v0 + step * (i - i0)
                cost = next((c for v, c in options[i] if v == want), None)
                if cost is None:
                    break
                total += cost
                places[i] = want
            else:
                fits.append((total, [places.get(i) for i in range(len(options))]))
    if not fits:
        return None
    fits.sort(key=lambda f: f[0])
    places = fits[0][1]
    # most given panels must be read as their likeliest reading: a row put together from second choices is
    # too unsure to answer
    firsts = sum(places[i] == options[i][0][0] for i in known)
    if firsts < FIRST_CHOICES * len(known):
        return None
    return places, fits[0][0]


def numbers(panels: List[Optional[Array]], count: int, symbols=None) -> List[tuple]:
    """
    Each panel read as a number with the learned symbols (symbol_learning.SymbolMemory); the numbers go up or
    down by a fixed step (not 0: equal numbers are a repeat); missing panels are filled in too. Panels are
    read in context: of each symbol's few likeliest readings, those that make the row go by a fixed step
    """
    given = [p for p in panels if p is not None]
    if symbols is None or len(given) < 2 or (len(given) < 3 and len(given) == len(panels)):
        return []
    options = []
    for p in panels:
        if p is None:
            options.append(None)
            continue
        seen = {}
        for meanings, cost in symbols.readings(p, 'digit'):
            value = int(''.join(str(m) for m in meanings))
            seen.setdefault(value, cost)
        options.append(sorted(seen.items(), key=lambda e: e[1]))
    fit = _in_context(options)
    if fit is None:
        return []
    values, cost = fit
    return [(rule, drawn, size + cost, at) for rule, drawn, size, at in
            _ordered(panels, count, values, lambda v: symbols.draw_number(v) if v >= 0 else None, 'numbers')]


def letters(panels: List[Optional[Array]], count: int, symbols=None) -> List[tuple]:
    """
    Each panel read as a letter with the learned symbols, all capitals or all lower case; their places in the
    alphabet go up or down by a fixed step; missing panels are filled in too; read in context as numbers are.
    Values: the letters
    """
    from symbol_learning import UPPER, LOWER
    given = [p for p in panels if p is not None]
    if symbols is None or len(given) < 2 or (len(given) < 3 and len(given) == len(panels)):
        return []
    out = []
    for domain, alphabet in (('upper', UPPER), ('lower', LOWER)):
        options = []
        for p in panels:
            if p is None:
                options.append(None)
                continue
            options.append([(alphabet.index(r[0]), c) for r, c in symbols.readings(p, domain) if len(r) == 1])
        fit = _in_context(options)
        if fit is None:
            continue
        places, cost = fit
        found = _ordered(panels, count, places,
                         lambda v, a=alphabet: symbols.draw([a[v]]) if 0 <= v < len(a) else None, 'letters')
        out += [(rule, drawn, size + cost, [alphabet[v] for v in at]) for rule, drawn, size, at in found]
    return sorted(out, key=lambda e: e[2])[:1]          # capitals or lower case: the better reading


def shapes(panels: List[Optional[Array]], count: int, symbols=None) -> List[tuple]:
    """
    Each panel read as the name of its shape with the learned symbols (domain 'shape': by silhouette, so
    copies of a shape need not be drawn alike); the names repeat as in repeat. Values: the next names
    """
    if symbols is None or any(p is None for p in panels):
        return []
    names, cost = [], 0.0
    for p in panels:
        alts = symbols.alternatives(p, 'shape', 1)
        if not alts:
            return []
        names.append(alts[0][0])
        cost += alts[0][1]
    # panels given one name must be copies of one drawing: alike in ink, size and colours (else the names would
    # merge what the panels themselves tell apart: two pictures, two sizes, two colours)
    from symbol_learning import ink, normalise
    looks = [normalise(ink(p)) for p in panels]
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            if names[i] != names[j]:
                continue
            a, b = panels[i], panels[j]
            if np.abs(looks[i] - looks[j]).mean() > SAME_DRAWING or \
                    any(abs(x - y) > SAME_SIZE * max(x, y) for x, y in zip(a.shape, b.shape)) or \
                    set(np.unique(a)) != set(np.unique(b)):
                return []
    stand_ins = [np.full((1, 1), sorted(set(names)).index(n) + 1) for n in names]
    found = repeat(stand_ins, count)
    if not found:
        return []
    rule, nxt, size = found[0]
    values = [sorted(set(names))[int(p[0, 0]) - 1] for p in nxt]
    drawn = [symbols.draw([v]) for v in values]
    return [(f'shapes: {rule}', drawn, size + 1 + cost, values)]


FAMILY_FUNCTIONS = {'repeat': repeat, 'runs': runs, 'growth': growth, 'numbers': numbers, 'letters': letters,
                    'shapes': shapes,
                    'arc_step': arc_step,
                    'repeat_all': repeat_all}


def predictions(puzzle: SequencePuzzle, families: Sequence[str] = FAMILIES,
                use_slow: Optional[bool] = None, symbols=None) -> List[Dict]:
    """
    Every family's predictions, best-ranked first: [{'family', 'rule', 'panels'}] (panels: one per blank;
    the numbers family also gives 'values'). use_slow: whether to try arc_step (None: only if the families
    before it predict nothing, or nothing among the choices). repeat_all is only used when nothing before it
    predicts. symbols: a SymbolMemory to read numbers with (the numbers family needs one)
    """
    out = []

    def run(name):
        try:
            if name in ('numbers', 'letters', 'shapes'):
                found = FAMILY_FUNCTIONS[name](puzzle.panels, puzzle.blanks, symbols)
            elif puzzle.missing:
                found = []                  # the other families continue a complete row
            else:
                found = FAMILY_FUNCTIONS[name](puzzle.panels, puzzle.blanks)
        except Exception:
            found = []
        for entry in found:
            e = {'family': name, 'rule': entry[0], 'panels': entry[1], 'size': entry[2]}
            if len(entry) > 3:
                e['values'] = entry[3]
            out.append(e)

    def wanted():
        if not out:
            return True
        return bool(puzzle.choices) and not any(_same(p['panels'][0], c) for p in out for c in puzzle.choices)

    for name in families:
        if name in SLOW or name == 'repeat_all':
            out.sort(key=lambda e: e['size'] if e['size'] is not None else float('inf'))
        if name in SLOW:
            if use_slow or (use_slow is None and wanted()):
                run(name)
        elif name == 'repeat_all':
            if wanted():
                run(name)
        else:
            run(name)
    return out


def solve(puzzle: SequencePuzzle, **kwargs) -> Optional[Dict]:
    """
    The answer: the best-ranked prediction (whose first panel is among the choices, if the puzzle has them), as
    a predictions() entry with, for a puzzle with choices, 'choice' (its index); None if no rule fits
    """
    for p in predictions(puzzle, **kwargs):
        if not puzzle.choices:
            return p
        for i, c in enumerate(puzzle.choices):
            if _same(p['panels'][0], c):
                return dict(p, choice=i)
    return None
