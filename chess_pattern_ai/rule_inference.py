#!/usr/bin/env python3
"""
Rule inference for cellular automata

A cellular automaton updates every cell of a grid at once from the cell's
neighbourhood. The learner is shown frame pairs (a grid and the grid one
step later) and must predict the next frames of new grids. It is not told
the rule; it learns it from a generic family of local rules, the same kind
the ARC solver's local rules and the constraint learner's counts use.

Hypotheses: "the next state is a function of a description of the cell's
neighbourhood", for these descriptions (descriptions(), coarsest first):
    3x3 counts           counts of each state in the 3x3 block, the cell included
    own + R count of S   own state, and how many cells of one state S are in
                         ring R (with more than two states)
    own + R counts       own state, and counts of each state in ring R
    own + 4 in order     own state and the 4 neighbours, in order (N, S, W, E)
    3x3 pattern          the 3x3 block, in order
where the ring R is the 4 neighbours, the 8 neighbours or the rest of the
5x5 square,
and each with two edge conventions: cells outside the grid are state 0
('zero edge') or the grid wraps around ('wrapped'). A hypothesis keeps a
table description -> next state built from the observed transitions; one
description seen with two different next states rejects it.

Prediction uses the coarsest consistent hypothesis that has seen every
description in the grid, if every other such hypothesis predicts the same;
if none has seen them all, or they disagree, the prediction is None
(unknown) rather than a guess.
"""

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ------------------------------------------------------------ automata

def _moore(a: Array, wrap: bool = True, radius: int = 1) -> Dict[int, Array]:
    """Per state s > 0: how many cells of state s are in the (2r+1)^2 square around each cell, itself excluded"""
    h, w = a.shape
    out = {}
    for s in range(1, int(a.max()) + 1 if a.size else 1):
        m = (a == s).astype(int)
        p = np.pad(m, radius, mode='wrap' if wrap else 'constant')
        tot = np.zeros_like(m)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr or dc:
                    tot += p[radius + dr:radius + dr + h, radius + dc:radius + dc + w]
        out[s] = tot
    return out


def life_like(born: Sequence[int], survive: Sequence[int]) -> Callable[[Array], Array]:
    def step(a):
        n = _moore(a).get(1, np.zeros_like(a))
        return np.where(a == 1, np.isin(n, survive), np.isin(n, born)).astype(int)
    return step


def majority(a: Array) -> Array:
    n = _moore(a).get(1, np.zeros_like(a)) + (a == 1)
    return (n >= 5).astype(int)


def fredkin(a: Array) -> Array:
    """Parity of the 4 orthogonal neighbours"""
    p = np.pad(a, 1, mode='wrap')
    s = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return (s % 2).astype(int)


def brians_brain(a: Array) -> Array:
    """0 off, 1 on, 2 dying: on -> dying -> off; off -> on with exactly 2 on neighbours"""
    n = _moore(a).get(1, np.zeros_like(a))
    return np.where(a == 1, 2, np.where(a == 2, 0, (n == 2).astype(int)))


def wireworld(a: Array) -> Array:
    """0 empty, 1 head, 2 tail, 3 conductor: head -> tail -> conductor; conductor -> head with 1 or 2 heads around"""
    heads = _moore(a).get(1, np.zeros_like(a))
    return np.where(a == 1, 2, np.where(a == 2, 3, np.where((a == 3) & np.isin(heads, (1, 2)), 1, a)))


def larger_than_life(a: Array) -> Array:
    """Radius 2: born with 8-10 live cells among the 24 others of the 5x5 square, survives with 7-11"""
    n = _moore(a, radius=2).get(1, np.zeros_like(a))
    return np.where(a == 1, (n >= 7) & (n <= 11), (n >= 8) & (n <= 10)).astype(int)


def drift(a: Array) -> Array:
    """Every cell takes the state of its left neighbour (wrapping)"""
    return np.roll(a, 1, axis=1)


def critters(a: Array, t: int) -> Array:
    """
    Margolus block automaton (Critters): the grid is cut into 2x2 blocks, offset
    by one cell on odd steps; a block with exactly two live cells is unchanged,
    any other is inverted, and one that had three live cells is also turned
    half round (needs the step number)
    """
    out = a.copy()
    off = t % 2
    h, w = a.shape
    for r in range(off, h - 1, 2):
        for c in range(off, w - 1, 2):
            block = a[r:r + 2, c:c + 2]
            s = int(block.sum())
            if s != 2:
                new = 1 - block
                out[r:r + 2, c:c + 2] = np.rot90(new, 2) if s == 3 else new
    return out


_LIFE = life_like([3], [2, 3])
_HIGHLIFE = life_like([3, 6], [2, 3])


def alternating(a: Array, t: int) -> Array:
    """Life on even steps, HighLife on odd steps (needs the step number)"""
    return (_LIFE if t % 2 == 0 else _HIGHLIFE)(a)


# name -> (step function taking (grid, step number), number of states, in the hypothesis family?)
AUTOMATA = {
    'Life B3/S23': (lambda a, t: _LIFE(a), 2, True),
    'HighLife B36/S23': (lambda a, t: _HIGHLIFE(a), 2, True),
    'Seeds B2/S': (lambda a, t: life_like([2], [])(a), 2, True),
    'Day & Night B3678/S34678': (lambda a, t: life_like([3, 6, 7, 8], [3, 4, 6, 7, 8])(a), 2, True),
    'Majority (3x3)': (lambda a, t: majority(a), 2, True),
    'Fredkin parity (4 neighbours)': (lambda a, t: fredkin(a), 2, True),
    "Brian's Brain (3 states)": (lambda a, t: brians_brain(a), 3, True),
    'Wireworld (4 states)': (lambda a, t: wireworld(a), 4, True),
    'Larger than Life (radius 2)': (lambda a, t: larger_than_life(a), 2, True),
    'Drift (left neighbour)': (lambda a, t: drift(a), 2, True),
    'Critters (2x2 blocks, alternating)': (critters, 2, False),
    'Life / HighLife alternating by step': (alternating, 2, False),
}


def random_grid(rng: np.random.Generator, shape, states: int) -> Array:
    if states == 4:                 # Wireworld: mostly empty and conductor, a few heads and tails
        return rng.choice(4, size=shape, p=[0.45, 0.05, 0.05, 0.45])
    return rng.choice(states, size=shape, p=[0.6] + [0.4 / (states - 1)] * (states - 1))


# ------------------------------------------------------------- learner

def _neigh(a: Array, wrap: bool, offsets) -> List[Array]:
    """The grid shifted so that each offset's neighbour sits at every cell (outside cells are 0 unless wrap)"""
    r = max(max(abs(dr), abs(dc)) for dr, dc in offsets) if offsets else 0
    p = np.pad(a, r, mode='wrap' if wrap else 'constant')
    h, w = a.shape
    return [p[r + dr:r + dr + h, r + dc:r + dc + w] for dr, dc in offsets]


_VN = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RING8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
_RING24 = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc]
_BLOCK = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _counts(layers: List[Array], states: int) -> List[Array]:
    return [sum((x == s).astype(int) for x in layers) for s in range(1, states)]


_RINGS = {'4': _VN, '8': _RING8, '5x5': _RING24}


def describe(a: Array, kind: str, wrap: bool, states: int) -> List[tuple]:
    """
    The description of every cell's neighbourhood, row by row. Kinds:
    '3x3 counts', 'own + R count of S' (one state S among ring R),
    'own + R counts' (every state among ring R), 'own + 4 in order',
    '3x3 pattern'; R is 4, 8 or 5x5
    """
    if kind == '3x3 counts':
        parts = _counts(_neigh(a, wrap, _BLOCK), states)
    elif kind.startswith('own + ') and ' count of ' in kind:
        ring, state = kind[6:].split(' count of ')
        parts = [a, sum((x == int(state)).astype(int) for x in _neigh(a, wrap, _RINGS[ring]))]
    elif kind.startswith('own + ') and kind.endswith(' counts'):
        parts = [a] + _counts(_neigh(a, wrap, _RINGS[kind[6:-7]]), states)
    elif kind == 'own + 4 in order':
        parts = [a] + _neigh(a, wrap, _VN)
    else:                                   # 3x3 pattern
        parts = _neigh(a, wrap, _BLOCK)
    return list(zip(*[p.ravel().tolist() for p in parts]))


def descriptions(states: int) -> List[str]:
    """Every description for a number of states, coarsest first"""
    kinds = ['3x3 counts']
    for ring in ('4', '8', '5x5'):
        kinds += [f'own + {ring} count of {s}' for s in range(1, states)] if states > 2 else []
    kinds += ['own + 4 counts', 'own + 8 counts', 'own + 5x5 counts', 'own + 4 in order', '3x3 pattern']
    return kinds


class RuleLearner:
    def __init__(self):
        self.states = 2
        self.history: List[Tuple[Array, Array]] = []
        self._reset()

    def _reset(self):
        self.hypotheses = {(kind, wrap): {} for kind in descriptions(self.states) for wrap in (False, True)}
        self.rejected = set()

    def observe(self, before: Array, after: Array):
        """Learn from one transition"""
        self.history.append((before, after))
        states = max(self.states, int(before.max()) + 1, int(after.max()) + 1)
        if states != self.states:
            # the descriptions depend on the states seen, so a new state changes them: relearn from the history
            self.states = states
            self._reset()
            for b, a in self.history[:-1]:
                self._learn(b, a)
        self._learn(before, after)

    def _learn(self, before: Array, after: Array):
        new = after.ravel().tolist()
        for h, table in self.hypotheses.items():
            if h in self.rejected:
                continue
            for key, v in zip(describe(before, h[0], h[1], self.states), new):
                if table.setdefault(key, v) != v:
                    self.rejected.add(h)
                    break

    def consistent(self) -> List[Tuple[str, bool]]:
        return [h for h in self.hypotheses if h not in self.rejected]

    def predict(self, a: Array) -> Tuple[Optional[Array], Optional[Tuple[str, bool]]]:
        """
        (next grid, hypothesis used): the prediction of the coarsest consistent
        hypothesis that knows every cell, provided every other consistent
        hypothesis that knows every cell agrees; (None, None) if none knows
        every cell or they disagree (the observations do not settle it)
        """
        answer, used = None, None
        for h in self.consistent():
            table = self.hypotheses[h]
            keys = describe(a, h[0], h[1], self.states)
            if all(k in table for k in keys):
                pred = np.array([table[k] for k in keys]).reshape(a.shape)
                if answer is None:
                    answer, used = pred, h
                elif not np.array_equal(pred, answer):
                    return None, None
        return answer, used
