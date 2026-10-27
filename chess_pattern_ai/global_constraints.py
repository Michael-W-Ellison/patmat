#!/usr/bin/env python3
"""
Global constraints on binary boards: connectivity, blocks, islands, degrees

A board state is described by the cells known to be filled (mines), known
to be empty, and unknown, plus the clue cells (numbers, never filled
unless clue_cells_may_fill) and marker cells (cells shown with a marker
value, outside the board). Rules (tuples, as the constraint learner names
its hypotheses):

    ('connected', 'mine' | 'not-mine', 'sides' | 'via V')
        all cells of that class form one group; cells are linked if they
        share a side, or (via V) if both touch the same cell showing marker V
    ('no 2x2', 'mine' | 'not-mine', '')
        no 2x2 block is entirely of that class
    ('islands', 'not-mine', 'one clue = size')
        every group of side-linked empty cells (clue cells included) holds
        exactly one clue, equal to the group's size
    ('degree', V, S)
        every cell showing marker V touches (by a side) a number of filled
        cells in the set S

holds(rule, ...) checks a fully known board; consistent(rule, ...) checks
that a partly known board can still satisfy the rule (a necessary
condition, used to find contradictions).
"""

import random
from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

Pos = Tuple[int, int]
SIDES = ((-1, 0), (1, 0), (0, -1), (0, 1))
DEGREE_SETS = (frozenset({0, 2}), frozenset({2}), frozenset({1}), frozenset({0, 1}), frozenset({0}))


def _near(p: Pos):
    return [(p[0] + dr, p[1] + dc) for dr, dc in SIDES]


def _links(cells: Set[Pos], how: str, markers: Dict[Pos, int]):
    """Adjacency among `cells`: by sides, or (via V) through shared marker cells of value V"""
    adj = {p: set() for p in cells}
    for p in cells:
        for q in _near(p):
            if q in cells:
                adj[p].add(q)
    if how.startswith('via '):
        v = int(how[4:])
        for m, mv in markers.items():
            if mv == v:
                touching = [q for q in _near(m) if q in cells]
                for a in touching:
                    adj[a].update(b for b in touching if b != a)
    return adj


def _groups(cells: Set[Pos], adj) -> List[Set[Pos]]:
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        group, queue = {start}, deque([start])
        seen.add(start)
        while queue:
            p = queue.popleft()
            for q in adj[p]:
                if q not in seen:
                    seen.add(q)
                    group.add(q)
                    queue.append(q)
        out.append(group)
    return out


def _blocks(cells: Set[Pos]):
    return [{(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} for r, c in cells
            if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells]


def holds(rule, mines: Set[Pos], empty: Set[Pos], clues: Dict[Pos, int], markers: Dict[Pos, int]) -> bool:
    """Whether a fully known board satisfies the rule (empty: board cells that are not mines)"""
    kind, what, how = rule
    if kind == 'connected':
        cells = mines if what == 'mine' else empty | set(clues)
        return len(_groups(cells, _links(cells, how, markers))) <= 1
    if kind == 'no 2x2':
        cells = mines if what == 'mine' else empty | set(clues)
        return not _blocks(cells)
    if kind == 'islands':
        white = empty | set(clues)
        for g in _groups(white, _links(white, 'sides', markers)):
            inside = [clues[p] for p in g if p in clues]
            if len(inside) != 1 or inside[0] != len(g):
                return False
        return True
    if kind == 'degree':
        v, allowed = what, how
        return all(sum(q in mines for q in _near(m)) in allowed for m, mv in markers.items() if mv == v)
    raise ValueError(rule)


def consistent(rule, mines: Set[Pos], empty: Set[Pos], unknown: Set[Pos], clues: Dict[Pos, int],
               markers: Dict[Pos, int]) -> bool:
    """Whether a partly known board can still satisfy the rule (False: a contradiction)"""
    kind, what, how = rule
    if not unknown:
        return holds(rule, mines, empty, clues, markers)
    if kind == 'connected':
        known = mines if what == 'mine' else empty | set(clues)
        if len(known) <= 1:
            return True
        possible = known | unknown
        groups = _groups(possible, _links(possible, how, markers))
        return sum(1 for g in groups if g & known) <= 1
    if kind == 'no 2x2':
        known = mines if what == 'mine' else empty | set(clues)
        return not _blocks(known)
    if kind == 'islands':
        white = empty | set(clues)
        possible = white | unknown
        # each known white group: at most one clue, no larger than it, and room to grow to it
        for g in _groups(white, _links(white, 'sides', markers)):
            inside = [clues[p] for p in g if p in clues]
            if len(inside) > 1 or (inside and len(g) > inside[0]):
                return False
            region = _reach(g, possible, len(g) + (inside[0] if inside else len(possible)))
            if inside and len(region) < inside[0]:
                return False
            if not inside and not any(p in clues for p in _reach(g, possible, len(possible))):
                return False
        return True
    if kind == 'degree':
        v, allowed = what, how
        for m, mv in markers.items():
            if mv != v:
                continue
            around = _near(m)
            lo = sum(q in mines for q in around)
            hi = lo + sum(q in unknown for q in around)
            if not any(lo <= s <= hi for s in allowed):
                return False
        return True
    raise ValueError(rule)


def _reach(start: Set[Pos], allowed: Set[Pos], limit: int) -> Set[Pos]:
    """Cells of `allowed` reachable from `start` by sides (stops after `limit` cells)"""
    seen, queue = set(start), deque(start)
    while queue and len(seen) < limit:
        p = queue.popleft()
        for q in _near(p):
            if q in allowed and q not in seen:
                seen.add(q)
                queue.append(q)
    return seen


def evidence(rule, mines: Set[Pos], empty: Set[Pos], clues: Dict[Pos, int], markers: Dict[Pos, int],
             rng: random.Random, shuffles: int = 20) -> Tuple[bool, float]:
    """
    (holds, surprise): whether a fully known board satisfies the rule, and
    how unlikely that would be by chance - the share of `shuffles` boards,
    with the same number of mine cells placed at random over the board, on
    which the rule fails. A rule that merely happens to hold (empty cells
    connected on a sparse board) earns little evidence; one that random
    boards almost never satisfy (all sea cells connected) earns nearly 1.
    """
    ok = holds(rule, mines, empty, clues, markers)
    if not ok:
        return False, 0.0
    board = sorted(mines | empty)
    fails = 0
    for _ in range(shuffles):
        m = set(rng.sample(board, len(mines)))
        fails += not holds(rule, m, set(board) - m, clues, markers)
    return True, fails / shuffles


def rules(marker_values: Iterable[int]):
    """Every global rule for the marker values seen"""
    out = [('connected', 'mine', 'sides'), ('connected', 'not-mine', 'sides'),
           ('no 2x2', 'mine', ''), ('no 2x2', 'not-mine', ''), ('islands', 'not-mine', 'one clue = size')]
    for v in sorted(set(marker_values)):
        out += [('connected', 'mine', f'via {v}')]
        out += [('degree', v, s) for s in DEGREE_SETS]
    return out
