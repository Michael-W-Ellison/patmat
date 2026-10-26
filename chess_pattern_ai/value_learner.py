#!/usr/bin/env python3
"""
Rule learner for number-placement puzzles (value_puzzles.py)

The learner is not told the rules. After each attempt it is shown the
solution, and it checks a generic family of hypotheses against it:

- different: "within every region of kind K the values are all different",
  K = row, column, a x b box (every box size that tiles the board, rows
  and columns excepted), diagonal, anti-diagonal, or the given cages
- clue:      "a cage whose clue shows symbol S has target = f(its values)",
  f = sum, product, difference, quotient, largest, smallest, number of
  cells, or the value of a one-cell cage

One violation rejects a hypothesis; one confirmed ACCEPT_AFTER times
without a violation is accepted. The values run from 1 to the largest
value seen in a solution.

Deduction narrows each empty cell's possible values with the accepted
constraints (value_puzzles.Constraints: a decided value leaves the rest of
its all-different regions; in a region with as many cells as values, a
value with one possible place goes there; a cage keeps only values that
appear in some combination meeting its clue). When that decides no empty
cell, search: each remaining value of each undecided cell is tried, and a
value that leads to a contradiction is removed. A cell with one value left
is filled; otherwise the learner guesses a value of the cell with the
fewest left.
"""

import random
from typing import Dict, List, Optional, Tuple

from value_puzzles import OPERATIONS, Constraints, Observation

Pos = Tuple[int, int]
ACCEPT_AFTER = 30


class ValueLearner:
    def __init__(self, rng: Optional[random.Random] = None, accept_after: int = ACCEPT_AFTER, search: bool = True):
        self.rng = rng or random.Random()
        self.accept_after = accept_after
        self.search = search
        self.hypotheses: Dict[tuple, Dict[str, int]] = {}
        self.accepted_at: Dict[tuple, int] = {}
        self.values = 0
        self.games = 0
        self.moves_by = {'deduced': 0, 'guessed': 0}

    # ---------------------------------------------------------- regions
    @staticmethod
    def region_kinds(n: int) -> List[str]:
        kinds = ['row', 'column']
        kinds += [f'box {a}x{b}' for a in range(1, n + 1) for b in range(1, n + 1)
                  if n % a == 0 and n % b == 0 and a * b >= 2 and (a, b) not in ((1, n), (n, 1))]
        return kinds + ['diagonal', 'anti-diagonal', 'cage']

    @staticmethod
    def regions(kind: str, n: int, cages: List[List[int]]) -> List[List[Pos]]:
        if kind == 'row':
            return [[(r, c) for c in range(n)] for r in range(n)]
        if kind == 'column':
            return [[(r, c) for r in range(n)] for c in range(n)]
        if kind == 'diagonal':
            return [[(i, i) for i in range(n)]]
        if kind == 'anti-diagonal':
            return [[(i, n - 1 - i) for i in range(n)]]
        if kind == 'cage':
            groups: Dict[int, List[Pos]] = {}
            for r in range(n):
                for c in range(n):
                    if cages[r][c] >= 0:
                        groups.setdefault(cages[r][c], []).append((r, c))
            return list(groups.values())
        a, b = map(int, kind[4:].split('x'))
        return [[(i + r, j + c) for r in range(a) for c in range(b)] for i in range(0, n, a) for j in range(0, n, b)]

    # ---------------------------------------------------------- learning
    def accepted(self) -> List[tuple]:
        return [h for h, s in self.hypotheses.items() if s['violated'] == 0 and s['confirmed'] >= self.accept_after]

    def end_game(self, final: Observation):
        """Learn from the solution shown after the attempt"""
        self.games += 1
        n = len(final.values)
        self.values = max(self.values, max(max(row) for row in final.values))
        for kind in self.region_kinds(n):
            self.hypotheses.setdefault(('different', kind), {'confirmed': 0, 'violated': 0})
        for _, symbol in final.clues.values():
            for op in OPERATIONS:
                self.hypotheses.setdefault(('clue', symbol, op), {'confirmed': 0, 'violated': 0})
        for h, stats in self.hypotheses.items():
            if h[0] == 'different':
                for reg in self.regions(h[1], n, final.cages):
                    vals = [final.values[r][c] for r, c in reg]
                    stats['confirmed' if len(set(vals)) == len(vals) else 'violated'] += 1
            else:
                _, symbol, op = h
                for cells in self.regions('cage', n, final.cages):
                    target, s = final.clues[final.cages[cells[0][0]][cells[0][1]]]
                    if s == symbol:
                        got = OPERATIONS[op]([final.values[r][c] for r, c in cells])
                        stats['confirmed' if got == target else 'violated'] += 1
        for h in self.accepted():
            self.accepted_at.setdefault(h, self.games)

    # --------------------------------------------------------- deduction
    def constraints(self, obs: Observation) -> Constraints:
        n = len(obs.values)
        values = self.values or n
        regions, cages = [], []
        accepted = self.accepted()
        for h in accepted:
            if h[0] == 'different':
                regions += self.regions(h[1], n, obs.cages)
        for cells in self.regions('cage', n, obs.cages):
            target, symbol = obs.clues[obs.cages[cells[0][0]][cells[0][1]]]
            for h in accepted:
                if h[0] == 'clue' and h[1] == symbol:
                    cages.append((cells, OPERATIONS[h[2]], target))
        return Constraints(n, values, regions, cages)

    def deduce(self, obs: Observation, search: bool = False) -> Optional[Dict[Pos, set]]:
        """Possible values of every cell (None if the accepted constraints contradict the board)"""
        cons = self.constraints(obs)
        dom = cons.domains(obs.values)
        if not cons.propagate(dom):
            return None
        if not search:
            return dom
        progress = True
        while progress:
            progress = False
            for p in sorted((q for q, d in dom.items() if len(d) > 1), key=lambda q: len(dom[q])):
                for v in sorted(dom[p]):
                    trial = {q: set(d) for q, d in dom.items()}
                    trial[p] = {v}
                    if not cons.propagate(trial):
                        dom[p].discard(v)
                        if not cons.propagate(dom):
                            return None
                        progress = True
                if progress:
                    break
        return dom

    # -------------------------------------------------------------- play
    def choose(self, obs: Observation, moves: List[Pos]) -> Tuple[Pos, int]:
        dom = self.deduce(obs)
        decided = [p for p in moves if dom is not None and len(dom[p]) == 1]
        if not decided and self.search and dom is not None:
            dom = self.deduce(obs, search=True)
            decided = [p for p in moves if dom is not None and len(dom[p]) == 1]
        if decided:
            self.moves_by['deduced'] += 1
            p = min(decided)
            return p, next(iter(dom[p]))
        self.moves_by['guessed'] += 1
        if dom is None:
            n = len(obs.values)
            return self.rng.choice(moves), self.rng.randint(1, self.values or n)
        p = min(moves, key=lambda q: (len(dom[q]), self.rng.random()))
        return p, self.rng.choice(sorted(dom[p]))

    def summary(self) -> Dict:
        def name(h):
            return f'all different in each {h[1]}' if h[0] == 'different' else f"clue '{h[1]}' = {h[2]}"
        return {'games': self.games, 'accepted': {name(h): self.accepted_at[h] for h in self.accepted()},
                'moves': dict(self.moves_by)}
