#!/usr/bin/env python3
"""
Local constraint learner for Minesweeper

The learner is not told what the numbers mean. It learns from the view the
game shows when it is over (every mine shown) and from what it sees while
playing:

1. Discover constraints. Hypotheses are a small generic family of local
   count relations: "a revealed number equals the count of cells with
   property P in neighbourhood K", P in {mine, not-mine}, K in {4
   neighbours, 8 neighbours, the 5x5 square around the cell}. Every
   finished game checks every hypothesis against every revealed number;
   one violation rejects a hypothesis, and one confirmed ACCEPT_AFTER times
   without a violation is accepted.

2. Deduce. Each accepted constraint turns every revealed number into a
   linear equation "these hidden cells hold exactly r mines". Equations are
   reduced with the mines already deduced; an equation with r = 0 makes its
   cells safe, one with r equal to its number of cells makes them mines,
   and for two equations whose cells are nested, the difference is
   deduced the same way. Repeated until nothing changes.

3. Act. A cell proven safe is revealed. Otherwise the learner guesses the
   cell with the lowest mine rate observed for cells with the same visible
   description (the description of minesweeper_headless_trainer.cell_features),
   counted from the end-of-game views.
"""

import json
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .minesweeper_game import HIDDEN, MINE

Pos = Tuple[int, int]
NEIGHBOURHOODS = {
    '4 neighbours': [(-1, 0), (1, 0), (0, -1), (0, 1)],
    '8 neighbours': [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc],
    '5x5 square': [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc],
}
PROPERTIES = ('mine', 'not-mine')
ACCEPT_AFTER = 200


def _cells(pos: Pos, offsets, rows: int, cols: int) -> List[Pos]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in offsets if 0 <= r + dr < rows and 0 <= c + dc < cols]


class ConstraintLearner:
    def __init__(self, rng: Optional[random.Random] = None, accept_after: int = ACCEPT_AFTER):
        self.rng = rng or random.Random()
        self.accept_after = accept_after
        # hypothesis -> {'confirmed': n, 'violated': n}
        self.hypotheses = {(p, k): {'confirmed': 0, 'violated': 0} for p in PROPERTIES for k in NEIGHBOURHOODS}
        self.accepted_at: Dict[Tuple[str, str], int] = {}
        self.mine_rate: Dict[str, List[int]] = {}       # description -> [mines, cells]
        self.games = 0
        self._seen: List[Tuple[Pos, str]] = []          # hidden cells described during the current game
        self.moves_by = {'deduced': 0, 'guessed': 0}

    # ---------------------------------------------------------- learning
    def accepted(self) -> List[Tuple[str, str]]:
        return [h for h, s in self.hypotheses.items() if s['violated'] == 0 and s['confirmed'] >= self.accept_after]

    def end_game(self, final_view: List[List[int]]):
        """Learn from the view shown after the game (mines as MINE)"""
        self.games += 1
        rows, cols = len(final_view), len(final_view[0])
        for r in range(rows):
            for c in range(cols):
                v = final_view[r][c]
                if v < 0:
                    continue
                for (prop, k), stats in self.hypotheses.items():
                    near = _cells((r, c), NEIGHBOURHOODS[k], rows, cols)
                    mines = sum(1 for y, x in near if final_view[y][x] == MINE)
                    count = mines if prop == 'mine' else len(near) - mines
                    stats['confirmed' if count == v else 'violated'] += 1
        for h in self.accepted():
            self.accepted_at.setdefault(h, self.games)
        for pos, desc in self._seen:
            counts = self.mine_rate.setdefault(desc, [0, 0])
            counts[0] += final_view[pos[0]][pos[1]] == MINE
            counts[1] += 1
        self._seen = []

    # --------------------------------------------------------- deduction
    def _equations(self, obs) -> List[Tuple[Set[Pos], int]]:
        rows, cols = len(obs), len(obs[0])
        eqs = []
        for prop, k in self.accepted():
            for r in range(rows):
                for c in range(cols):
                    v = obs[r][c]
                    if v == HIDDEN:
                        continue
                    near = _cells((r, c), NEIGHBOURHOODS[k], rows, cols)
                    mines = v if prop == 'mine' else len(near) - v
                    hidden = {p for p in near if obs[p[0]][p[1]] == HIDDEN}
                    eqs.append((hidden, mines))
        return eqs

    def deduce(self, obs) -> Tuple[Set[Pos], Set[Pos]]:
        """(cells proven safe, cells proven mines) under the accepted constraints"""
        mines: Set[Pos] = set()
        safe: Set[Pos] = set()
        eqs = self._equations(obs)
        changed = True
        while changed:
            changed = False
            reduced = []
            for cells, m in eqs:
                unknown = cells - mines - safe
                reduced.append((frozenset(unknown), m - len(cells & mines)))
            reduced = [e for e in set(reduced) if e[0]]
            found_safe, found_mines = set(), set()
            for unknown, r in reduced:
                if r == 0:
                    found_safe |= unknown
                elif r == len(unknown):
                    found_mines |= unknown
            if not found_safe and not found_mines:
                for u1, r1 in reduced:
                    for u2, r2 in reduced:
                        if u1 < u2:
                            diff, rd = u2 - u1, r2 - r1
                            if rd == 0:
                                found_safe |= diff
                            elif rd == len(diff):
                                found_mines |= diff
            found_safe -= safe
            found_mines -= mines
            if found_safe & found_mines:
                break               # contradiction: the constraints do not hold here
            if found_safe or found_mines:
                safe |= found_safe
                mines |= found_mines
                changed = True
        return safe, mines

    # -------------------------------------------------------------- play
    def _rate(self, desc: str) -> float:
        m, n = self.mine_rate.get(desc, (0, 0))
        return (m + 1) / (n + 2)

    def choose(self, obs, moves: Iterable[Pos]) -> Pos:
        from .minesweeper_headless_trainer import cell_features
        moves = list(moves)
        described = [(m, cell_features(obs, m)) for m in moves]
        self._seen.extend(described)
        safe, mines = self.deduce(obs)
        safe_moves = [m for m in moves if m in safe]
        if safe_moves:
            self.moves_by['deduced'] += 1
            return min(safe_moves)
        self.moves_by['guessed'] += 1
        candidates = [(m, d) for m, d in described if m not in mines] or described
        best = min(self._rate(d) for _, d in candidates)
        return self.rng.choice([m for m, d in candidates if self._rate(d) == best])

    # ------------------------------------------------------------- state
    def summary(self) -> Dict:
        return {
            'games': self.games,
            'accepted': {f'number = {p} cells among {k}': self.accepted_at[(p, k)] for p, k in self.accepted()},
            'hypotheses': {f'number = {p} cells among {k}': s for (p, k), s in self.hypotheses.items()},
            'moves': dict(self.moves_by),
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'hypotheses': [[p, k, s] for (p, k), s in self.hypotheses.items()],
                       'accepted_at': [[p, k, g] for (p, k), g in self.accepted_at.items()],
                       'mine_rate': self.mine_rate, 'games': self.games}, f)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.hypotheses = {(p, k): s for p, k, s in d['hypotheses']}
        self.accepted_at = {(p, k): g for p, k, g in d['accepted_at']}
        self.mine_rate = {k: list(v) for k, v in d['mine_rate'].items()}
        self.games = d['games']
