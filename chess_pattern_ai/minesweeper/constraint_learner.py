#!/usr/bin/env python3
"""
Constraint learner for Minesweeper and other constraint puzzles

The learner is not told what the numbers mean. It learns from the view
shown when a game or puzzle attempt is over (every mine / filled cell
shown as MINE) and from what it sees while playing.

1. Discover constraints. Hypotheses come from a generic family:
   - count:  "a clue equals the number of cells with property P in scope S"
             (P mine or not-mine; S the 4 neighbours, 8 neighbours or 5x5
             square around the clue, each with or without the clue cell
             itself, or the clue's row and column)
   - runs:   "the clues lying outside the board along a row or column,
             read in order, equal the lengths of the runs of mine cells in
             that line"
   - region: "every row / every column / every 3x3 box of the board holds
             the same number k of mine cells" (k read from the first
             region seen)
   - global: connectivity of the mine (or not-mine) cells, by sides or
             through cells showing one marker value; no 2x2 block of one
             class; islands (every group of not-mine cells holds one clue,
             equal to its size); degree (every cell showing marker V touches
             a number of mine cells in a set S). See global_constraints.py.
             Each attempt adds as evidence how unlikely the rule would be to
             hold by chance (on boards with the mines placed at random); one
             is accepted at ACCEPT_AFTER_GLOBAL
   Every finished attempt checks every hypothesis wherever it applies; one
   violation rejects a hypothesis, and one confirmed ACCEPT_AFTER times
   without a violation is accepted.

   Two facts about the puzzle are learned the same way rather than assumed:
   the board is the cells ever seen hidden, shown empty or shown as a mine
   (margins never are), and whether a cell showing a clue can itself be a
   mine (in Minesweeper it cannot; in Mosaic it can, so clue cells are then
   unknowns too). Mines shown during play (given cells) count as known.

2. Deduce. Count and region constraints become linear equations "these
   unknown cells hold exactly r mines": an equation with r = 0 makes its
   cells safe, one with r equal to its number of cells makes them mines,
   and for two equations whose cells are nested the difference is deduced
   the same way. Run constraints are solved line by line: every placement
   of the runs that agrees with the cells already known is enumerated, and
   cells that are the same in all of them are fixed. Repeated until
   nothing changes. When that finds no safe cell, search: each undecided
   cell the constraints mention is assumed to be a mine, then safe; an
   assumption that leads to a contradiction proves the opposite, which is
   added and propagated before the next trial.

3. Act. A playable cell proven safe is revealed. Otherwise the learner
   guesses the cell with the lowest mine rate observed for cells with the
   same visible description (minesweeper_headless_trainer.cell_features).
"""

import json
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

import global_constraints

from .minesweeper_game import HIDDEN, MINE, SHOWN_EMPTY

Pos = Tuple[int, int]
_RING8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
_RING24 = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc]
NEIGHBOURHOODS = {
    '4 neighbours': [(-1, 0), (1, 0), (0, -1), (0, 1)],
    '8 neighbours': _RING8,
    '5x5 square': _RING24,
    '4 neighbours and itself': [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    '3x3 block': [(0, 0)] + _RING8,
    '5x5 square and itself': [(0, 0)] + _RING24,
}
LINES = 'its row and column'
PROPERTIES = ('mine', 'not-mine')
REGIONS = ('row', 'column', '3x3 box')
ACCEPT_AFTER = 200
ACCEPT_AFTER_GLOBAL = 10       # global rules (connected, no 2x2, islands, degree): evidence (surprise) needed
GLOBAL_KINDS = ('connected', 'no 2x2', 'islands', 'degree')
SOLVE_BUDGET = 3000            # backtracking steps when global rules are accepted and trials leave cells open

Hypothesis = Tuple[str, str, str]


def _cells(pos: Pos, offsets, rows: int, cols: int) -> List[Pos]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in offsets if 0 <= r + dr < rows and 0 <= c + dc < cols]


def _scope(pos: Pos, scope: str, rows: int, cols: int) -> List[Pos]:
    if scope == LINES:
        r, c = pos
        return [(r, x) for x in range(cols) if x != c] + [(y, c) for y in range(rows) if y != r]
    return _cells(pos, NEIGHBOURHOODS[scope], rows, cols)


def _runs(states: List[bool]) -> List[int]:
    runs, n = [], 0
    for v in states + [False]:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return runs or [0]


def _placements(runs: List[int], known: List[Optional[bool]]) -> List[List[bool]]:
    """Every line agreeing with `known` (None = unknown) whose runs of True cells are `runs`"""
    n = len(known)
    runs = [r for r in runs if r > 0]
    out = []

    def place(i, start, line):
        if i == len(runs):
            full = line + [False] * (n - len(line))
            if all(k is None or k == v for k, v in zip(known, full)):
                out.append(full)
            return
        rest = sum(runs[i + 1:]) + len(runs) - i - 1
        for s in range(start, n - runs[i] - rest + 1):
            cand = line + [False] * (s - len(line)) + [True] * runs[i]
            if i + 1 < len(runs):
                cand = cand + [False]
            if all(k is None or k == v for k, v in zip(known, cand)):
                place(i + 1, len(cand), cand)
    place(0, 0, [])
    return out


class ConstraintLearner:
    def __init__(self, rng: Optional[random.Random] = None, accept_after: int = ACCEPT_AFTER, search: bool = True):
        self.rng = rng or random.Random()
        self.accept_after = accept_after
        self.search = search
        self.hypotheses: Dict[Hypothesis, Dict] = {}
        for scope in list(NEIGHBOURHOODS) + [LINES]:
            for prop in PROPERTIES:
                self.hypotheses[('count', scope, prop)] = {'confirmed': 0, 'violated': 0}
        self.hypotheses[('runs', 'line', 'mine')] = {'confirmed': 0, 'violated': 0}
        for kind in REGIONS:
            self.hypotheses[('region', kind, 'mine')] = {'confirmed': 0, 'violated': 0, 'k': None}
        for rule in global_constraints.rules([]):
            self.hypotheses[rule] = {'confirmed': 0, 'violated': 0}
        self.marker_values: Set[int] = set()
        self.accepted_at: Dict[Hypothesis, int] = {}
        self.clues_can_be_mines = False
        self.mine_rate: Dict[str, List[int]] = {}       # description -> [mines, cells]
        self.games = 0
        self._seen: List[Tuple[Pos, str]] = []          # hidden cells described during the current game
        self._clues: Dict[Pos, int] = {}                # clue numbers seen during the current game
        self._board: Set[Pos] = set()                   # cells seen hidden during the current game
        self._markers: Dict[Pos, int] = {}              # cells shown with a marker value (outside the board)
        self.moves_by = {'deduced': 0, 'guessed': 0}

    # ---------------------------------------------------------- learning
    def accepted(self) -> List[Hypothesis]:
        def needed(h):
            return min(self.accept_after, ACCEPT_AFTER_GLOBAL) if h[0] in GLOBAL_KINDS else self.accept_after
        return [h for h, s in self.hypotheses.items() if s['violated'] == 0 and s['confirmed'] >= needed(h)]

    def _observe(self, obs):
        for r, row in enumerate(obs):
            for c, v in enumerate(row):
                if v in (HIDDEN, SHOWN_EMPTY, MINE):
                    self._board.add((r, c))
                elif v < 0:
                    self._markers[(r, c)] = v
                elif v >= 0:
                    self._clues[(r, c)] = v

    @staticmethod
    def _lines(rows: int, cols: int, board: Set[Pos], clues: Dict[Pos, int]):
        """(board cells of the line in order, clues outside the board along it in order) for lines with both"""
        lines = []
        for r in range(rows):
            cells = [(r, c) for c in range(cols)]
            on = [p for p in cells if p in board]
            off = [clues[p] for p in cells if p in clues and p not in board]
            if on and off:
                lines.append((on, off))
        for c in range(cols):
            cells = [(r, c) for r in range(rows)]
            on = [p for p in cells if p in board]
            off = [clues[p] for p in cells if p in clues and p not in board]
            if on and off:
                lines.append((on, off))
        return lines

    @staticmethod
    def _regions(kind: str, board: Set[Pos]) -> List[List[Pos]]:
        if not board:
            return []
        r0, c0 = min(r for r, _ in board), min(c for _, c in board)
        if kind == 'row':
            key = lambda p: p[0]
        elif kind == 'column':
            key = lambda p: p[1]
        else:
            h = max(r for r, _ in board) - r0 + 1
            w = max(c for _, c in board) - c0 + 1
            if h % 3 or w % 3:
                return []
            key = lambda p: ((p[0] - r0) // 3, (p[1] - c0) // 3)
        groups: Dict = {}
        for p in board:
            groups.setdefault(key(p), []).append(p)
        return list(groups.values())

    def end_game(self, final_view: List[List[int]]):
        """Learn from the view shown after the attempt (mines / filled cells as MINE)"""
        self.games += 1
        rows, cols = len(final_view), len(final_view[0])
        mines = {(r, c) for r in range(rows) for c in range(cols) if final_view[r][c] == MINE}
        clues = dict(self._clues)
        clues.update({(r, c): final_view[r][c] for r in range(rows) for c in range(cols) if final_view[r][c] >= 0})
        board = self._board or {(r, c) for r in range(rows) for c in range(cols)}
        if any(p in mines for p in clues):
            self.clues_can_be_mines = True
        markers = {p: v for p, v in self._markers.items() if p not in board}
        for v in set(markers.values()) - self.marker_values:
            for rule in global_constraints.rules([v]):
                self.hypotheses.setdefault(rule, {'confirmed': 0, 'violated': 0})
        self.marker_values |= set(markers.values())
        for (kind, what, prop), stats in self.hypotheses.items():
            if kind in GLOBAL_KINDS:
                ok, surprise = global_constraints.evidence((kind, what, prop), mines & board, board - mines,
                                                           {p: v for p, v in clues.items() if p not in board},
                                                           markers, self.rng)
                if not ok:
                    stats['violated'] += 1
                else:
                    stats['confirmed'] += surprise
            elif kind == 'count':
                for pos, v in clues.items():
                    near = _scope(pos, what, rows, cols)
                    m = sum(1 for p in near if p in mines)
                    stats['confirmed' if (m if prop == 'mine' else len(near) - m) == v else 'violated'] += 1
            elif kind == 'runs':
                for on, off in self._lines(rows, cols, board, clues):
                    stats['confirmed' if _runs([p in mines for p in on]) == off else 'violated'] += 1
            else:
                for region in self._regions(what, board):
                    m = sum(1 for p in region if p in mines)
                    if stats['k'] is None:
                        stats['k'] = m
                    stats['confirmed' if m == stats['k'] else 'violated'] += 1
        for h in self.accepted():
            self.accepted_at.setdefault(h, self.games)
        for pos, desc in self._seen:
            counts = self.mine_rate.setdefault(desc, [0, 0])
            counts[0] += pos in mines
            counts[1] += 1
        self._seen, self._clues, self._board, self._markers = [], {}, set(), {}

    # --------------------------------------------------------- deduction
    def _constraints(self, obs):
        """Linear equations (unknown cells, mines among them) and run lines (cells, runs)"""
        rows, cols = len(obs), len(obs[0])
        self._observe(obs)
        clues = {(r, c): obs[r][c] for r in range(rows) for c in range(cols) if obs[r][c] >= 0}
        unknown = {(r, c) for r in range(rows) for c in range(cols) if obs[r][c] == HIDDEN}
        known_mines = {(r, c) for r in range(rows) for c in range(cols) if obs[r][c] == MINE}
        if self.clues_can_be_mines:
            unknown |= set(clues)
        eqs, lines = [], []
        for kind, what, prop in self.accepted():
            if kind == 'count':
                for pos, v in clues.items():
                    near = _scope(pos, what, rows, cols)
                    m = (v if prop == 'mine' else len(near) - v) - sum(1 for p in near if p in known_mines)
                    eqs.append(({p for p in near if p in unknown}, m))
            elif kind == 'runs':
                lines += [(on, off) for on, off in self._lines(rows, cols, self._board, clues)]
            elif kind == 'region':
                k = self.hypotheses[(kind, what, prop)]['k']
                eqs += [({p for p in region if p in unknown}, k - sum(1 for p in region if p in known_mines))
                        for region in self._regions(what, self._board)]
        rules = [h for h in self.accepted() if h[0] in GLOBAL_KINDS]
        glob = None
        if rules:
            markers = {p: v for p, v in self._markers.items() if p not in self._board}
            glob = (rules, set(self._board), {p: v for p, v in clues.items() if p not in self._board}, markers)
        return eqs, lines, unknown, known_mines, glob

    @staticmethod
    def _global_ok(glob, unknown, known_mines, safe, mines) -> bool:
        """The accepted global rules can still hold given what is known"""
        rules, board, clues, markers = glob
        open_cells = (unknown & board) - safe - mines
        filled = (known_mines | mines) & board
        empty = board - open_cells - filled
        return all(global_constraints.consistent(r, filled, empty, open_cells, clues, markers) for r in rules)

    @staticmethod
    def _propagate(eqs, lines, unknown, known_mines, safe: Set[Pos], mines: Set[Pos], glob=None):
        """
        Everything the constraints force, starting from the given safe and
        mine cells: (safe, mines, consistent); consistent is False when the
        constraints contradict each other or the starting cells (global rules
        are checked once the local ones force nothing more)
        """
        safe, mines = set(safe), set(mines)
        changed = True
        while changed:
            changed = False
            reduced = set()
            for cells, m in eqs:
                rest = cells - mines - safe
                r = m - len(cells & mines)
                if r < 0 or r > len(rest):
                    return safe, mines, False
                if rest:
                    reduced.add((frozenset(rest), r))
            found_safe, found_mines = set(), set()
            for rest, r in reduced:
                if r == 0:
                    found_safe |= rest
                elif r == len(rest):
                    found_mines |= rest
            for cells, runs in lines:
                known = [True if p in mines or p in known_mines else False if (p in safe or p not in unknown)
                         else None for p in cells]
                options = _placements(runs, known)
                if not options:
                    return safe, mines, False
                for i, p in enumerate(cells):
                    if known[i] is None and all(o[i] == options[0][i] for o in options):
                        (found_mines if options[0][i] else found_safe).add(p)
            if not found_safe and not found_mines:
                for u1, r1 in reduced:
                    for u2, r2 in reduced:
                        if u1 < u2:
                            diff, rd = u2 - u1, r2 - r1
                            if rd < 0 or rd > len(diff):
                                return safe, mines, False
                            if rd == 0:
                                found_safe |= diff
                            elif rd == len(diff):
                                found_mines |= diff
            found_safe -= safe
            found_mines -= mines
            if found_safe & found_mines:
                return safe, mines, False
            if found_safe or found_mines:
                safe |= found_safe
                mines |= found_mines
                changed = True
        if glob is not None and not ConstraintLearner._global_ok(glob, unknown, known_mines, safe, mines):
            return safe, mines, False
        return safe, mines, True

    def deduce(self, obs, search: bool = False) -> Tuple[Set[Pos], Set[Pos]]:
        """
        (cells proven safe, cells proven mines) under the accepted constraints.
        With search, every undecided cell the constraints mention is also
        tried both ways: if assuming it is a mine (or safe) leads to a
        contradiction, it is the other; repeated until nothing changes. With
        global rules accepted (whose checks on a partly known board are only
        necessary conditions), cells still open are settled by enumerating
        the solutions the rules allow: if there is exactly one, every cell is
        decided.
        """
        eqs, lines, unknown, known_mines, glob = self._constraints(obs)
        safe, mines, consistent = self._propagate(eqs, lines, unknown, known_mines, set(), set(), glob)
        if not search or not consistent:
            return safe, mines
        mentioned = set().union(*(cells for cells, _ in eqs), *(set(c) for c, _ in lines)) & unknown
        if glob is not None:
            mentioned = unknown & glob[1]          # global rules concern every board cell
        progress = True
        while progress:
            progress = False
            for p in sorted(mentioned - safe - mines):
                if p in safe or p in mines:
                    continue
                for assume_mine in (True, False):
                    trial_safe = safe if assume_mine else safe | {p}
                    trial_mines = mines | {p} if assume_mine else mines
                    if not self._propagate(eqs, lines, unknown, known_mines, trial_safe, trial_mines, glob)[2]:
                        # the assumption is impossible, so the opposite holds
                        safe, mines, consistent = self._propagate(
                            eqs, lines, unknown, known_mines,
                            safe | {p} if assume_mine else safe, mines if assume_mine else mines | {p}, glob)
                        if not consistent:
                            return safe, mines
                        progress = True
                        break
        if glob is not None:
            # global rules: if trials leave cells open, enumerate the solutions the rules allow
            open_cells = (unknown & glob[1]) - safe - mines
            if open_cells:
                solutions, complete = self._enumerate(eqs, lines, unknown, known_mines, glob, safe, mines)
                if complete and len(solutions) == 1:
                    safe |= open_cells - solutions[0]
                    mines |= open_cells & solutions[0]
        return safe, mines

    def _enumerate(self, eqs, lines, unknown, known_mines, glob, safe, mines, limit=2):
        """
        Solutions (mine sets) the accepted constraints allow, up to `limit`, by
        backtracking with propagation; (solutions, complete), complete False if
        SOLVE_BUDGET steps ran out first
        """
        board = glob[1]
        found, steps = [], [0]

        def rec(safe, mines):
            if len(found) >= limit or steps[0] > SOLVE_BUDGET:
                return
            steps[0] += 1
            safe, mines, ok = self._propagate(eqs, lines, unknown, known_mines, safe, mines, glob)
            if not ok:
                return
            open_cells = (unknown & board) - safe - mines
            if not open_cells:
                found.append(mines & board)
                return
            decided = safe | mines | (board - unknown)
            p = max(sorted(open_cells), key=lambda q: sum((q[0] + dr, q[1] + dc) in decided
                                                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)))
            rec(safe, mines | {p})
            rec(safe | {p}, mines)
        rec(set(safe), set(mines))
        return found, steps[0] <= SOLVE_BUDGET

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
        if self.search and not any(m in safe for m in moves):
            safe, mines = self.deduce(obs, search=True)     # only when it would otherwise guess
        safe_moves = [m for m in moves if m in safe]
        if safe_moves:
            self.moves_by['deduced'] += 1
            return min(safe_moves)
        self.moves_by['guessed'] += 1
        candidates = [(m, d) for m, d in described if m not in mines] or described
        best = min(self._rate(d) for _, d in candidates)
        return self.rng.choice([m for m, d in candidates if self._rate(d) == best])

    # ------------------------------------------------------------- state
    def describe(self, h: Hypothesis) -> str:
        kind, what, prop = h
        if kind == 'connected':
            return f'{what} cells connected ({"by sides" if prop == "sides" else "through cells showing " + prop[4:]})'
        if kind == 'no 2x2':
            return f'no 2x2 block of {what} cells'
        if kind == 'islands':
            return 'every group of not-mine cells holds one clue, equal to its size'
        if kind == 'degree':
            return f'every cell showing {what} touches {sorted(prop)} mine cells'
        if kind == 'count':
            return f'number = {prop} cells in {what if what == LINES else "its " + what}'
        if kind == 'runs':
            return 'clues outside the board along a line = runs of mine cells in it'
        return f'every {what} holds {self.hypotheses[h]["k"]} mine cells'

    def summary(self) -> Dict:
        return {
            'games': self.games,
            'accepted': {self.describe(h): self.accepted_at[h] for h in self.accepted()},
            'clues_can_be_mines': self.clues_can_be_mines,
            'hypotheses': {self.describe(h): {k: v for k, v in s.items() if k != 'k'}
                           for h, s in self.hypotheses.items()},
            'moves': dict(self.moves_by),
        }

    @staticmethod
    def _plain(h):
        return [sorted(x) if isinstance(x, frozenset) else x for x in h]

    @staticmethod
    def _key(x):
        return tuple(frozenset(v) if isinstance(v, list) else v for v in x[:3])

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'hypotheses': [[*self._plain(h), s] for h, s in self.hypotheses.items()],
                       'accepted_at': [[*self._plain(h), g] for h, g in self.accepted_at.items()],
                       'clues_can_be_mines': self.clues_can_be_mines,
                       'mine_rate': self.mine_rate, 'games': self.games}, f)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.hypotheses = {self._key(x): x[3] for x in d['hypotheses']}
        self.accepted_at = {self._key(x): x[3] for x in d['accepted_at']}
        self.clues_can_be_mines = d.get('clues_can_be_mines', False)
        self.mine_rate = {k: list(v) for k, v in d['mine_rate'].items()}
        self.games = d['games']
