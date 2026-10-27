#!/usr/bin/env python3
"""
Effect learner: learns what actions do, then plans with what it learned
(action_puzzles.py)

The learner sees grids before and after its own actions. It is told the
goal ("no cell holds a value in goal_absent") but not what the actions do.

Models (hypotheses about an action's effect), simplest first:
- cell model, for actions that name a cell: the cells at a fixed set of
  offsets from the named cell change by a value table per offset; every
  other cell stays as it was. Offsets and tables are learned; a value not
  yet seen at an offset whose entries so far all swap 0 and 1 is taken to
  swap too.
- agent model (k, shared), for other actions: one cell is the agent (the
  cell holding a value from the agent values: a set of values that together
  occupy exactly one cell in the states seen, except in at most
  AGENT_MISS_RATE of them (the agent may leave the view for a while); of
  these, the one whose cell moves most often, less the states it is
  missing from); an action moves along a direction (learned per
  action from how the agent moved); the k cells at the agent and ahead of
  it in that direction change by a table keyed by their current values
  (one table for all directions if shared, else one per action); every
  other cell stays as it was.
A transition that contradicts a cell model's table, or changes a cell the
model says stays, rejects the model. An agent model keeps, for each key,
how often each outcome was seen and predicts the most frequent one; a
transition that differs from it, or changes a cell the model says stays,
is an error, and the model is dropped while its errors exceed
ERROR_RATE of its transitions (so a few exceptions, such as walls that are
not drawn, do not reject it). A transition already seen (the same grid
and action) is predicted as it was seen. An action that has changed nothing in
NOOP_STATES different states (and never changed anything) is predicted to
change nothing.

Planning uses the consistent model with the lowest cost, errors plus
TABLE_COST per table entry (so a model that avoids errors only by keeping
a separate entry for nearly every case is not preferred), the first listed
on a tie:
- if it says every action toggles a fixed set of cells whatever their
  values (a cell model whose tables all swap 0 and 1), effects add up
  modulo 2 and the plan with the fewest actions solves a linear system;
- otherwise breadth-first search over the states the model predicts,
  leaving out transitions it cannot predict; a subclass may also name
  moves that reach the goal directly (wins_with), and the search stops at
  the first state where one can be made.
With no way to the goal, the learner explores: the search also finds the
nearest state with an action whose outcome the model cannot predict, and
the learner goes there and tries it; with no such state, it takes an
action it cannot predict here, or else a random one. A plan is dropped
when an outcome differs from its prediction.
"""

import itertools
import random
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from action_puzzles import solve_xor

OUT = -9            # value of a cell outside the grid in a key
BFS_LIMIT = 100_000
NOOP_STATES = 3
ERROR_RATE = 0.1
TABLE_COST = 0.5
AGENT_MISS_RATE = 0.5


class CellModel:
    name = 'cell model'

    def __init__(self):
        self.offsets: set = set()
        self.tables: Dict[tuple, Dict[int, int]] = {}
        self.consistent = True

    def fit(self, history):
        self.offsets = set()
        for before, action, after in history:
            r, c = action
            self.offsets |= {(y - r, x - c) for y, x in zip(*np.nonzero(before != after))}
        self.tables = {o: {} for o in self.offsets}
        self.consistent = True
        for t in history:
            self.update(*t)

    def update(self, before, action, after) -> bool:
        """Learn one transition; False if it needs offsets the model does not have"""
        r, c = action
        changed = {(y - r, x - c) for y, x in zip(*np.nonzero(before != after))}
        if not changed <= self.offsets:
            return False
        for o in self.offsets:
            p = (r + o[0], c + o[1])
            if 0 <= p[0] < before.shape[0] and 0 <= p[1] < before.shape[1]:
                if self.tables[o].setdefault(int(before[p]), int(after[p])) != int(after[p]):
                    self.consistent = False
        return True

    @staticmethod
    def _swaps(table) -> bool:
        return bool(table) and all(k in (0, 1) and v == 1 - k for k, v in table.items())

    def predict(self, grid, action) -> Optional[np.ndarray]:
        """
        The grid after the action; a value an offset has not been seen with is
        predicted only if that offset's table so far swaps 0 and 1 (the simplest
        rule for it)
        """
        r, c = action
        out = grid.copy()
        for o in self.offsets:
            p = (r + o[0], c + o[1])
            if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]:
                v = self.tables[o].get(int(grid[p]))
                if v is None:
                    if not (self._swaps(self.tables[o]) and grid[p] in (0, 1)):
                        return None
                    v = 1 - int(grid[p])
                out[p] = v
        return out

    def toggles(self) -> bool:
        """Every learned effect swaps 0 and 1 (so effects add up modulo 2)"""
        return bool(self.offsets) and all(self._swaps(t) for t in self.tables.values())


class AgentModel:
    def __init__(self, k: int, shared: bool):
        self.k, self.shared = k, shared
        self.name = f'agent model (k={k}, {"one table" if shared else "a table per action"})'
        self.table: Dict[tuple, Counter] = {}
        self.updates = self.violations = 0

    @property
    def errors(self) -> int:
        return self.violations + sum(sum(c.values()) - max(c.values()) for c in self.table.values())

    @property
    def consistent(self) -> bool:
        return self.errors <= ERROR_RATE * self.updates

    def key(self, grid, agent, d, action):
        cells = [(agent[0] + i * d[0], agent[1] + i * d[1]) for i in range(self.k)]
        vals = tuple(int(grid[p]) if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] else OUT
                     for p in cells)
        return cells, (vals if self.shared else (action,) + vals)

    def update(self, before, after, agent, d, action):
        cells, key = self.key(before, agent, d, action)
        inside = [p for p in cells if 0 <= p[0] < before.shape[0] and 0 <= p[1] < before.shape[1]]
        mask = np.zeros(before.shape, dtype=bool)
        for p in inside:
            mask[p] = True
        self.updates += 1
        if (before != after)[~mask].any():
            self.violations += 1
            return
        new = tuple(int(after[p]) if p in inside else OUT for p in cells)
        self.table.setdefault(key, Counter())[new] += 1

    def predict(self, grid, agent, d, action) -> Optional[np.ndarray]:
        cells, key = self.key(grid, agent, d, action)
        seen = self.table.get(key)
        if seen is None:
            return None
        new = seen.most_common(1)[0][0]
        out = grid.copy()
        for p, v in zip(cells, new):
            if v != OUT:
                out[p] = v
        return out


class EffectLearner:
    def __init__(self, goal_absent, rng: Optional[random.Random] = None):
        self.goal_absent = tuple(goal_absent)
        self.rng = rng or random.Random()
        self.history: List[Tuple[np.ndarray, object, np.ndarray]] = []
        self.cell = CellModel()
        self.agent_models = [AgentModel(k, shared) for shared in (True, False) for k in (1, 2, 3, 4)]
        self.agent_values: Optional[set] = None
        self.directions: Dict[object, Tuple[int, int]] = {}
        self.pending: List[Tuple[np.ndarray, object, np.ndarray]] = []
        self.plan: List = []
        self.exploring = False
        self.moves_by = {'planned': 0, 'explored': 0}
        self.unchanged_in: Dict[object, set] = {}     # action -> states it changed nothing in (None once it changed one)
        self.outcomes: Dict[tuple, np.ndarray] = {}   # (grid shape, grid bytes, action) -> grid after
        self._av = None                               # incremental state of _agent_values

    # ---------------------------------------------------------- learning
    def _agent_values(self) -> Optional[set]:
        """
        The set of values that together occupy exactly one cell in the states
        seen (but for AGENT_MISS_RATE of them); of these, the one whose cell
        moves most often, less the states it is missing from
        """
        # kept incrementally over the states before each transition (the last state after is checked
        # at each call); a value not seen before starts the count again over all states
        last = self.history[-1][2]
        av = self._av
        new = [b for b, _, _ in self.history[av['n']:]] if av else []
        if av is None or any(not set(np.unique(g).tolist()) <= av['values'] for g in new + [last]):
            states = [b for b, _, _ in self.history]
            values = set(np.unique(np.concatenate([g.ravel() for g in states + [last]])).tolist())
            av = self._av = {'n': 0, 'values': values,
                             'subsets': {sub: [None, 0, 0] for size in (1, 2, 3)
                                         for sub in itertools.combinations(sorted(values), size)}}
            new = states
        for g in new:
            self._agent_step(g, av['subsets'])
        av['n'] = len(self.history)
        counts = dict(zip(*np.unique(last, return_counts=True)))
        best, score = None, None
        for sub, (where, n, miss) in av['subsets'].items():
            if sum(counts.get(v, 0) for v in sub) == 1:
                pos = tuple(np.argwhere(np.isin(last, sub))[0])
                n += where is not None and pos != where
            else:
                miss += 1
            if miss <= AGENT_MISS_RATE * (av['n'] + 1) and (score is None or n - miss > score):
                best, score = set(sub), n - miss
        return best

    @staticmethod
    def _agent_step(grid, subsets):
        """Count, for each value set, a state it does not occupy exactly one cell of, or a move"""
        counts = dict(zip(*np.unique(grid, return_counts=True)))
        for sub in list(subsets):
            if sum(counts.get(v, 0) for v in sub) != 1:
                subsets[sub][2] += 1
                continue
            pos = tuple(np.argwhere(np.isin(grid, sub))[0])
            if subsets[sub][0] is not None and pos != subsets[sub][0]:
                subsets[sub][1] += 1
            subsets[sub][0] = pos

    def _agent(self, grid) -> Optional[Tuple[int, int]]:
        if not self.agent_values:
            return None
        cells = list(zip(*np.nonzero(np.isin(grid, list(self.agent_values)))))
        return tuple(int(x) for x in cells[0]) if len(cells) == 1 else None

    def _noop(self, action) -> bool:
        seen = self.unchanged_in.get(action)
        return seen is not None and len(seen) >= NOOP_STATES

    def observe(self, before, action, after):
        self.history.append((before, action, after))
        self.outcomes[(before.shape, before.tobytes(), action)] = after
        if np.array_equal(before, after):
            if self.unchanged_in.get(action, set()) is not None:
                self.unchanged_in.setdefault(action, set()).add(before.tobytes())
        else:
            self.unchanged_in[action] = None
        if isinstance(action, tuple):
            if not self.cell.update(before, action, after):
                self.cell.fit(self.history)
            return
        values = self._agent_values()
        if values != self.agent_values:
            self.agent_values = values
            self._refit_agents()
            return
        a0, a1 = self._agent(before), self._agent(after)
        if a0 and a1 and a0 != a1 and action not in self.directions:
            self.directions[action] = (int(np.sign(a1[0] - a0[0])), int(np.sign(a1[1] - a0[1])))
            self._refit_agents()
            return
        self._learn_agent(before, action, after)

    def _refit_agents(self):
        self.agent_models = [AgentModel(m.k, m.shared) for m in self.agent_models]
        self.directions = {}
        for before, action, after in self.history:
            a0, a1 = self._agent(before), self._agent(after)
            if a0 and a1 and a0 != a1 and action not in self.directions:
                self.directions[action] = (int(np.sign(a1[0] - a0[0])), int(np.sign(a1[1] - a0[1])))
        for t in self.history:
            if not isinstance(t[1], tuple):
                self._learn_agent(*t)

    def _learn_agent(self, before, action, after):
        agent, d = self._agent(before), self.directions.get(action)
        if agent is None or d is None:
            return
        for m in self.agent_models:
            m.update(before, after, agent, d, action)

    def model(self):
        if self.history and isinstance(self.history[0][1], tuple):
            return self.cell if self.cell.consistent and self.cell.offsets else None
        usable = [(m.errors + TABLE_COST * len(m.table), i) for i, m in enumerate(self.agent_models)
                  if m.consistent and m.table]
        return self.agent_models[min(usable)[1]] if usable else None

    def predict(self, grid, action) -> Optional[np.ndarray]:
        seen = self.outcomes.get((grid.shape, grid.tobytes(), action))
        if seen is not None:
            return seen.copy()
        if self._noop(action):
            return grid.copy()
        m = self.model()
        if m is None:
            return None
        if isinstance(m, CellModel):
            return m.predict(grid, action)
        agent, d = self._agent(grid), self.directions.get(action)
        return None if agent is None or d is None else m.predict(grid, agent, d, action)

    # ---------------------------------------------------------- planning
    def _goal(self, grid) -> bool:
        return not np.isin(grid, self.goal_absent).any()

    def wins_with(self, grid, action) -> bool:
        """Whether the action reaches the goal from this grid (other than by the state it leads to)"""
        return False

    def make_plan(self, grid, actions) -> Optional[List]:
        m = self.model()
        if m is None:
            return None
        if isinstance(m, CellModel) and m.toggles():
            zero = np.zeros_like(grid)
            masks = []
            for a in actions:
                after = m.predict(zero, a)
                if after is None:
                    return None
                masks.append(after.ravel())
            x = solve_xor(np.array(masks), np.isin(grid, self.goal_absent).ravel().astype(int))
            self.exploring = False
            return None if x is None else [a for a, bit in zip(actions, x) if bit]
        start = grid.tobytes()
        seen = {start: None}
        queue = deque([grid])
        explore = None           # path to the nearest state with an action the model cannot predict

        def path_to(k):
            path = []
            while seen[k] is not None:
                k, a = seen[k]
                path.append(a)
            return path[::-1]
        while queue and len(seen) < BFS_LIMIT:
            g = queue.popleft()
            if self._goal(g):
                self.exploring = False
                return path_to(g.tobytes())
            for a in actions:
                if self.wins_with(g, a):
                    self.exploring = False
                    return path_to(g.tobytes()) + [a]
                h = self.predict(g, a)
                if h is None:
                    if explore is None:
                        explore = path_to(g.tobytes()) + [a]
                    continue
                k = h.tobytes()
                if k not in seen:
                    seen[k] = (g.tobytes(), a)
                    queue.append(h)
        # no known way to the goal: go and try the nearest action whose outcome is unknown
        self.exploring = explore is not None
        return explore

    # -------------------------------------------------------------- play
    def act(self, grid, actions):
        if not self.plan:
            self.plan = self.make_plan(grid, actions) or []
        if self.plan:
            self.moves_by['explored' if self.exploring else 'planned'] += 1
            return self.plan.pop(0)
        self.moves_by['explored'] += 1
        unknown = [a for a in actions if self.predict(grid, a) is None]
        return self.rng.choice(unknown or actions)

    def after_step(self, before, action, after):
        predicted = self.predict(before, action)
        self.observe(before, action, after)
        if predicted is None or not np.array_equal(predicted, after):
            self.plan = []          # the model was wrong or did not know: plan again

    def new_episode(self):
        self.plan = []
