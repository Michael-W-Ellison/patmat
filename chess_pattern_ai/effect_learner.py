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
  occupy exactly one cell in every state seen, the one that moves most
  often if several do); an action moves along a direction (learned per
  action from how the agent moved); the k cells at the agent and ahead of
  it in that direction change by a table keyed by their current values
  (one table for all directions if shared, else one per action); every
  other cell stays as it was.
A transition that contradicts a table entry, or changes a cell the model
says stays, rejects the model.

Planning uses the first consistent model:
- if it says every action toggles a fixed set of cells whatever their
  values (a cell model whose tables all swap 0 and 1), effects add up
  modulo 2 and the plan with the fewest actions solves a linear system;
- otherwise breadth-first search over the states the model predicts,
  leaving out transitions it cannot predict.
With no way to the goal, the learner explores: the search also finds the
nearest state with an action whose outcome the model cannot predict, and
the learner goes there and tries it; with no such state, it takes an
action it cannot predict here, or else a random one. A plan is dropped
when an outcome differs from its prediction.
"""

import itertools
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from action_puzzles import solve_xor

OUT = -9            # value of a cell outside the grid in a key
BFS_LIMIT = 100_000


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
        self.table: Dict[tuple, tuple] = {}
        self.consistent = True

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
        if (before != after)[~mask].any():
            self.consistent = False
            return
        new = tuple(int(after[p]) if p in inside else OUT for p in cells)
        if self.table.setdefault(key, new) != new:
            self.consistent = False

    def predict(self, grid, agent, d, action) -> Optional[np.ndarray]:
        cells, key = self.key(grid, agent, d, action)
        new = self.table.get(key)
        if new is None:
            return None
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

    # ---------------------------------------------------------- learning
    def _agent_values(self) -> Optional[set]:
        """
        The set of values that together occupy exactly one cell in every state
        seen (among values occurring at most once); if several sets do, the one
        whose cell moves most often
        """
        states = [b for b, _, _ in self.history] + [self.history[-1][2]]
        values = sorted(set(np.unique(np.concatenate([s.ravel() for s in states])).tolist()))
        once = [v for v in values if all((s == v).sum() <= 1 for s in states)]
        best, moves = None, -1
        for size in range(1, min(len(once), 3) + 1):
            for subset in itertools.combinations(once, size):
                if not all(np.isin(s, subset).sum() == 1 for s in states):
                    continue
                where = [tuple(zip(*np.nonzero(np.isin(s, subset))))[0] for s in states]
                n = sum(1 for a, b in zip(where, where[1:]) if a != b)
                if n > moves:
                    best, moves = set(subset), n
        return best

    def _agent(self, grid) -> Optional[Tuple[int, int]]:
        if not self.agent_values:
            return None
        cells = list(zip(*np.nonzero(np.isin(grid, list(self.agent_values)))))
        return tuple(int(x) for x in cells[0]) if len(cells) == 1 else None

    def observe(self, before, action, after):
        self.history.append((before, action, after))
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
            if m.consistent:
                m.update(before, after, agent, d, action)

    def model(self):
        if self.history and isinstance(self.history[0][1], tuple):
            return self.cell if self.cell.consistent and self.cell.offsets else None
        return next((m for m in self.agent_models if m.consistent and m.table), None)

    def predict(self, grid, action) -> Optional[np.ndarray]:
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
