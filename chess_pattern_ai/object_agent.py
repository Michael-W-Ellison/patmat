#!/usr/bin/env python3
"""
Object-level agent for ARC-AGI-3 style games (grids from arc3_agent.GridReader)

The cell-level learner (effect_learner.AgentModel) finds the agent as the
cell holding a value that occurs once. Here the agent is a set of cells
that moves together, which may grow (absorb other objects) and hold several
values. What is learned:

- the background value and the agent: from the first transition in which
  a group of cells moved by one step in a direction, leaving background
  behind; the agent's pattern at a level start is remembered and found
  again in later levels
- each action's direction (the agent's displacement when it moved)
- contact rules: when the agent moves, the cells it would enter (its front)
  are background, or belong to objects. An object is a 4-connected group of
  non-background cells (excluding the agent and cells of wall values),
  except that touching copies of a known one-cell object stay separate; its
  key is its pattern (values over its bounding box). What happens on
  contact is learned per key, and per value of the cells entered (the
  fallback for a key not seen, when all of an object's values agree):
    block   nothing moves
    merge   the object joins the agent, which then moves (or, if its new
            front is not free, stays where it is)
    push    the object and the agent move by one step if the object's
            front is free; if the object hits another, what happens is
            learned per pair of keys: nothing moves, or both vanish (the
            agent stays)
  A value whose contacts only ever block is a wall value. When two outcomes
  give the same grid (a merged agent that moved, or a pushed object), the
  agent's next move tells which: whether the object moves with it.
- the goal, from level completions: the winning move either touched an
  object (contact goal, by key), or produced an agent whose pattern equals
  another object's pattern (shape goal). If the outcome of the winning move
  was not known, the outcomes that would make it a shape match are tried
  and the one that does is learned.
- after a transition the rules do not explain, the agent is found again
  as the one object of its pattern, if there is one; while it is not
  found, transitions seen from the same grid are still predicted
- a background cell the agent could not enter is a hidden wall for the
  rest of the level; transitions seen are remembered exactly, actions that never change anything are no-ops, and
  moves that lost the game are not planned again.

Planning: breadth-first search over (grid, agent cells) states predicted
by the rules, up to BFS_LIMIT states. Under the rules learned an agent
only grows, so once the shape goal is known, a state whose agent no longer
fits inside any object of the starting grid (same values at the same
relative places) is not searched further. The search it stops at the first move that
reaches a goal, else goes to the nearest move whose outcome is unknown or
would give the agent a pattern not seen before (with no goal known yet,
the search stops there).
"""

import random
from collections import Counter, deque
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

Pos = Tuple[int, int]
SIDES = ((-1, 0), (1, 0), (0, -1), (0, 1))
BFS_LIMIT = 30_000
NOOP_STATES = 3
OUTCOMES = ('block', 'merge', 'push')


def _inside(grid, p) -> bool:
    return 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]


def pattern(grid, cells) -> tuple:
    """Values of the cells over their bounding box (-1 off the cells)"""
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    r0, c0 = min(rows), min(cols)
    out = [[-1] * (max(cols) - c0 + 1) for _ in range(max(rows) - r0 + 1)]
    for p in cells:
        out[p[0] - r0][p[1] - c0] = int(grid[p])
    return tuple(map(tuple, out))


def shifted(cells, d) -> FrozenSet[Pos]:
    return frozenset((p[0] + d[0], p[1] + d[1]) for p in cells)


class ObjectAgent:
    learns_from_losses = True

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.bg: Optional[int] = None
        self.agent: Optional[FrozenSet[Pos]] = None
        self.level_fresh = True                   # no agent found yet in this level
        self.start_patterns: set = set()
        self.directions: Dict[object, Tuple[int, int]] = {}
        self.rules: Dict[tuple, Counter] = {}       # object key -> outcomes seen
        self.value_rules: Dict[int, Counter] = {}   # value -> outcomes of objects holding it
        self.pair_rules: Dict[tuple, Counter] = {}  # (pushed key, key of the object it hit) -> block / vanish
        self.outcomes: Dict[tuple, tuple] = {}      # (state key, action) -> (grid after, agent after)
        self.outcomes_by_grid: Dict[tuple, tuple] = {}  # (grid key, action) -> the same, used while the agent is lost
        self.unchanged_in: Dict[object, Optional[set]] = {}
        self.losing: set = set()                    # (state key, action)
        self.win_contacts: set = set()
        self.shape_goal = False
        self.pending = None                         # a contact whose outcome the next move will tell
        self.hidden_walls: set = set()              # cells of this level that block though drawn as background
        self.seen_patterns: set = set()             # patterns the agent has had
        self.plan: List = []
        self.wins = self.losses = 0
        self.moves_by = {'planned': 0, 'explored': 0}
        self.exploring = False
        self._before_agent = None

    # ------------------------------------------------------------ objects
    def _walls(self) -> set:
        return {v for v, c in self.value_rules.items() if set(c) == {'block'}}

    def _object(self, grid, start: Pos, agent) -> FrozenSet[Pos]:
        walls = self._walls()
        if int(grid[start]) in walls:
            return frozenset([start])
        seen, queue = {start}, [start]
        while queue:
            p = queue.pop()
            for dr, dc in SIDES:
                q = (p[0] + dr, p[1] + dc)
                if (q not in seen and _inside(grid, q) and q not in agent and int(grid[q]) != self.bg
                        and int(grid[q]) not in walls):
                    seen.add(q)
                    queue.append(q)
        obj = frozenset(seen)
        # touching copies of a known one-cell object are separate objects
        if len(obj) > 1 and pattern(grid, obj) not in self.rules and \
                all(((int(grid[p]),),) in self.rules for p in obj):
            return frozenset([start])
        return obj

    def _contacts(self, grid, agent, d, exclude=frozenset()) -> Optional[List[FrozenSet[Pos]]]:
        """Objects in the front of `agent` (a set of cells moving by d); None if the front leaves the grid"""
        front = shifted(agent, d) - agent
        if any(not _inside(grid, p) for p in front):
            return None
        objs = []
        for p in sorted(front):
            if int(grid[p]) != self.bg and p not in exclude and not any(p in o for o in objs):
                objs.append(self._object(grid, p, agent | exclude))
        return objs

    def _rule(self, grid, obj) -> Optional[str]:
        seen = self.rules.get(pattern(grid, obj))
        if seen:
            return seen.most_common(1)[0][0]
        kinds = {self.value_rules[v].most_common(1)[0][0] if v in self.value_rules else None
                 for v in {int(grid[p]) for p in obj}}
        return kinds.pop() if len(kinds) == 1 else None

    def _components(self, grid, agent) -> List[FrozenSet[Pos]]:
        walls, seen, out = self._walls(), set(agent), []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if (r, c) not in seen and int(grid[r, c]) != self.bg and int(grid[r, c]) not in walls:
                    obj = self._object(grid, (r, c), agent)
                    seen |= obj
                    out.append(obj)
        return out

    def _shape_match(self, grid, agent) -> bool:
        pat = pattern(grid, agent)
        return any(pattern(grid, o) == pat for o in self._components(grid, agent))

    # --------------------------------------------------------- prediction
    def _move(self, grid, cells, d) -> Optional[np.ndarray]:
        new = shifted(cells, d)
        if any(not _inside(grid, p) for p in new):
            return None
        out = grid.copy()
        for p in cells:
            out[p] = self.bg
        for p in cells:
            out[p[0] + d[0], p[1] + d[1]] = grid[p]
        return out

    def _free(self, grid, cells, d) -> bool:
        return all(_inside(grid, p) and int(grid[p]) == self.bg for p in shifted(cells, d) - cells)

    def _apply(self, grid, agent, d, kind, objs):
        """(grid, agent) after a move whose contacts all have outcome `kind`"""
        if kind == 'block':
            return grid, agent
        if kind == 'merge':
            whole = agent.union(*objs)
            if self._free(grid, whole, d):
                return self._move(grid, whole, d), shifted(whole, d)
            return grid, whole
        if kind == 'push':
            moved = frozenset().union(*objs)
            if self._free(grid, moved, d):
                return self._move(grid, agent | moved, d), shifted(agent, d)
            hit = self._contacts(grid, moved, d, exclude=agent)
            if hit is None or not hit or any(int(grid[p]) in self._walls() for o in hit for p in o):
                return grid, agent
            seen = [self.pair_rules.get((pattern(grid, moved), pattern(grid, o))) for o in hit]
            if any(c is None for c in seen):
                return None
            if all(c.most_common(1)[0][0] == 'vanish' for c in seen):
                return self._vanish(grid, moved, hit), agent
            return grid, agent
        raise ValueError(kind)

    def _vanish(self, grid, moved, hit):
        out = grid.copy()
        for p in moved.union(*hit):
            out[p] = self.bg
        return out

    @staticmethod
    def _key(grid, agent) -> tuple:
        return grid.shape, grid.tobytes(), tuple(sorted(agent)) if agent else None

    def predict(self, grid, agent, action):
        seen = self.outcomes.get((self._key(grid, agent), action))
        if seen is None and agent is None:
            seen = self.outcomes_by_grid.get((grid.shape, grid.tobytes(), action))
        if seen is not None:
            return seen
        u = self.unchanged_in.get(action)
        if u is not None and len(u) >= NOOP_STATES:
            return grid, agent
        d = self.directions.get(action)
        if agent is None or d is None or self.bg is None:
            return None
        objs = self._contacts(grid, agent, d)
        if objs is None or (shifted(agent, d) - agent) & self.hidden_walls:
            return grid, agent
        if not objs:
            return self._move(grid, agent, d), shifted(agent, d)
        kinds = [self._rule(grid, o) for o in objs]
        if None in kinds:
            return None
        if 'block' in kinds:
            return grid, agent
        if len(set(kinds)) > 1:
            return None
        return self._apply(grid, agent, d, kinds[0], objs)       # (None if a pushed object's collision is unknown)

    # ----------------------------------------------------------- learning
    def _identify(self, before, after):
        """(agent cells before, direction) from a transition in which one group of cells moved one step"""
        changed = [(int(r), int(c)) for r, c in np.argwhere(before != after)]
        if not changed:
            return None
        bgs = [self.bg] if self.bg is not None else sorted({int(after[p]) for p in changed})
        for bg in bgs:
            for d in SIDES:
                moved = frozenset((p[0] - d[0], p[1] - d[1]) for p in changed if int(after[p]) != bg)
                if not moved or any(not _inside(before, p) or int(before[p]) == bg for p in moved):
                    continue
                old, self.bg = self.bg, bg
                ok = np.array_equal(self._move(before, moved, d) if self._free(before, moved, d) else before, after)
                self.bg = old
                if ok:
                    return bg, moved, d
        return None

    def _learn_collision(self, before, agent, d, objs, after) -> bool:
        """A pushed object that hit another: did they both vanish, or did nothing move?"""
        moved = frozenset().union(*objs)
        hit = self._contacts(before, moved, d, exclude=agent)
        if not hit:
            return False
        for kind, g in (('vanish', self._vanish(before, moved, hit)), ('block', before)):
            if np.array_equal(g, after):
                for o in hit:
                    self.pair_rules.setdefault((pattern(before, moved), pattern(before, o)), Counter())[kind] += 1
                for o in objs:
                    self._learn_rule(before, o, 'push', shifted(agent, d))
                return True
        return False

    def _learn_rule(self, grid, obj, kind, front):
        """Record the outcome for the object's key, and for the values of the cells entered"""
        self.rules.setdefault(pattern(grid, obj), Counter())[kind] += 1
        for v in {int(grid[p]) for p in obj & front}:
            self.value_rules.setdefault(v, Counter())[kind] += 1

    def observe(self, before, action, after):
        state = self._key(before, self.agent)
        if before.shape != after.shape:             # the view changed size: nothing to learn from
            self.agent = None
            return
        if np.array_equal(before, after):
            if self.unchanged_in.get(action, set()) is not None:
                self.unchanged_in.setdefault(action, set()).add(state)
        else:
            self.unchanged_in[action] = None
        agent_after = self._follow(before, action, after)
        self.outcomes[(state, action)] = (after, agent_after)
        self.outcomes_by_grid[(before.shape, before.tobytes(), action)] = (after, agent_after)
        self.agent = agent_after
        if agent_after:
            self.seen_patterns.add(pattern(after, agent_after))

    def _follow(self, before, action, after) -> Optional[FrozenSet[Pos]]:
        """Learn from the transition; the agent's cells after it (None if lost track)"""
        if self.pending is not None and not np.array_equal(before, after):
            # an earlier contact had outcomes giving the same grid but different agents (merge or push):
            # the agent that moved now tells which
            grid0, objs0, front0, alternatives = self.pending
            self.pending = None
            d = self.directions.get(action)
            for kind, alt in alternatives.items():
                if d is not None and self._free(before, alt, d) and np.array_equal(self._move(before, alt, d), after):
                    for o in objs0:
                        self._learn_rule(grid0, o, kind, front0)
                    return shifted(alt, d)
            self.agent = None
        agent, d = self.agent, self.directions.get(action)
        if agent is not None and d is None and not np.array_equal(before, after):
            for s in SIDES:
                if self._free(before, agent, s) and np.array_equal(self._move(before, agent, s), after):
                    d = self.directions[action] = s
        if agent is not None and d is not None:
            objs = self._contacts(before, agent, d)
            if objs is not None and objs:
                known = {self._rule(before, o) for o in objs}
                if len(known) == 1 and None not in known:
                    kind = known.pop()
                    nxt = self._apply(before, agent, d, kind, objs)
                    if nxt is not None and np.array_equal(nxt[0], after):
                        for o in objs:
                            self._learn_rule(before, o, kind, shifted(agent, d))
                        return nxt[1]
                matches = {}
                for kind in OUTCOMES:
                    nxt = self._apply(before, agent, d, kind, objs)
                    if nxt is not None and np.array_equal(nxt[0], after):
                        matches[kind] = nxt[1]
                if not matches and self._learn_collision(before, agent, d, objs, after):
                    return agent
                if len(set(matches.values())) > 1:
                    self.pending = (before, objs, shifted(agent, d), matches)
                    return next(iter(matches.values()))
                if matches:
                    kind, a = next(iter(matches.items()))
                    for o in objs:
                        self._learn_rule(before, o, kind, shifted(agent, d))
                    return a
            elif objs is not None:
                if np.array_equal(self._move(before, agent, d), after):
                    return shifted(agent, d)
                if np.array_equal(before, after):
                    self.hidden_walls |= shifted(agent, d) - agent      # blocked where nothing is drawn
                    return agent
        if np.array_equal(before, after):
            return agent
        found = self._identify(before, after)
        if found is None:
            if agent is not None:
                # not explained: the agent is where an object of its pattern is, if one is
                pat = pattern(before, agent)
                same = [o for o in self._components(after, frozenset()) if pattern(after, o) == pat]
                if len(same) == 1:
                    return same[0]
            return None
        self.bg, moved, d = found
        self.directions.setdefault(action, d)
        new = shifted(moved, d)
        if self.level_fresh:
            self.start_patterns.add(pattern(after, new))
            self.level_fresh = False
        return new

    def _locate(self, grid):
        if self.bg is None or not self.start_patterns:
            return None
        found = [o for o in self._components(grid, frozenset()) if pattern(grid, o) in self.start_patterns]
        return found[0] if len(found) == 1 else None

    # ---------------------------------------------------------- the goal
    def wins_with(self, grid, agent, action, nxt) -> bool:
        d = self.directions.get(action)
        if agent is None or d is None:
            return False
        if self.win_contacts:
            objs = self._contacts(grid, agent, d) or []
            if any(pattern(grid, o) in self.win_contacts for o in objs):
                return True
        # (the agent's pattern can only have changed if its number of cells did)
        return (self.shape_goal and nxt is not None and nxt[1] is not None and len(nxt[1]) != len(agent)
                and self._shape_match(*nxt))

    def _targets(self, grid, agent) -> List[Dict[Pos, int]]:
        return [{p: int(grid[p]) for p in o} for o in self._components(grid, agent or frozenset())]

    @staticmethod
    def _fits(grid, agent, targets) -> bool:
        """Whether the agent's cells, translated, fall on cells of the same values in one of the targets"""
        cells = [(p, int(grid[p])) for p in agent]
        (a0, v0) = cells[0]
        for t in targets:
            if len(t) < len(cells):
                continue
            for q, v in t.items():
                if v == v0 and all(t.get((p[0] - a0[0] + q[0], p[1] - a0[1] + q[1])) == w for p, w in cells):
                    return True
        return False

    def level_won(self, grid, action):
        self.wins += 1
        self.hidden_walls = set()
        agent, d = self.agent, self.directions.get(action)
        if agent is not None and d is not None:
            objs = self._contacts(grid, agent, d) or []
            if objs:
                known = [self._rule(grid, o) for o in objs]
                kinds = [known[0]] if None not in known and len(set(known)) == 1 else list(OUTCOMES)
                for kind in kinds:
                    nxt = self._apply(grid, agent, d, kind, objs)
                    if nxt is None:
                        continue
                    g, a = nxt
                    if pattern(g, a) != pattern(grid, agent) and self._shape_match(g, a):
                        self.shape_goal = True
                        if kind != known[0] or None in known:
                            for o in objs:
                                self._learn_rule(grid, o, kind, shifted(agent, d))
                        break
                else:
                    self.win_contacts |= {pattern(grid, o) for o in objs}
        self._new_level()

    def lost(self, grid=None, action=None):
        self.losses += 1
        if grid is not None:
            self.losing.add((self._key(grid, self._before_agent), action))
        self._new_level()

    def _new_level(self):
        self.agent = None
        self.pending = None
        self.level_fresh = True
        self.plan = []

    def new_episode(self):
        self._new_level()

    # ---------------------------------------------------------- planning
    def make_plan(self, grid, actions) -> Optional[List]:
        start = (grid, self.agent)
        k0 = self._key(*start)
        parent = {k0: None}
        queue = deque([start])
        explore = None
        targets = self._targets(grid, self.agent) if self.shape_goal else None

        def path_to(k):
            out = []
            while parent[k] is not None:
                k, a = parent[k]
                out.append(a)
            return out[::-1]
        while queue and len(parent) < BFS_LIMIT:
            g, agent = queue.popleft()
            k = self._key(g, agent)
            for a in actions:
                if (k, a) in self.losing:
                    continue
                nxt = self.predict(g, agent, a)
                if self.wins_with(g, agent, a, nxt):
                    self.exploring = False
                    return path_to(k) + [a]
                grew = nxt is not None and nxt[1] is not None and agent is not None and len(nxt[1]) != len(agent)
                if grew and targets is not None and not self._fits(nxt[0], nxt[1], targets):
                    continue
                novel = grew and pattern(*nxt) not in self.seen_patterns
                if nxt is None or novel:
                    if explore is None:
                        explore = path_to(k) + [a]
                        if not self.win_contacts and not self.shape_goal:
                            queue.clear()           # no goal known yet: nothing to search further for
                            break
                    continue
                k2 = self._key(*nxt)
                if k2 not in parent:
                    parent[k2] = (k, a)
                    queue.append(nxt)
        self.exploring = explore is not None
        return explore

    def act(self, grid, actions):
        if self.agent is None:
            self.agent = self._locate(grid)
        if not self.plan:
            self.plan = self.make_plan(grid, actions) or []
        self._before_agent = self.agent
        if self.plan:
            self.moves_by['explored' if self.exploring else 'planned'] += 1
            return self.plan.pop(0)
        self.moves_by['explored'] += 1
        unknown = [a for a in actions if self.predict(grid, self.agent, a) is None]
        return self.rng.choice(unknown or actions)

    def after_step(self, before, action, after):
        predicted = self.predict(before, self.agent, action)
        self.observe(before, action, after)
        if predicted is None or predicted[0].shape != after.shape or not np.array_equal(predicted[0], after):
            self.plan = []

    def describe(self) -> str:
        return (f'background {self.bg}, directions {self.directions}, walls {sorted(self._walls())}, '
                f'contact rules {len(self.rules)}, win contacts {len(self.win_contacts)}, '
                f'shape goal {self.shape_goal}, start patterns {len(self.start_patterns)}')
