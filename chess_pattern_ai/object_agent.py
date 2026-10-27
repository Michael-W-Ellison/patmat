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
  give the same grid (a merged agent that moved, or a pushed object; when
  nothing changed, block is taken), the
  agent's next moves tell which (the first one that only one of them
  predicts); if the level ends first, the first outcome listed is taken.
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

Clicks (ACTION6, actions ('click', row, col)): what clicking a cell does
is learned per value of the clicked cell, as the first of these that fits
every click seen on that value:
    none      nothing changes
    map       the clicked object (the 4-connected cells of the clicked
              value) changes by a table of values (a
              table to the background removes it); nothing else changes
    offsets   the cells at fixed offsets from the clicked cell change by a
              table per offset (an offset whose entries all swap two values
              is taken to swap them for a value not yet seen there)
The clicks considered in a state are one per value not yet clicked, one
per object of a value whose clicks map, and every cell of a value whose
clicks act at offsets (at most MAX_CLICKS).
Goals for clicks: a level won by a click teaches "no cell holds v" for the
values the click removed. While no goal at all is known and clicks have
some effect, each value other than the background is tried as such a goal
in turn: a plan that removes it is made and carried out, and the value is
dropped as a goal if the level does not end. When clicks act at the same
offsets and swap the same two values for every value present, a plan to
remove a value is found by solving the presses modulo 2 (as for Lights
Out) instead of by search.

Frames: when an action produces several frames, the first shows the
agent's own effect (learned as above) and each later frame something
else, learned per position in the sequence:
    drift     the agent moved one step in a fixed direction by itself (as
              under gravity); when it cannot, it has come to rest (locked)
              and is part of the scene; an object that then appears out of
              background, with nothing around it changed, becomes the agent
              (preferring a pattern the agent has had), as does, while no
              agent is known, an object that drifts
    player    another player's move, learned like a click: the first
              changed cell is taken as the cell it acted on, with a click
              model per value of that cell (as many players as frames)
An own action that changes the agent's shape without moving it (a turn)
is learned as a transform of that pattern for that action (the new pattern
and where its box goes); it is blocked when the new cells are not free. When an action's
transforms are a geometric operation of the pattern (a quarter or half
turn, a mirror image, the transpose) with the same box offset at least
twice, the operation is used for patterns not seen yet; a pattern the
operation would change, that did not change though the cells were free,
is learned as left as it is (as is, with no operation known, a pattern
unchanged in two states).

Line goals: a line is a maximal run of cells of a value v (or of any
non-background value, ANY) along a direction (row, column, diagonal); a
line hypothesis (v, k, d) says "a new line of at least k cells forms" for
direction d or any direction (d None). Each win counts the hypotheses
formed in its frames (but the last, which may show the next level): k >= 3,
the longest formed per value and direction (not a win that a goal known
before it already explains, such as a shape match); each other action
counts the
ones formed in its frames as misses. The hypotheses kept as goals are
those formed in at least half of the wins that formed any, and without a
win at most 1 / MISS_RATIO as often as in wins (a full row passed through
by a falling piece, before it comes to rest, scores nothing). Losses
(hypotheses formed in the frames that lost, after the agent's own) are
kept the same way. Contact
goals are likewise dropped when the contact happens without a win, and
"no cell holds v" goals when v is removed without a win.

Score and drift: with a drift learned, the agent plans where its piece
comes to rest: a search over its own actions, each followed by the drift,
ends in the states where the piece locks; each is valued by
    lines of the goal formed (1 each, the goal itself) + w . (the columns'
    heights of the mass against the drift, summed; background cells covered
    against the drift; roughness: the differences of height between
    neighbouring columns)
(mass: non-background cells that were not there in every frame of the
episode). The weights w are learned from the score by the cross-entropy
method: each episode plays with weights drawn from a normal distribution,
and after every CE_POPULATION episodes the distribution's mean and spread
move (by CE_SMOOTH) to those of the CE_ELITE episodes with the highest
score (levels completed, then actions survived), the spread kept at least
CE_MIN_STD.

With other players' moves known, the agent plans by minimax: its
own moves, then each other player's possible moves (all assumed to play
against it), to MAX_DEPTH of its own moves, a win counting 1 and a loss -1,
discounted by DISCOUNT per move so that quicker wins and later losses are
preferred; moves of equal value are chosen among at random.

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

from action_puzzles import solve_xor

Pos = Tuple[int, int]
SIDES = ((-1, 0), (1, 0), (0, -1), (0, 1))
BFS_LIMIT = 30_000
NOOP_STATES = 3
OUTCOMES = ('block', 'merge', 'push')
CLICK = 6
MAX_CLICKS = 64
MAX_DEPTH = 5
DISCOUNT = 0.9
LINE_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
ANY = -100                  # line hypotheses over any non-background value
MISS_RATIO = 3              # a line hypothesis is kept while formed MISS_RATIO times as often in wins as elsewhere
CE_POPULATION, CE_ELITE, CE_MIN_STD, CE_SMOOTH = 8, 3, 0.2, 0.7
GEOMETRY = {
    'turn right': lambda a: np.rot90(a, -1), 'turn left': lambda a: np.rot90(a, 1), 'half turn': lambda a: np.rot90(a, 2),
    'mirror left-right': np.fliplr, 'mirror up-down': np.flipud, 'transpose': np.transpose,
}
FEATURES = ('height', 'holes', 'roughness')


def runs(mask, d) -> List[int]:
    """Lengths of the maximal runs of True cells along direction d"""
    out = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not (0 <= r - d[0] < rows and 0 <= c - d[1] < cols and mask[r - d[0], c - d[1]]):
                n, q = 0, (r, c)
                while 0 <= q[0] < rows and 0 <= q[1] < cols and mask[q]:
                    n += 1
                    q = (q[0] + d[0], q[1] + d[1])
                out.append(n)
    return out


class ClickModel:
    """What clicking a cell that holds one value does (see the module notes)"""

    def __init__(self):
        self.seen: List[tuple] = []                 # (grid, cell, object, grid after)
        self.ok = {'none': True, 'map': True, 'offsets': True}
        self.table: Dict[int, int] = {}
        self.offsets: set = set()
        self.off_tables: Dict[Pos, Dict[int, int]] = {}

    def update(self, grid, cell, obj, after):
        self.seen.append((grid, cell, obj, after))
        changed = {(int(r), int(c)) for r, c in np.argwhere(grid != after)}
        if changed:
            self.ok['none'] = False
        if self.ok['map']:
            if not changed <= obj:
                self.ok['map'] = False
            for q in obj:
                if self.table.setdefault(int(grid[q]), int(after[q])) != int(after[q]):
                    self.ok['map'] = False
        self.offsets |= {(q[0] - cell[0], q[1] - cell[1]) for q in changed}
        self.off_tables = {o: {} for o in self.offsets}
        self.ok['offsets'] = True
        for g, p, _, a in self.seen:
            for o in self.offsets:
                q = (p[0] + o[0], p[1] + o[1])
                if _inside(g, q) and self.off_tables[o].setdefault(int(g[q]), int(a[q])) != int(a[q]):
                    self.ok['offsets'] = False

    def kind(self) -> Optional[str]:
        if not self.seen:
            return None
        return next((k for k in ('none', 'map', 'offsets') if self.ok[k]), None)

    @staticmethod
    def _swap(table) -> Optional[Tuple[int, int]]:
        """The two values the table swaps, if every entry swaps them"""
        pairs = {frozenset((v, w)) for v, w in table.items()}
        if len(pairs) == 1 and all(v != w for v, w in table.items()):
            a, b = sorted(pairs.pop())
            return a, b
        return None

    def toggles(self) -> Optional[Tuple[int, int]]:
        """The two values every offset swaps, if the clicks act at offsets that way"""
        if self.kind() != 'offsets':
            return None
        swaps = {self._swap(t) for t in self.off_tables.values()}
        return swaps.pop() if len(swaps) == 1 and None not in swaps else None

    def predict(self, grid, cell, obj) -> Optional[np.ndarray]:
        kind = self.kind()
        if kind == 'none':
            return grid
        out = grid.copy()
        if kind == 'map':
            for q in obj:
                if int(grid[q]) not in self.table:
                    return None
                out[q] = self.table[int(grid[q])]
            return out
        if kind == 'offsets':
            for o, t in self.off_tables.items():
                q = (cell[0] + o[0], cell[1] + o[1])
                if not _inside(grid, q):
                    continue
                v = int(grid[q])
                if v not in t:
                    swap = self._swap(t)
                    if swap is None or v not in swap:
                        return None
                    t = {swap[0]: swap[1], swap[1]: swap[0]}
                out[q] = t[v]
            return out
        return None


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
    uses_frames = True

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
        self.clicks = True
        self.click_models: Dict[int, ClickModel] = {}
        self.absent_goals: set = set()              # values whose removal won a level
        self.absent_rejected: set = set()           # values removed without winning
        self.trying: Optional[int] = None           # the value a plan is removing to test it as the goal
        self.others: List[Dict[int, ClickModel]] = []   # per other player (later frames): its move models
        self.line_goals: Optional[set] = None       # version space of (v, k, d) for winning (None: no win yet)
        self.line_losses: Optional[set] = None      # the same for losing
        self.line_hits, self.line_misses = Counter(), Counter()     # wins in which, and other times, formed
        self.loss_hits, self.loss_misses = Counter(), Counter()
        self.line_events = self.loss_events = 0     # wins (losses) in which any line hypothesis formed
        self.transforms: Dict[tuple, tuple] = {}    # (action, pattern) -> (new pattern, box offset)
        self.still: Dict[tuple, set] = {}           # (action, pattern) -> states where it changed nothing
        self.transform_ops: Dict[object, Counter] = {}  # action -> (geometric operation, box offset) seen
        self.drift: Dict[int, Counter] = {}         # frame position -> directions the agent moved by itself
        self.agent_patterns: set = set()            # patterns the agent has been identified with
        self.base = None                            # cells non-background in every frame of the episode
        self.ce_mean = np.zeros(len(FEATURES))
        self.ce_std = np.ones(len(FEATURES))
        self.ce_done: List[tuple] = []              # (fitness, weights) of this generation's episodes
        self.weights = self.ce_mean.copy()
        self.episode = {'score': 0, 'actions': 0, 'used': False}
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
        if isinstance(action, tuple):
            return self._click(grid, agent, action[1:])
        u = self.unchanged_in.get(action)
        if u is not None and len(u) >= NOOP_STATES:
            return grid, agent
        return self._by_rules(grid, agent, action)

    # ------------------------------------------------------------- clicks
    def _clicked(self, grid, cell) -> FrozenSet[Pos]:
        """The object a click on the cell acts on: the 4-connected cells of its value (the cell alone on background)"""
        v = int(grid[cell])
        if self.bg is None or v == self.bg:
            return frozenset([cell])
        seen, queue = {cell}, [cell]
        while queue:
            p = queue.pop()
            for dr, dc in SIDES:
                q = (p[0] + dr, p[1] + dc)
                if q not in seen and _inside(grid, q) and int(grid[q]) == v:
                    seen.add(q)
                    queue.append(q)
        return frozenset(seen)

    def _click(self, grid, agent, cell):
        model = self.click_models.get(int(grid[cell]))
        if model is None:
            return None
        after = model.predict(grid, cell, self._clicked(grid, cell))
        if after is None:
            return None
        return after, self._agent_after(grid, agent, after)

    @staticmethod
    def _agent_after(grid, agent, after):
        if agent is None or all(after[p] == grid[p] for p in agent):
            return agent
        return None

    def _click_actions(self, grid) -> List[tuple]:
        out = []
        for v in sorted(set(np.unique(grid).tolist())):
            cells = [(int(r), int(c)) for r, c in np.argwhere(grid == v)]
            model = self.click_models.get(v)
            kind = model.kind() if model else None
            if kind is None:
                out.append(cells[0])            # not clicked yet: one click to learn what it does
            elif kind == 'map':
                done = set()
                for p in cells:
                    if p not in done:
                        obj = self._clicked(grid, p)
                        done |= obj
                        out.append(p)
            elif kind == 'offsets':
                out.extend(cells)
        return [('click',) + p for p in out[:MAX_CLICKS]]

    def _actions(self, grid, actions) -> List:
        moves = [a for a in actions if a != CLICK]
        return moves + (self._click_actions(grid) if CLICK in actions else [])

    def _xor_plan(self, grid, v) -> Optional[List[tuple]]:
        """Clicks that remove value v when every value present toggles the same offsets between two values"""
        values = set(np.unique(grid).tolist())
        models = [self.click_models.get(u) for u in values]
        if any(m is None for m in models):
            return None
        swaps = {m.toggles() for m in models}
        offsets = {frozenset(m.offsets) for m in models}
        if len(swaps) != 1 or None in swaps or len(offsets) != 1 or not values <= set(next(iter(swaps))) \
                or v not in values:
            return None
        offs = next(iter(offsets))
        cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
        masks = np.zeros((len(cells), len(cells)), dtype=np.uint8)
        index = {p: i for i, p in enumerate(cells)}
        for i, p in enumerate(cells):
            for o in offs:
                q = (p[0] + o[0], p[1] + o[1])
                if q in index:
                    masks[i, index[q]] = 1
        x = solve_xor(masks, (grid == v).ravel().astype(np.uint8))
        return None if x is None else [('click',) + p for p, bit in zip(cells, x) if bit]

    def _geometric(self, action, pat):
        """The transform an action's geometric operation gives the pattern (if the action has one)"""
        c = self.transform_ops.get(action)
        if not c:
            return None
        (op, offset), n = c.most_common(1)[0]
        if n < 2:
            return None
        new = GEOMETRY[op](np.array(pat))
        return tuple(map(tuple, new.tolist())), offset

    def _transform(self, grid, agent, action):
        pat = pattern(grid, agent)
        t = self.transforms.get((action, pat)) or self._geometric(action, pat)
        if t is None:
            return None
        new_pat, (dr, dc) = t
        r0, c0 = min(p[0] for p in agent), min(p[1] for p in agent)
        cells = {(r0 + dr + i, c0 + dc + j): v for i, row in enumerate(new_pat) for j, v in enumerate(row) if v != -1}
        if any(not _inside(grid, p) or (int(grid[p]) != self.bg and p not in agent) for p in cells):
            return grid, agent                  # no room: nothing happens
        out = grid.copy()
        for p in agent:
            out[p] = self.bg
        for p, v in cells.items():
            out[p] = v
        return out, frozenset(cells)

    def _by_rules(self, grid, agent, action):
        """The transition the contact rules predict (None if they do not say)"""
        d = self.directions.get(action)
        if agent is not None and d is None and self.bg is not None:
            return self._transform(grid, agent, action)
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
                # the moved cells: from the changed cells that held something, through neighbours of
                # their values whose value is found one step on (an object may overlap where it was)
                seeds = [p for p in changed if int(before[p]) != bg]
                values = {int(before[p]) for p in seeds}
                moved, queue = set(seeds), list(seeds)
                while queue:
                    p = queue.pop()
                    for dr, dc in SIDES:
                        q = (p[0] + dr, p[1] + dc)
                        e = (q[0] + d[0], q[1] + d[1])
                        if q not in moved and _inside(before, q) and _inside(before, e) and \
                                int(before[q]) in values and int(after[e]) == int(before[q]):
                            moved.add(q)
                            queue.append(q)
                moved = frozenset(moved)
                if not moved or any(not _inside(before, (p[0] + d[0], p[1] + d[1])) or
                                    int(after[p[0] + d[0], p[1] + d[1]]) != int(before[p]) for p in moved):
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
        if isinstance(action, tuple):
            cell = action[1:]
            if self.bg is None:
                vals, counts = np.unique(before, return_counts=True)
                self.bg = int(vals[np.argmax(counts)])
            self.click_models.setdefault(int(before[cell]), ClickModel()).update(
                before, cell, self._clicked(before, cell), after)
            agent_after = self._agent_after(before, self.agent, after)
            self.outcomes[(state, action)] = (after, agent_after)
            self.outcomes_by_grid[(before.shape, before.tobytes(), action)] = (after, agent_after)
            self.agent = agent_after
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
            keys = [pattern(grid0, o) for o in objs0]
            fits = {}
            for kind, alt in alternatives.items():
                saved = {k: self.rules.get(k) for k in keys}
                for k in keys:
                    self.rules[k] = Counter({kind: 1})
                nxt = self._by_rules(before, alt, action)
                for k, v in saved.items():
                    if v is None:
                        del self.rules[k]
                    else:
                        self.rules[k] = v
                if nxt is not None and np.array_equal(nxt[0], after):
                    fits[kind] = nxt[1]
            if len(fits) == 1:
                kind, a = next(iter(fits.items()))
                for o in objs0:
                    self._learn_rule(grid0, o, kind, front0)
                return a
            if len(fits) > 1:
                self.pending = (grid0, objs0, front0, fits)      # still not told apart
                return next(iter(fits.values()))
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
                if 'block' in matches:
                    matches = {'block': matches['block']}     # nothing changed: the simplest account
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
            if agent is not None and self.directions.get(action) is None:
                key = (action, pattern(before, agent))
                geo = self._geometric(action, key[1])
                if key not in self.transforms and geo is not None:
                    turned = self._transform(before, agent, action)
                    if turned is not None and not np.array_equal(turned[0], before):
                        self.transforms[key] = (key[1], (0, 0))     # room to turn, and nothing happened
                elif key not in self.transforms:
                    tries = self.still.setdefault(key, set())
                    tries.add(before.tobytes())
                    if len(tries) >= 2:     # it changed nothing in two places: it does not change this pattern
                        self.transforms[key] = (key[1], (0, 0))
            return agent
        if agent is not None and self.directions.get(action) is None:
            # the agent's own cells changed shape without moving: a transform
            changed = {(int(r), int(c)) for r, c in np.argwhere(before != after)}
            new = frozenset(p for p in agent | changed if int(after[p]) != self.bg)
            if new != agent and len(new) == len(agent) and changed <= agent | new and \
                    sorted(int(before[p]) for p in agent) == sorted(int(after[p]) for p in new):
                r0, c0 = min(p[0] for p in agent), min(p[1] for p in agent)
                r1, c1 = min(p[0] for p in new), min(p[1] for p in new)
                old_pat, new_pat = pattern(before, agent), pattern(after, new)
                self.transforms[(action, old_pat)] = (new_pat, (r1 - r0, c1 - c0))
                for op, f in GEOMETRY.items():
                    if np.array_equal(f(np.array(old_pat)), np.array(new_pat)):
                        self.transform_ops.setdefault(action, Counter())[(op, (r1 - r0, c1 - c0))] += 1
                return new
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
        self.agent_patterns.add(pattern(after, new))
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
    def _removes(self, grid, nxt, values) -> Optional[int]:
        """The first of the values present in grid and absent after the transition"""
        if nxt is None or not values:
            return None
        return next((v for v in sorted(values) if (grid == v).any() and not (nxt[0] == v).any()), None)

    def _goal_trials(self, grid) -> set:
        """Values to try as "no cell holds v" goals: only while no goal is known and clicks change something"""
        if self.win_contacts or self.shape_goal or self.absent_goals or self.line_goals:
            return set()
        if self.drift:
            return set()
        if not any(m.kind() in ('map', 'offsets') for m in self.click_models.values()):
            return set()
        return set(np.unique(grid).tolist()) - {self.bg} - self._walls() - self.absent_rejected

    def _mask(self, grid, v):
        return grid != self.bg if v == ANY else grid == v

    def _count(self, grid, line) -> int:
        v, k, d = line
        mask = self._mask(grid, v)
        return sum(1 for dd in ([d] if d else LINE_DIRS) for n in runs(mask, dd) if n >= k)

    def _formed(self, before, after, lines) -> int:
        """How many new lines the transition formed, for the hypothesis that forms most"""
        if not lines or before.shape != after.shape or self.bg is None:
            return 0
        return max((self._count(after, h) - self._count(before, h) for h in lines), default=0)

    def _line_hypotheses(self, before, after) -> set:
        """(v, k, d): the longest new lines (k >= 3) the transition formed, per value and direction"""
        out = set()
        if before.shape != after.shape or self.bg is None:
            return out
        for v in [ANY] + [u for u in np.unique(after).tolist() if u != self.bg]:
            mb, ma = self._mask(before, v), self._mask(after, v)
            if (ma & ~mb).sum() == 0:
                continue                        # nothing of v was added
            for d in LINE_DIRS:
                ra, rb = runs(ma, d), runs(mb, d)
                for k in sorted({n for n in ra if n >= 3}, reverse=True):
                    if sum(n >= k for n in ra) > sum(n >= k for n in rb):
                        out |= {(v, k, d), (v, k, None)}
                        break
        return out

    @staticmethod
    def _kept(hits, misses, events) -> Optional[set]:
        """
        The hypotheses formed in at least half of the events (wins or losses that formed any), and
        elsewhere at most 1 / MISS_RATIO as often (None while no event has formed any)
        """
        if not events:
            return None
        return {h for h, n in hits.items() if 2 * n >= events and MISS_RATIO * misses[h] <= n}

    def _transitions(self, before, frames):
        seq = [before] + list(frames or [])
        return [(a, b) for a, b in zip(seq, seq[1:]) if a.shape == b.shape]

    def _falsify(self, before, frames, lost=False):
        """
        A hypothesis formed in frames that did not win (or, for losses, lose) counts a miss; it is dropped
        once its misses outnumber the wins (losses) in which it formed
        """
        for prev, cur in self._transitions(before, frames):
            for space, hits, misses, skip in ((self.line_goals, self.line_hits, self.line_misses, False),
                                              (self.line_losses, self.loss_hits, self.loss_misses, lost)):
                if not space or skip:
                    continue
                for h in [h for h in space if self._count(cur, h) > self._count(prev, h)]:
                    misses[h] += 1
                    if MISS_RATIO * misses[h] > hits[h]:
                        space.discard(h)        # (and it stays out: _kept checks the same)
            if self.absent_goals and not lost:
                self.absent_goals = {v for v in self.absent_goals if not ((prev == v).any() and not (cur == v).any())}

    def wins_with(self, grid, agent, action, nxt) -> bool:
        if self._removes(grid, nxt, self.absent_goals) is not None:
            return True
        if self.line_goals and nxt is not None and self._formed(grid, nxt[0], self.line_goals) > 0:
            return True
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

    def level_won(self, grid, action, frames=None):
        self.wins += 1
        self.episode['score'] += 1
        self.hidden_walls = set()
        # a win that a goal known before it explains gives line hypotheses no support
        lines, self.line_goals = self.line_goals, None
        explained = self.wins_with(grid, self.agent, action, self.predict(grid, self.agent, action))
        self.line_goals = lines
        found = set()
        # (the last frame of the action may already show the next level)
        for prev, cur in self._transitions(grid, frames[:-1] if frames and len(frames) > 1 else frames):
            found |= self._line_hypotheses(prev, cur)
        if not explained:
            self.line_hits.update(found)
            self.line_events += bool(found)
            self.line_goals = self._kept(self.line_hits, self.line_misses, self.line_events)
        if isinstance(action, tuple):
            if self.trying is not None:
                self.absent_goals.add(self.trying)
            nxt = self._click(grid, self.agent, action[1:])
            if nxt is not None:
                self.absent_goals |= {v for v in np.unique(grid).tolist() if not (nxt[0] == v).any()}
        self.trying = None
        own = frames[0] if frames and frames[0].shape == grid.shape and not np.array_equal(frames[0], grid) \
            else None
        if own is None:
            nxt = self.predict(grid, self.agent, action)
            own = nxt[0] if nxt is not None and nxt[0].shape == grid.shape else None
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

    def lost(self, grid=None, action=None, frames=None):
        self.losses += 1
        if grid is not None:
            self.losing.add((self._key(grid, self._before_agent), action))
        found = set()
        for prev, cur in self._transitions(grid, frames)[1:]:      # (frames after the agent's own)
            found |= self._line_hypotheses(prev, cur)
        self.loss_hits.update(found)
        self.loss_events += bool(found)
        self.line_losses = self._kept(self.loss_hits, self.loss_misses, self.loss_events)
        if grid is not None:
            self._falsify(grid, frames, lost=True)
        self.base = None
        self._new_level()

    def _new_level(self):
        if self.pending is not None:
            # not told apart before the level ended: the first outcome that fitted, as one observation
            grid0, objs0, front0, alternatives = self.pending
            for o in objs0:
                self._learn_rule(grid0, o, next(iter(alternatives)), front0)
        self.agent = None
        self.pending = None
        self.level_fresh = True
        self.plan = []

    def new_episode(self):
        self._ce_next()
        self.base = None
        self._new_level()

    # ---------------------------------------------------------- planning
    # ------------------------------------------------------ other players
    def _others_known(self) -> bool:
        return any(m.kind() in ('map', 'offsets') for d in self.others for m in d.values())

    def _other_moves(self, i, grid) -> List[np.ndarray]:
        out, seen = [], set()
        for v, m in self.others[i].items():
            if m.kind() not in ('map', 'offsets'):
                continue
            done = set()
            for r, c in np.argwhere(grid == v):
                p = (int(r), int(c))
                if p in done:
                    continue
                obj = self._clicked(grid, p)
                if m.kind() == 'map':
                    done |= obj
                g = m.predict(grid, p, obj)
                if g is not None and not np.array_equal(g, grid) and g.tobytes() not in seen:
                    seen.add(g.tobytes())
                    out.append(g)
        return out

    def _minimax(self, grid, actions):
        """(value, best actions) of the agent's move here, against the other players"""
        memo = {}

        def ours(g, agent, depth):
            key = ('us', g.tobytes(), depth)
            if key in memo:
                return memo[key]
            best, acts = None, []
            for a in self._actions(g, actions):
                nxt = self.predict(g, agent, a)
                if nxt is None or nxt[0].shape != g.shape or np.array_equal(nxt[0], g):
                    continue
                if self.wins_with(g, agent, a, nxt):
                    v = 1.0
                elif self._formed(g, nxt[0], self.line_losses) > 0:
                    v = -1.0
                elif depth <= 1:
                    v = 0.0
                else:
                    v = DISCOUNT * theirs(nxt[0], nxt[1], 0, depth)
                if best is None or v > best:
                    best, acts = v, [a]
                elif v == best:
                    acts.append(a)
            memo[key] = (best if best is not None else 0.0, acts)
            return memo[key]

        def theirs(g, agent, i, depth):
            if i == len(self.others):
                return ours(g, agent, depth - 1)[0]
            key = ('them', g.tobytes(), i, depth)
            if key in memo:
                return memo[key]
            moves = self._other_moves(i, g)
            if not moves:
                v = theirs(g, agent, i + 1, depth)
            else:
                v = min(-1.0 if self._formed(g, m, self.line_losses) > 0 else theirs(m, agent, i + 1, depth)
                        for m in moves)
            memo[key] = v
            return v
        return ours(grid, self.agent, MAX_DEPTH)

    # ------------------------------------------------------------ drift
    def _settle(self, grid, agent, action):
        """The agent's action followed by the drift: (grid, agent, locked) or None if not known"""
        nxt = self.predict(grid, agent, action)
        if nxt is None or nxt[1] is None or nxt[0].shape != grid.shape:
            return None
        g, a = nxt
        d = self._drift_dir(1)
        if self._free(g, a, d):
            g, a = self._move(g, a, d), shifted(a, d)
        return g, a, not self._free(g, a, d)

    def _features(self, before, locked) -> np.ndarray:
        d = self._drift_dir(1)
        base = self.base if self.base is not None and self.base.shape == locked.shape else np.zeros(locked.shape, bool)
        mass = (locked != self.bg) & ~base
        free = (locked == self.bg) & ~base
        proj = lambda p: p[0] * d[0] + p[1] * d[1]
        across = (abs(d[1]), abs(d[0]))         # the direction of the columns' index
        cols: Dict[int, List] = {}
        for r, c in np.argwhere(~base):
            cols.setdefault(int(r) * across[0] + int(c) * across[1], []).append((int(r), int(c)))
        heights, holes = [], 0
        for cells in cols.values():
            cells.sort(key=proj)
            top = max(proj(p) for p in cells)
            filled = [proj(p) for p in cells if mass[p]]
            heights.append(top - min(filled) + 1 if filled else 0)
            if filled:
                holes += sum(1 for p in cells if free[p] and proj(p) > min(filled))
        rough = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        return np.array([sum(heights), holes, rough], dtype=float)

    def _value(self, before, locked) -> float:
        """Lines of the goal formed (worth 1 each), plus the learned weights on the other features"""
        lines = self._formed(before, locked, self.line_goals) if self.line_goals else 0
        return lines + float(self.weights @ self._features(before, locked))

    def _drift_plan(self, grid, actions) -> Optional[List]:
        """The actions to the resting place of the best value (and an action not known yet first)"""
        moves = [a for a in actions if a != CLICK]
        unknown = [a for a in moves if self.predict(grid, self.agent, a) is None]
        if unknown:
            self.exploring = True
            return [unknown[0]]
        start = (grid, self.agent)
        parent = {self._key(*start): None}
        queue = deque([start])
        best, best_path = None, None

        def path_to(k):
            out = []
            while parent[k] is not None:
                k, a = parent[k]
                out.append(a)
            return out[::-1]
        while queue and len(parent) < BFS_LIMIT:
            g, agent = queue.popleft()
            k = self._key(g, agent)
            for a in moves:
                nxt = self._settle(g, agent, a)
                if nxt is None:
                    continue
                g2, a2, locked = nxt
                if locked:
                    value = self._value(grid, g2)
                    if best is None or value > best:
                        best, best_path = value, path_to(k) + [a]
                    continue
                k2 = self._key(g2, a2)
                if k2 not in parent:
                    parent[k2] = (k, a)
                    queue.append((g2, a2))
        self.exploring = False
        self.episode['used'] = True
        return best_path

    def _ce_next(self):
        """End of an episode: record its fitness; after a generation, move the distribution to the elite"""
        if self.episode['used']:
            fitness = self.episode['score'] + 0.001 * self.episode['actions']
            self.ce_done.append((fitness, self.weights.copy()))
            if len(self.ce_done) >= CE_POPULATION:
                elite = np.array([w for _, w in sorted(self.ce_done, key=lambda t: -t[0])[:CE_ELITE]])
                self.ce_mean = CE_SMOOTH * elite.mean(axis=0) + (1 - CE_SMOOTH) * self.ce_mean
                self.ce_std = np.maximum(CE_SMOOTH * elite.std(axis=0) + (1 - CE_SMOOTH) * self.ce_std, CE_MIN_STD)
                self.ce_done = []
        self.weights = np.array([self.rng.gauss(m, s) for m, s in zip(self.ce_mean, self.ce_std)])
        self.episode = {'score': 0, 'actions': 0, 'used': False}

    def make_plan(self, grid, actions) -> Optional[List]:
        if self._drift_dir(1) is not None and self.agent is not None:
            plan = self._drift_plan(grid, actions)
            if plan:
                return plan
        if self._others_known():
            # explore what is not known first (a value not clicked yet), then play the minimax move
            unknown = [a for a in self._actions(grid, actions) if self.predict(grid, self.agent, a) is None]
            value, best = self._minimax(grid, actions)
            if best and (value != 0 or not unknown or self.line_goals is not None):
                self.exploring = False
                return [self.rng.choice(best)]
            if unknown:
                self.exploring = True
                return [unknown[0]]
        start = (grid, self.agent)
        k0 = self._key(*start)
        parent = {k0: None}
        queue = deque([start])
        explore = None
        targets = self._targets(grid, self.agent) if self.shape_goal else None
        trials = self._goal_trials(grid)
        for v in sorted(self.absent_goals | trials, key=lambda v: int((grid == v).sum())):
            xor = self._xor_plan(grid, v)
            if xor:
                self.exploring = False
                self.trying = None if v in self.absent_goals else v
                return xor

        def path_to(k):
            out = []
            while parent[k] is not None:
                k, a = parent[k]
                out.append(a)
            return out[::-1]
        while queue and len(parent) < BFS_LIMIT:
            g, agent = queue.popleft()
            k = self._key(g, agent)
            for a in self._actions(g, actions):
                if (k, a) in self.losing:
                    continue
                nxt = self.predict(g, agent, a)
                if self.wins_with(g, agent, a, nxt):
                    self.exploring = False
                    self.trying = None
                    return path_to(k) + [a]
                tried = self._removes(g, nxt, trials)
                if tried is not None:
                    self.exploring = False
                    self.trying = tried
                    return path_to(k) + [a]
                grew = nxt is not None and nxt[1] is not None and agent is not None and len(nxt[1]) != len(agent)
                if grew and targets is not None and not self._fits(nxt[0], nxt[1], targets):
                    continue
                novel = grew and pattern(*nxt) not in self.seen_patterns
                if nxt is None or novel:
                    if explore is None:
                        explore = path_to(k) + [a]
                        if not self.win_contacts and not self.shape_goal and not self.absent_goals \
                                and not trials:
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
        self.episode['actions'] += 1
        if self.agent is None:
            self.agent = self._locate(grid)
        if not self.plan:
            self.plan = self.make_plan(grid, actions) or []
        self._before_agent = self.agent
        if self.plan:
            self.moves_by['explored' if self.exploring else 'planned'] += 1
            return self.plan.pop(0)
        self.moves_by['explored'] += 1
        options = self._actions(grid, actions)
        unknown = [a for a in options if self.predict(grid, self.agent, a) is None]
        if unknown or options:
            return self.rng.choice(unknown or options)
        return ('click', self.rng.randrange(grid.shape[0]), self.rng.randrange(grid.shape[1]))

    def _drift_dir(self, i=1) -> Optional[Pos]:
        c = self.drift.get(i)
        if not c:
            return None
        d, n = c.most_common(1)[0]
        return d if n >= 2 and d in SIDES else None

    def _appeared(self, prev, cur) -> Optional[FrozenSet[Pos]]:
        """
        An object of cells that were background before, with nothing around it changed (one of a pattern
        the agent has had, if there are several)
        """
        if self.bg is None:
            return None
        new = (cur != self.bg) & (prev == self.bg)
        found, seen = [], set()
        for r, c in np.argwhere(new):
            p = (int(r), int(c))
            if p in seen:
                continue
            comp, queue = {p}, [p]
            while queue:
                q = queue.pop()
                for dr, dc in SIDES:
                    x = (q[0] + dr, q[1] + dc)
                    if x not in comp and _inside(cur, x) and new[x]:
                        comp.add(x)
                        queue.append(x)
            seen |= comp
            around = {(q[0] + dr, q[1] + dc) for q in comp for dr, dc in SIDES} - comp
            if all(not _inside(cur, q) or prev[q] == cur[q] for q in around):
                found.append(frozenset(comp))
        known = [o for o in found if pattern(cur, o) in self.agent_patterns]
        if len(known) == 1:
            return known[0]
        return found[0] if len(found) == 1 else None

    def _learn_frames(self, frames):
        """The frames after the agent's own: drift of the agent, a new agent, or other players' moves"""
        agent = self.agent
        for i, (prev, cur) in enumerate(zip(frames, frames[1:]), start=1):
            while len(self.others) < i:
                self.others.append({})
            if prev.shape != cur.shape:
                agent = None
                continue
            if agent is not None and self.bg is not None:
                moved = next((s for s in SIDES if self._free(prev, agent, s)
                              and np.array_equal(self._move(prev, agent, s), cur)), None)
                if moved is not None:
                    self.drift.setdefault(i, Counter())[moved] += 1
                    agent = shifted(agent, moved)
                    continue
                d = self._drift_dir(i) or self._drift_dir(1)
                if d is not None and not self._free(prev, agent, d):
                    self.drift.setdefault(i, Counter())['rest'] += 1
                    agent = None                # it came to rest: now part of the scene
                    if np.array_equal(prev, cur):
                        continue
            if agent is None and self._drift_dir(1) is not None:
                found = self._identify(prev, cur)
                if found is not None and found[2] == self._drift_dir(1):
                    agent = shifted(found[1], found[2])     # the object that drifted is the agent
                    self.agent_patterns.add(pattern(cur, agent))
                    continue
                agent = self._appeared(prev, cur)
                if agent is not None or self.drift.get(i):
                    continue
            if i in self.drift:
                continue
            changed = sorted((int(r), int(c)) for r, c in np.argwhere(prev != cur))
            if not changed:
                continue
            cell = changed[0]
            self.others[i - 1].setdefault(int(prev[cell]), ClickModel()).update(prev, cell,
                                                                                self._clicked(prev, cell), cur)
        self.agent = agent

    def _track_base(self, grids):
        for g in grids:
            if self.bg is None:
                return
            if self.base is None or self.base.shape != g.shape:
                self.base = g != self.bg
            else:
                self.base &= g != self.bg

    def _contact_falsify(self, before, action):
        """A contact goal touched without a win is not the goal"""
        d = self.directions.get(action)
        if self.win_contacts and self.agent is not None and d is not None:
            objs = self._contacts(before, self.agent, d) or []
            self.win_contacts -= {pattern(before, o) for o in objs}

    def after_step(self, before, action, after, frames=None):
        self._track_base([before] + list(frames or [after]))
        self._falsify(before, frames or [after])
        self._contact_falsify(before, action)
        if frames and len(frames) > 1:
            own = frames[0]
            self.observe(before, action, own)
            self._learn_frames(frames)
            self.plan = []                      # (other frames followed: plan again from what they show)
            return
        predicted = self.predict(before, self.agent, action)
        self.observe(before, action, after)
        if self.trying is not None and before.shape == after.shape and not (after == self.trying).any():
            self.absent_rejected.add(self.trying)        # removed, and the level did not end
            self.trying = None
            self.plan = []
        if predicted is None or predicted[0].shape != after.shape or not np.array_equal(predicted[0], after):
            self.plan = []

    def describe(self) -> str:
        return (f'background {self.bg}, directions {self.directions}, walls {sorted(self._walls())}, '
                f'contact rules {len(self.rules)}, win contacts {len(self.win_contacts)}, '
                f'shape goal {self.shape_goal}, start patterns {len(self.start_patterns)}, '
                f'clicks {({v: m.kind() for v, m in sorted(self.click_models.items())})}, '
                f'absent goals {sorted(self.absent_goals)}, rejected {sorted(self.absent_rejected)}, '
                f'other players {len(self.others)}, line goals {sorted(self.line_goals or [], key=str)}, '
                f'line losses {sorted(self.line_losses or [], key=str)}, '
                f'drift {({i: dict(c) for i, c in self.drift.items()})}, transforms {len(self.transforms)}, '
                f'weights {np.round(self.ce_mean, 2).tolist()}')
