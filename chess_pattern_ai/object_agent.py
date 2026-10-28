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
    drop      one cell changes from a value e to a value w: the cell where
              something dropped along a direction d in the clicked line
              lands (past the cells that are not e at the line's start, the
              last of the run of e that follows), as in Connect Four
(for other players, whose clicked cell is not seen, a drop that fits at
least 3 moves is preferred to the others, as it allows the fewest moves)
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

With other players' moves known, the agent plans by search: its own
moves, then each other player's possible moves (all assumed to play
against it), by alpha-beta, deepened one of its moves at a time up to
MAX_DEPTH while each depth stays within SEARCH_NODES positions; a win
counts 1 and a loss -1, discounted by DISCOUNT per move so that quicker
wins and later losses are preferred, and a position at the depth limit is
valued by progress toward the learned lines (the windows of k cells along
a goal line that hold only its value and cells that moves replace, each
worth 4 ^ (cells of its value) / 4 ^ k, the other players' loss lines
counting against; scaled into (-0.5, 0.5)). Each player's moves are
limited to MAX_OPTIONS, those next to cells that moves have marked (on an
empty board, those nearest the centre). Moves of equal value are chosen
among at random.

Planning: breadth-first search over (grid, agent cells) states predicted
by the rules, up to BFS_LIMIT states. Under the rules learned an agent
only grows, so once the shape goal is known, a state whose agent no longer
fits inside any object of the starting grid (same values at the same
relative places) is not searched further. The search it stops at the first move that
reaches a goal, else goes to the nearest move whose outcome is unknown or
would give the agent a pattern not seen before (with no goal known yet,
the search stops there).

Stored patterns (pattern_store.py): facts() lists what was learned with
each value replaced by its role in the game (roles(): placed by the
agent's clicks, by the other players, replaced by moves, background,
wall), so that what two games share can be found though they are drawn in
other colours; export_state() / import_state() keep the learned rules,
models, goals and weights of one game. use_generic() offers the agent
patterns found in several games, which it uses only where its own
evidence is still missing, and only once the pattern's conditions hold in
this game (_apply_priors): a line goal or loss of the role's values (the
length tried rising each time a line forms without a win or loss), losses
as the goal's lines for the other players, the other players' moves as
drops like the agent's clicks, the score weights to start from, drift and
turns taken from one observation instead of two, the directions of actions
not yet seen to move the agent (dropped when an action is seen to do
otherwise), and, while no goal is known, touching objects of the kind that
won elsewhere (the only object of its pattern) as candidate goals.
priors_applied records which were used.

An action that changed nothing counts toward its being a no-op (or, for a
pattern, toward an action that leaves it as it is) only where it could not
have been a blocked move: see _maybe_blocked.
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
SEARCH_NODES = 3000         # positions searched per depth of the iterative deepening
MAX_OPTIONS = 12            # moves considered per player in a position (those near marks, if more)
LINE_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
ANY = -100                  # line hypotheses over any non-background value
MIN_LINE = 3                # the shortest line a line hypothesis is about
MISS_RATIO = 3              # a line hypothesis is kept while formed MISS_RATIO times as often in wins as elsewhere
CE_POPULATION, CE_ELITE, CE_MIN_STD, CE_SMOOTH = 8, 3, 0.2, 0.7
TURNS = ('turn right', 'turn left', 'half turn')
MIRRORS = ('mirror left-right', 'mirror up-down', 'transpose')
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
        self.drops = {d: True for d in SIDES}       # drop along d still fits every click seen
        self.drop_values: Optional[Tuple[int, int]] = None    # (value the cell held, value placed)
        self.like: Optional[Pos] = None             # a drop direction expected (a generic or universal pattern)

    @staticmethod
    def drop_target(grid, cell, d, empty) -> Optional[Pos]:
        """
        Where something dropped along d in the line through the cell lands: from the line's start
        (against d), past cells that are not empty, the last of the run of empty cells that follows
        """
        p = cell
        while _inside(grid, (p[0] - d[0], p[1] - d[1])):
            p = (p[0] - d[0], p[1] - d[1])
        while _inside(grid, p) and int(grid[p]) != empty:
            p = (p[0] + d[0], p[1] + d[1])
        last = None
        while _inside(grid, p) and int(grid[p]) == empty:
            last, p = p, (p[0] + d[0], p[1] + d[1])
        return last

    def _update_drop(self, grid, cell, changed, after):
        if len(changed) != 1:
            if changed:
                self.drops = {d: False for d in SIDES}
            return
        q = next(iter(changed))
        values = (int(grid[q]), int(after[q]))
        if self.drop_values is None:
            self.drop_values = values
        if values != self.drop_values:
            self.drops = {d: False for d in SIDES}
            return
        for d in SIDES:
            if self.drops[d] and self.drop_target(grid, cell, d, values[0]) != q:
                self.drops[d] = False

    def update(self, grid, cell, obj, after):
        self.seen.append((grid, cell, obj, after))
        changed = {(int(r), int(c)) for r, c in np.argwhere(grid != after)}
        self._update_drop(grid, cell, changed, after)
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

    def drop_dir(self) -> Optional[Pos]:
        if self.drop_values is None:
            return None
        if self.like is not None and self.drops[self.like]:
            return self.like
        return next((d for d in SIDES if self.drops[d]), None)

    def kind(self, narrow=False) -> Optional[str]:
        """
        The first kind that fits every click seen. narrow (for other players, whose clicked cell is
        not seen): a drop that fits at least 3 moves comes first, as it allows the fewest moves
        """
        if not self.seen:
            return None
        if narrow and (len(self.seen) >= 3 or self.like is not None) and self.drop_dir() is not None \
                and not self.ok['none']:
            return 'drop'
        if self.like is not None and self.drop_dir() == self.like and not self.ok['none']:
            return 'drop'                       # a drop expected (generic or universal pattern) that fits
        kind = next((k for k in ('none', 'map', 'offsets') if self.ok[k]), None)
        if kind is None and self.drop_dir() is not None:
            return 'drop'
        return kind

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

    def predict(self, grid, cell, obj, narrow=False) -> Optional[np.ndarray]:
        kind = self.kind(narrow)
        if kind == 'none':
            return grid
        out = grid.copy()
        if kind == 'drop':
            q = self.drop_target(grid, cell, self.drop_dir(), self.drop_values[0])
            if q is not None:
                out[q] = self.drop_values[1]
            return out
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
        self.contact_kinds: Dict[tuple, str] = {}  # contact goal pattern -> 'unique' / 'repeated' (in its level)
        self.contact_rejected: set = set()          # patterns touched without a win
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
        self._window_cache: Dict[tuple, np.ndarray] = {}
        self._action_cache: Dict[bytes, List] = {}
        self.transforms: Dict[tuple, tuple] = {}    # (action, pattern) -> (new pattern, box offset)
        self.still: Dict[tuple, set] = {}           # (action, pattern) -> states where it changed nothing
        self.transform_ops: Dict[object, Counter] = {}  # action -> (geometric operation, box offset) seen
        self.drift: Dict[int, Counter] = {}         # frame position -> directions the agent moved by itself
        self.agent_patterns: set = set()            # patterns the agent has been identified with
        self.base = None                            # cells non-background in every frame of the episode
        self.ce_mean = np.zeros(len(FEATURES))
        self.ce_std = np.ones(len(FEATURES))
        self.ce_done: List[tuple] = []              # (fitness, weights) of this generation's episodes
        self.ce_generations = 0
        self.weights = self.ce_mean.copy()
        self.episode = {'score': 0, 'actions': 0, 'used': False}
        self.plan: List = []
        self.wins = self.losses = 0
        self.moves_by = {'planned': 0, 'explored': 0}
        self.exploring = False
        self._before_agent = None
        self.generic: Dict[tuple, dict] = {}        # generic patterns to try (pattern_store), by key
        self.priors_applied: set = set()            # keys of the generic patterns used in this game
        self.prior_lines: Dict[tuple, dict] = {}    # line pattern key -> {'k': length tried, 'falsified': bool}
        self._prior_hyps: Dict[tuple, tuple] = {}   # line hypothesis -> the line pattern key it came from
        self._generic_weights = False               # the score weights started from a generic pattern
        self.shape: Optional[Tuple[int, int]] = None    # the last grid's shape
        self.prior_directions: Dict[object, Pos] = {}   # action -> direction from generic patterns (not seen yet)
        self.directions_rejected: set = set()           # actions whose generic direction did not hold here
        self._dir_misses: Counter = Counter()           # action -> times it left the agent where it was, with room
        self.prior_contacts: set = set()                # object patterns to try touching (a generic contact goal)
        self._level_first = None                        # (grid, agent cells) when contact goals were first tried in the level
        self.meta: Dict[str, dict] = {}                 # universal elements to use (meta_patterns)
        self.meta_applied: set = set()                  # the elements used in this game
        self.meta_gravity: Optional[Pos] = None
        self.meta_ops: set = set()                      # geometric operations of the rotation / reflection elements
        self.prior_shape = False                        # "take another object's shape" tried as the goal
        self.shape_rejected = False                     # ... and reached without a win

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
            elif kind == 'drop':
                targets = set()
                for p in cells:
                    q = model.drop_target(grid, p, model.drop_dir(), model.drop_values[0])
                    if q is not None and q not in targets:
                        targets.add(q)
                        out.append(p)
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
        if n == 1 and ('transform', action, op, offset) in self.generic:
            self.priors_applied.add(('transform', action, op, offset))
        elif n == 1 and op in self.meta_ops:
            self.meta_applied.add('rotation' if op in TURNS else 'reflection')
        elif n < 2:
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

    def _dir(self, action) -> Optional[Pos]:
        """The action's direction: seen in this game, else from a generic pattern (while not contradicted)"""
        d = self.directions.get(action)
        if d is None and action in self.prior_directions:
            d = self.prior_directions[action]
            self.priors_applied.add(('direction', action, d))
        return d

    def _by_rules(self, grid, agent, action):
        """The transition the contact rules predict (None if they do not say)"""
        d = self._dir(action)
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
            blocked = self._maybe_blocked(before, action)
            if self.unchanged_in.get(action, set()) is not None and not blocked:
                self.unchanged_in.setdefault(action, set()).add(state)
        else:
            self.unchanged_in[action] = None
        agent = self.agent
        agent_after = self._follow(before, action, after)
        self._check_prior_direction(before, action, after, agent)
        self.outcomes[(state, action)] = (after, agent_after)
        self.outcomes_by_grid[(before.shape, before.tobytes(), action)] = (after, agent_after)
        self.agent = agent_after
        if agent_after:
            self.seen_patterns.add(pattern(after, agent_after))

    def _maybe_blocked(self, grid, action) -> bool:
        """
        Whether an action that changed nothing may have been a move that was blocked: its direction (seen or
        generic) is blocked, or, not known to move the agent, the agent is blocked on a side that no known
        direction takes. Such a state is no sign that the action does nothing.
        """
        if self.agent is None or isinstance(action, tuple):
            return False
        d = self.directions.get(action) or self.prior_directions.get(action)
        if d is not None:
            return not self._free(grid, self.agent, d)
        taken = set(self.directions.values()) | set(self.prior_directions.values())
        return any(not self._free(grid, self.agent, s) for s in SIDES if s not in taken)

    def _check_prior_direction(self, before, action, after, agent):
        """
        A generic direction is dropped for the action when, with room to move that way, the action changed
        the grid in some other way, or left it as it was twice (once could be a wall that is not drawn)
        """
        d = self.prior_directions.get(action)
        if d is None or action in self.directions or agent is None or not self._free(before, agent, d):
            return
        if np.array_equal(before, after):
            self._dir_misses[action] += 1
            if self._dir_misses[action] < 2:
                return
        self.directions_rejected.add(action)
        del self.prior_directions[action]

    def _pushed_by_prior(self, before, action, after, agent) -> bool:
        """
        Whether the action's generic direction explains a change at objects in the agent's front: one of
        OUTCOMES other than block gives the grid after, or every changed cell is within two steps ahead
        of the agent's or those objects' cells (a pushed object that hit another)
        """
        d = self.prior_directions.get(action)
        objs = self._contacts(before, agent, d) if d is not None else None
        if not objs:
            return False
        for kind in OUTCOMES[1:]:
            nxt = self._apply(before, agent, d, kind, objs)
            if nxt is not None and np.array_equal(nxt[0], after):
                return True
        near = set(agent).union(*objs)
        near |= shifted(near, d) | shifted(shifted(near, d), d)
        return all((int(r), int(c)) in near for r, c in np.argwhere(before != after))

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
            if d is None and self._pushed_by_prior(before, action, after, agent):
                d = self.directions[action] = self.prior_directions[action]
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
                elif key not in self.transforms and not self._maybe_blocked(before, action):
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

    def _windows(self, shape, k, d) -> np.ndarray:
        """Flat indices of every k cells in a straight line along d (all directions if None)"""
        key = (shape, k, d)
        if key not in self._window_cache:
            idx = []
            for dd in ([d] if d else LINE_DIRS):
                for r in range(shape[0]):
                    for c in range(shape[1]):
                        r1, c1 = r + (k - 1) * dd[0], c + (k - 1) * dd[1]
                        if 0 <= r1 < shape[0] and 0 <= c1 < shape[1]:
                            idx.append([(r + i * dd[0]) * shape[1] + c + i * dd[1] for i in range(k)])
            self._window_cache[key] = np.array(idx, dtype=np.int64).reshape(-1, k)
        return self._window_cache[key]

    def _count(self, grid, line) -> int:
        """Windows of k cells along the line's direction(s) that all hold its value"""
        v, k, d = line
        w = self._windows(grid.shape, k, d)
        if not len(w):
            return 0
        vals = grid.ravel()[w]
        return int(((vals != self.bg) if v == ANY else (vals == v)).all(axis=1).sum())

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
                for k in sorted({n for n in ra if n >= MIN_LINE}, reverse=True):
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
                        if h in self._prior_hyps:
                            self.prior_lines[self._prior_hyps[h]]['falsified'] = True
            if self.absent_goals and not lost:
                self.absent_goals = {v for v in self.absent_goals if not ((prev == v).any() and not (cur == v).any())}

    def wins_with(self, grid, agent, action, nxt) -> bool:
        if self._removes(grid, nxt, self.absent_goals) is not None:
            return True
        if self.line_goals and nxt is not None and self._formed(grid, nxt[0], self.line_goals) > 0:
            return True
        d = self._dir(action)
        if agent is None or d is None:
            return False
        if self.win_contacts or self.prior_contacts:
            objs = self._contacts(grid, agent, d) or []
            if any(pattern(grid, o) in self.win_contacts | self.prior_contacts for o in objs):
                return True
        # (the agent's pattern can only have changed if its number of cells did)
        return ((self.shape_goal or self.prior_shape) and nxt is not None and nxt[1] is not None and len(nxt[1]) != len(agent)
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
        self.prior_contacts = set()
        self.prior_shape = False
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
        agent, d = self.agent, self._dir(action)
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
                    for o in objs:
                        self.win_contacts.add(pattern(grid, o))
                        self.contact_kinds[pattern(grid, o)] = self._contact_kind(grid, o, agent)
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
        self._level_first = None

    def new_episode(self):
        self._ce_next()
        self.base = None
        self._new_level()

    # -------------------------------------------------- stored patterns
    def roles(self) -> Dict[int, str]:
        """
        The role of each value in this game: 'mine' (placed by the agent's clicks), 'theirs' (placed by
        the other players' moves), 'empty' (replaced by moves), 'background', 'wall'; the first that
        applies, in that order (ANY is 'any')
        """
        mine, _ = self._placed([self.click_models])
        theirs, _ = self._placed(self.others)
        _, empty = self._placed()
        out = {ANY: 'any'}
        for role, values in (('wall', self._walls()), ('background', {self.bg}), ('empty', empty),
                             ('theirs', theirs - mine), ('mine', mine)):
            out.update({v: role for v in values if v is not None})
        return out

    def _role_values(self, role) -> set:
        return {v for v, r in self.roles().items() if r == role}

    def _click_keys(self, role) -> set:
        """(role of the clicked value, kind, drop direction) of the agent's click models whose kind is known"""
        out = set()
        for v, m in self.click_models.items():
            kind = m.kind()
            if kind in ('map', 'offsets', 'drop'):
                out.add((role.get(v, 'other'), kind, m.drop_dir() if kind == 'drop' else None))
        return out

    def _players(self) -> int:
        """Other players whose moves are known (a model of a known kind)"""
        return sum(any(m.kind(narrow=True) in ('map', 'offsets', 'drop') for m in models.values())
                   for models in self.others)

    def facts(self) -> Dict[tuple, dict]:
        """
        What was learned in this game, with values replaced by their roles (roles()): key -> parameters.
            ('direction', action, d)
            ('no_direction', action)            a generic direction for the action did not hold here
            ('acts_by', how)                    'moves' (actions move the agent), 'clicks' (clicks change cells)
            ('own_click', role, kind, d)        the agent's clicks on a value (d: a drop's direction)
            ('other_click', role, kind, d)      the same for the other players' moves
            ('players', n)                      n other players whose moves are known
            ('others_like_me',)                 every other player's moves are of a kind the agent's are
            ('line_goal', role, d)  {k, full}   a line of k cells of the role's value wins (d None: any
            ('line_loss', role, d)  {k, full}   direction); loses (full: k is the grid's extent along d)
            ('loss_like_goal',)                 a line of k in any direction wins for 'mine', loses for 'theirs'
            ('absent_goal', role)               no cell holding the value wins ('cleared': a value its
                                                clicks clear to the background, 'toggled': one they toggle)
            ('contact_goal', kind)              touching an object wins ('unique': no other object of its
                                                pattern was in the level; 'repeated')
            ('shape_goal',)
            ('drift', i, d)                     at frame i the agent moves along d by itself
            ('transform', action, op, offset)   the action turns or mirrors the agent's pattern
            ('drift_weights',)  {mean, std}     the score weights learned for where a piece comes to rest
        Line goals and losses are included once wins (losses) have formed lines.
        """
        role = self.roles()
        r = lambda v: role.get(v, 'other')
        out = {('direction', a, d): {} for a, d in self.directions.items()}
        out.update({('no_direction', a): {} for a in self.directions_rejected if a not in self.directions})
        own = self._click_keys(role)
        out.update({('own_click',) + k: {} for k in own})
        out.update({k: {} for k in self._acts_by(own)})
        like = True
        for models in self.others:
            for v, m in models.items():
                kind = m.kind(narrow=True)
                if kind in ('map', 'offsets', 'drop'):
                    key = (r(v), kind, m.drop_dir() if kind == 'drop' else None)
                    out[('other_click',) + key] = {}
                    like &= key in own
        players = self._players()
        if players:
            out[('players', players)] = {}
            if like and own:
                out[('others_like_me',)] = {}
        lines = {}
        for name, space, events in (('line_goal', self.line_goals, self.line_events),
                                    ('line_loss', self.line_losses, self.loss_events)):
            if events and space:
                for v, k, d in space:
                    p = out.setdefault((name, r(v), d), {'k': [], 'full': True})
                    p['k'].append(k)
                    p['full'] &= d is not None and k == self._extent(d)
                lines[name] = {(r(v), k, d) for v, k, d in space}
        if len(lines) == 2:
            goal = {k for rv, k, d in lines['line_goal'] if rv == 'mine' and d is None}
            loss = {k for rv, k, d in lines['line_loss'] if rv == 'theirs' and d is None}
            if goal & loss:
                out[('loss_like_goal',)] = {}
        for p in out.values():
            if 'k' in p:
                p['k'] = sorted(set(p['k']))
        for v in self.absent_goals:
            m = self.click_models.get(v)
            if m is not None and m.kind() == 'map' and m.table.get(v) == self.bg:
                out[('absent_goal', 'cleared')] = {}
            elif any(v in (mm.toggles() or ()) for mm in self.click_models.values()):
                out[('absent_goal', 'toggled')] = {}
            else:
                out[('absent_goal', r(v))] = {}
        for pat in self.win_contacts:
            out[('contact_goal', self.contact_kinds.get(pat, 'unique'))] = {}
        if self.shape_goal:
            out[('shape_goal',)] = {}
        for i, c in self.drift.items():
            d, n = c.most_common(1)[0]
            if d in SIDES and n >= 2:
                out[('drift', i, d)] = {}
        for a, c in self.transform_ops.items():
            (op, offset), n = c.most_common(1)[0]
            if n >= 2:
                out[('transform', a, op, offset)] = {}
        if self.ce_generations:
            out[('drift_weights',)] = {'mean': self.ce_mean.tolist(), 'std': self.ce_std.tolist()}
        return out

    def _acts_by(self, own) -> set:
        return {('acts_by', how) for how, yes in (('moves', bool(self.directions)), ('clicks', bool(own))) if yes}

    def _extent(self, d) -> Optional[int]:
        """The most cells a line along d can have in the last grid"""
        if self.shape is None or d is None:
            return None
        return min(n for n, step in zip(self.shape, d) if step)

    def conditions(self) -> set:
        """The facts a generic pattern's conditions are checked against: how the agent acts, players, drift, own clicks"""
        own = self._click_keys(self.roles())
        out = {('own_click',) + k for k in own} | self._acts_by(own)
        players = self._players()
        if players:
            out.add(('players', players))
        for i, c in self.drift.items():
            d, n = c.most_common(1)[0]
            if d in SIDES and n >= 2:
                out.add(('drift', i, d))
        return out

    def use_generic(self, patterns):
        """
        Generic patterns (pattern_store) to try in this game. Score weights start from the pattern's
        (unless this game's own were learned); the others are used as described in _apply_priors.
        """
        self.generic = {p['key']: p for p in patterns}
        w = self.generic.get(('drift_weights',))
        if w and not self.ce_generations:
            self.ce_mean = np.array(w['params']['mean'], dtype=float)
            self.ce_std = np.maximum(np.array(w['params']['std'], dtype=float), CE_MIN_STD)
            self.weights = self.ce_mean.copy()
            self._generic_weights = True

    def _apply_priors(self, grid):
        """
        Generic patterns used while this game's own evidence is missing (each once its conditions, the
        pattern's 'context', hold here):
        - line goals (losses), while no win (loss) has formed lines: a line of the role's values; its
          length k differs between games (the pattern lists those seen), so the shortest, MIN_LINE, is
          tried first, and when a line of k forms without a win (loss), k + 1 (up to the board's size);
          if k was the grid's extent along the line in every game (a full row), that extent is tried. With loss_like_goal, goals and losses share k, and once the
          goal is known from wins, losses are taken to be its lines for the other players
        - others_like_me: the other players' moves are taken to drop like the agent's clicks, as soon as
          their moves fit that (rather than after 3 moves)
        - directions: an action not seen to move the agent yet is taken to move it as the pattern says
          (_dir), until it is seen to do otherwise (_check_prior_direction); not if a no_direction
          or a transform pattern for the action applies too, or the action was seen to turn the agent
          here (an action the agent believes moves it may never be tried as a turn)
        - contact goals, while no goal is known: the objects of the kind the pattern names ('unique':
          the only object of its pattern in the grid; 'repeated') are tried as goals, nearest first,
          among those unchanged since they were first tried in the level; one touched without a win is
          not tried again
        (drift and transforms need one observation instead of two when a generic pattern names them; see
        _drift_dir and _geometric)
        """
        usable = {}
        if self.generic:
            cond = self.conditions()
            usable = {k: p for k, p in self.generic.items() if p.get('context') and set(p['context']) <= cond}
        meta = self._meta_usable(grid, usable)
        usable.update(meta)
        if not usable:
            return
        mirror = ('loss_like_goal',) in usable
        for name, events in (('line_goal', self.line_events), ('line_loss', self.loss_events)):
            if events:
                continue
            for key, p in usable.items():
                if key[0] != name or key in self.prior_lines or not self._role_values(key[1]):
                    continue
                full = p['params'].get('full') and key[2] is not None
                self.prior_lines[key] = {'k': self._extent(key[2]) if full else MIN_LINE, 'falsified': False}
                self.priors_applied.add(key)
        for state in self.prior_lines.values():
            if state['falsified']:
                state['k'] += 1
                state['falsified'] = False
        goal_keys = [k for k in self.prior_lines if k[0] == 'line_goal']
        loss_keys = [k for k in self.prior_lines if k[0] == 'line_loss']
        if mirror and loss_keys and not self.loss_events:
            self.priors_applied.add(('loss_like_goal',))
            if self.line_events and self.line_goals:
                mine = self._role_values('mine')
                ks = [k for v, k, d in self.line_goals if v in mine]
                if ks:
                    for key in loss_keys:
                        self.prior_lines[key]['k'] = max(self.prior_lines[key]['k'], min(ks))
            elif goal_keys:
                k = max(self.prior_lines[key]['k'] for key in goal_keys + loss_keys)
                for key in goal_keys + loss_keys:
                    self.prior_lines[key]['k'] = k
        size = max(grid.shape)
        for name, events in (('line_goal', self.line_events), ('line_loss', self.loss_events)):
            if events:
                continue
            hyps = set()
            for key, state in self.prior_lines.items():
                if key[0] == name and state['k'] <= size:
                    for v in self._role_values(key[1]):
                        hyps.add((v, state['k'], key[2]))
                        self._prior_hyps[(v, state['k'], key[2])] = key
            if name == 'line_goal':
                self.line_goals = hyps or None
            else:
                self.line_losses = hyps or None
        dirs: Dict[object, set] = {}
        for key in usable:
            if key[0] == 'direction':
                dirs.setdefault(key[1], set()).add(key[2])
        self.prior_directions = {a: ds.pop() for a, ds in dirs.items()
                                 if len(ds) == 1 and a not in self.directions and a not in self.directions_rejected
                                 and ('no_direction', a) not in usable and a not in self.transform_ops
                                 and not any(k[0] == 'transform' and k[1] == a for k in usable)}
        kinds = {key[1] for key in usable if key[0] == 'contact_goal'}
        inside = any(key[0] == 'contact_goal' and key[2:] == ('inside',) for key in usable)
        self.prior_contacts = set()
        if kinds and self.agent is not None and not (self.win_contacts or self.shape_goal or self.absent_goals
                                                     or self.line_events):
            objs = [(o, pattern(grid, o)) for o in self._components(grid, self.agent)]
            if self._level_first is None or self._level_first[0].shape != grid.shape:
                self._level_first = (grid.copy(), self.agent)
            counts = Counter(p for _, p in objs)
            # (only objects unchanged since then: not, for example, a bar that shrinks as actions are used)
            edge = lambda o: any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in o)
            self.prior_contacts = {p for o, p in objs if self._unchanged(grid, o) and p not in self.contact_rejected
                                   and ('unique' if counts[p] == 1 else 'repeated') in kinds
                                   and not (inside and edge(o))}
            for kind in kinds:
                if any(('unique' if n == 1 else 'repeated') == kind for p, n in counts.items()
                       if p in self.prior_contacts):
                    self.priors_applied.add(('contact_goal', kind) + (('inside',) if inside else ()))
        if ('others_like_me',) in usable:
            drops = {m.drop_dir() for m in self.click_models.values() if m.kind() == 'drop'}
            if len(drops) == 1:
                d = drops.pop()
                for models in self.others:
                    for m in models.values():
                        m.like = d
                self.priors_applied.add(('others_like_me',))
        for key, element in meta.items():
            if key in self.priors_applied and key not in self.generic:
                self.priors_applied.discard(key)
                self.meta_applied.add(element['element'])

    def use_meta(self, elements: Dict[str, dict]):
        """Universal elements (meta_patterns.MetaStore.elements) to use where nothing more specific applies"""
        self.meta = elements
        g = elements.get('gravity')
        if g:
            ds = Counter(tuple(p['d']) for p in g['params'] if p.get('d'))
            self.meta_gravity = ds.most_common(1)[0][0] if ds else (1, 0)
        ops = set()
        if 'rotation' in elements:
            ops |= set(TURNS)
        if 'reflection' in elements:
            ops |= set(MIRRORS)
        self.meta_ops = ops

    def _meta_usable(self, grid, usable) -> Dict[tuple, dict]:
        """
        Generic-like patterns made from the universal elements, for what no usable generic pattern covers
        (see meta_patterns): alignment, in a game where moves place marks on empty cells (lines of the
        agent's marks win, of the other players' lose, sharing their length) or pieces come to rest
        (lines of any cells across the direction of gravity, full rows if a full row was an instance);
        reach, while the agent is known (the only object of its pattern, not touching the grid's edge). Gravity, rotation and reflection
        act through the click models, _drift_dir and _geometric; shape_match through prior_shape.
        """
        out = {}
        if not self.meta:
            return out
        if 'alignment' in self.meta and not any(k[0] in ('line_goal', 'line_loss') for k in usable):
            mine, empty = self._placed([self.click_models])
            theirs, _ = self._placed(self.others)
            el = {'element': 'alignment', 'context': [], 'params': {'k': [], 'full': False}}
            if mine - {self.bg} and empty:
                out[('line_goal', 'mine', None)] = el
                if theirs - mine - {self.bg}:
                    out[('line_loss', 'theirs', None)] = el
                    out[('loss_like_goal',)] = el
            elif self._drift_dir(1) is not None and any(p.get('full') and p.get('role') == 'any'
                                                        for p in self.meta['alignment']['params']):
                # (only a full row: where every piece forms short lines, lines of an unknown length
                # mislead the placement while the length is being found)
                d = self._drift_dir(1)
                out[('line_goal', 'any', (abs(d[1]), abs(d[0])))] = dict(el, params={'k': [], 'full': True})
        if 'reach' in self.meta and not any(k[0] == 'contact_goal' for k in usable):
            out[('contact_goal', 'unique', 'inside')] = {'element': 'reach', 'context': [], 'params': {}}
        if 'shape_match' in self.meta:
            grew = any('merge' in c for c in self.rules.values())
            self.prior_shape = grew and not self.shape_rejected and not (
                self.win_contacts or self.shape_goal or self.absent_goals or self.line_events)
            if self.prior_shape:
                self.meta_applied.add('shape_match')
        if self.meta_gravity is not None:
            for models in [self.click_models] + self.others:
                for m in models.values():
                    if m.like is None:
                        m.like = self.meta_gravity
                    if m.like == self.meta_gravity and m.kind(narrow=models is not self.click_models) == 'drop':
                        self.meta_applied.add('gravity')
        return out

    def _unchanged(self, grid, obj) -> bool:
        """Whether the object's cells and those around it hold what they held when first seen (the agent aside)"""
        first, first_agent = self._level_first
        skip = set(self.agent or ()) | set(first_agent or ())
        around = {(p[0] + dr, p[1] + dc) for p in obj for dr, dc in SIDES} | set(obj)
        return all(first[p] == grid[p] for p in around if _inside(grid, p) and p not in skip)

    STATE = ('bg', 'start_patterns', 'directions', 'rules', 'value_rules', 'pair_rules', 'win_contacts',
             'shape_goal', 'seen_patterns', 'click_models', 'absent_goals', 'absent_rejected', 'others',
             'line_goals', 'line_losses', 'line_hits', 'line_misses', 'loss_hits', 'loss_misses',
             'line_events', 'loss_events', 'transforms', 'transform_ops', 'drift', 'agent_patterns',
             'ce_mean', 'ce_std', 'ce_done', 'ce_generations', 'wins', 'losses', 'shape', 'contact_kinds',
             'contact_rejected', 'directions_rejected')

    def export_state(self) -> dict:
        """
        What the agent learned of a game (STATE: rules, models, goals, weights), to go on with the game
        later; the exact transitions remembered, and what holds only for the current level, are left out
        """
        return {name: getattr(self, name) for name in self.STATE}

    def import_state(self, state: dict):
        for name, value in state.items():
            if name in self.STATE:
                setattr(self, name, value)
        self.weights = self.ce_mean.copy()

    # ---------------------------------------------------------- planning
    # ------------------------------------------------------ other players
    def _others_known(self) -> bool:
        return any(m.kind(narrow=True) in ('map', 'offsets', 'drop') for d in self.others for m in d.values())

    def _other_moves(self, i, grid, limit=False) -> List[np.ndarray]:
        out, seen = [], set()
        for v, m in self.others[i].items():
            kind = m.kind(narrow=True)
            if kind not in ('map', 'offsets', 'drop'):
                continue
            done = set()
            cells = [(int(r), int(c)) for r, c in np.argwhere(grid == v)]
            if limit and kind != 'drop':
                cells = self._closest(grid, cells)
            for p in cells:
                if p in done:
                    continue
                obj = self._clicked(grid, p)
                if kind == 'map':
                    done |= obj
                g = m.predict(grid, p, obj, narrow=True)
                if g is not None and not np.array_equal(g, grid) and g.tobytes() not in seen:
                    seen.add(g.tobytes())
                    out.append(g)
        return out

    def _placed(self, groups=None) -> Tuple[set, set]:
        """
        (values moves place, values they replace): from the agent's clicks and the other players' moves
        (or the groups of click models given)
        """
        marks, empties = set(), set()
        for models in groups if groups is not None else [self.click_models] + self.others:
            for v, m in models.items():
                kind = m.kind()
                if kind == 'drop' or (kind is None and m.drop_dir() is not None):
                    empties.add(m.drop_values[0])
                    marks.add(m.drop_values[1])
                elif kind == 'map' and m.table.get(v, v) != v:
                    empties.add(v)
                    marks.add(m.table[v])
                elif kind == 'offsets' and m.off_tables.get((0, 0), {}).get(v, v) != v:
                    empties.add(v)
                    marks.add(m.off_tables[(0, 0)][v])
        return marks, empties - marks

    def _closest(self, grid, cells, cell_of=lambda p: p):
        """At most MAX_OPTIONS of the cells: those next to a mark (cells moves have marked), else nearest the centre"""
        if len(cells) <= MAX_OPTIONS:
            return cells
        marks, _ = self._placed()
        placed = np.isin(grid, list(marks)) if marks else np.zeros(grid.shape, bool)
        centre = ((grid.shape[0] - 1) / 2, (grid.shape[1] - 1) / 2)

        def closeness(x):
            r, c = cell_of(x)
            touch = placed[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any()
            return (0 if touch else 1, abs(r - centre[0]) + abs(c - centre[1]))
        return sorted(cells, key=closeness)[:MAX_OPTIONS]

    def _search_actions(self, g, actions) -> List:
        """The agent's moves considered in a search position: its moves, and at most MAX_OPTIONS clicks"""
        key = g.tobytes()
        if key not in self._action_cache:
            acts = self._actions(g, actions)
            moves = [a for a in acts if not isinstance(a, tuple)]
            clicks = [a for a in acts if isinstance(a, tuple)]

            def cell_of(a):
                m = self.click_models.get(int(g[a[1:]]))
                if m is not None and m.kind() == 'drop':
                    q = m.drop_target(g, a[1:], m.drop_dir(), m.drop_values[0])
                    return q if q is not None else a[1:]
                return a[1:]
            self._action_cache[key] = moves + self._closest(g, clicks, cell_of)
        return self._action_cache[key]

    def _eval(self, grid) -> float:
        """
        Progress toward the learned lines, in (-0.5, 0.5): each window of a goal line holding only its
        value and cells moves replace counts 4^(its value's cells) / 4^k; the opponents' (loss lines) count
        against
        """
        _, empties = self._placed()
        score = 0.0
        for sign, lines in ((1, self.line_goals or ()), (-1, self.line_losses or ())):
            for v, k, d in lines:
                if v == ANY:
                    continue
                w = self._windows(grid.shape, k, d)
                if not len(w):
                    continue
                vals = grid.ravel()[w]
                mine = vals == v
                open_ = (mine | np.isin(vals, list(empties))).all(axis=1)
                score += sign * float((4.0 ** mine.sum(axis=1)[open_]).sum()) / 4.0 ** k
        return 0.5 * float(np.tanh(score))

    def _search(self, grid, actions):
        """
        (value, best actions): iterative deepening of a depth-limited search (alpha-beta, all other players
        against the agent), each depth with a budget of SEARCH_NODES positions; the deepest depth finished
        is used
        """
        result = None
        self._action_cache = {}
        for depth in range(1, MAX_DEPTH + 1):
            self._nodes = SEARCH_NODES
            value, acts = self._ours(grid, self.agent, depth, -2.0, 2.0, actions, root=True)
            if self._nodes < 0 and result is not None:
                break
            result = (value, acts)
            if abs(value) >= 0.99 * DISCOUNT ** (depth - 1) and value != 0:
                break                           # a forced win or loss is found
        return result

    def _ours(self, g, agent, depth, alpha, beta, actions, root=False):
        options = []
        seen = set()
        for a in self._search_actions(g, actions):
            nxt = self.predict(g, agent, a)
            if nxt is None or nxt[0].shape != g.shape or np.array_equal(nxt[0], g) or nxt[0].tobytes() in seen:
                continue
            seen.add(nxt[0].tobytes())
            options.append((a, nxt))
        if not options:
            return 0.0, []
        scored = []
        for a, nxt in options:
            if self.wins_with(g, agent, a, nxt):
                scored.append((1.0, a, nxt, True))
            elif self.line_losses and self._formed(g, nxt[0], self.line_losses) > 0:
                scored.append((-1.0, a, nxt, True))
            else:
                scored.append((self._eval(nxt[0]), a, nxt, False))
        scored.sort(key=lambda t: -t[0])
        best, acts = None, []
        for value, a, nxt, final in scored:
            self._nodes -= 1
            if not final and depth > 1:
                value = DISCOUNT * self._theirs(nxt[0], nxt[1], 0, depth, alpha, beta, actions)
            if best is None or value > best:
                best, acts = value, [a]
            elif value == best:
                acts.append(a)
            alpha = max(alpha, value)
            if not root and (alpha >= beta or self._nodes < 0):
                break
        return best, acts

    def _theirs(self, g, agent, i, depth, alpha, beta, actions):
        if i == len(self.others):
            return self._ours(g, agent, depth - 1, alpha, beta, actions)[0]
        moves = self._other_moves(i, g, limit=True)[:MAX_OPTIONS]
        if not moves:
            return self._theirs(g, agent, i + 1, depth, alpha, beta, actions)
        scored = []
        for m in moves:
            if self.line_losses and self._formed(g, m, self.line_losses) > 0:
                scored.append((-1.0, m, True))
            else:
                scored.append((self._eval(m), m, False))
        scored.sort(key=lambda t: t[0])
        worst = None
        for value, m, final in scored:
            self._nodes -= 1
            if not final:
                value = self._theirs(m, agent, i + 1, depth, alpha, beta, actions)
            worst = value if worst is None else min(worst, value)
            beta = min(beta, value)
            if alpha >= beta or self._nodes < 0:
                break
        return worst

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
        if self._generic_weights:
            self.priors_applied.add(('drift_weights',))
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
                self.ce_generations += 1
        self.weights = np.array([self.rng.gauss(m, s) for m, s in zip(self.ce_mean, self.ce_std)])
        self.episode = {'score': 0, 'actions': 0, 'used': False}

    def make_plan(self, grid, actions) -> Optional[List]:
        self._apply_priors(grid)
        if self._drift_dir(1) is not None and self.agent is not None:
            plan = self._drift_plan(grid, actions)
            if plan:
                return plan
        if self._others_known():
            # explore what is not known first (a value not clicked yet), then play the minimax move
            unknown = [a for a in self._actions(grid, actions) if self.predict(grid, self.agent, a) is None]
            value, best = self._search(grid, actions)
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
        targets = self._targets(grid, self.agent) if self.shape_goal or self.prior_shape else None
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
                                and not trials and not self.prior_contacts and not self.prior_shape:
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
        self.shape = grid.shape
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
        """The direction the agent drifts in at frame i: seen twice (once, if a generic pattern says it)"""
        c = self.drift.get(i)
        if not c:
            return None
        d, n = c.most_common(1)[0]
        if d not in SIDES:
            return None
        if n == 1 and ('drift', i, d) in self.generic:
            self.priors_applied.add(('drift', i, d))
            return d
        if n == 1 and d == self.meta_gravity:
            self.meta_applied.add('gravity')
            return d
        return d if n >= 2 else None

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

    def _contact_kind(self, grid, obj, agent) -> str:
        """'unique' if no other object in the grid has the object's pattern, else 'repeated'"""
        pat = pattern(grid, obj)
        n = sum(pattern(grid, o) == pat for o in self._components(grid, agent or frozenset()))
        return 'unique' if n <= 1 else 'repeated'

    def _contact_falsify(self, before, action):
        """A contact goal (or a pattern tried as one) touched without a win is not the goal"""
        d = self._dir(action)
        if self.agent is not None and d is not None:
            touched = {pattern(before, o) for o in self._contacts(before, self.agent, d) or []}
            self.win_contacts -= touched
            self.contact_rejected |= touched
            self.prior_contacts -= touched

    def after_step(self, before, action, after, frames=None, lost=False):
        """lost: the step lost the game (lost() follows); its lines are no evidence against loss hypotheses"""
        if self.prior_shape and self.agent is not None:
            nxt = self.predict(before, self.agent, action)
            if nxt is not None and nxt[1] is not None and len(nxt[1]) != len(self.agent) and \
                    np.array_equal(nxt[0], after) and self._shape_match(*nxt):
                self.shape_rejected = True          # the shapes matched and the level did not end
                self.prior_shape = False
        self._track_base([before] + list(frames or [after]))
        self._falsify(before, frames or [after], lost=lost)
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
