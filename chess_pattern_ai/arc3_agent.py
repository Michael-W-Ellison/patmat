#!/usr/bin/env python3
"""
ARC-AGI-3 games (interactive, turn-based, 64x64 frames) for the effect learner

The games are run through the ARC-AGI toolkit (package arc_agi; games run
locally from game files, or remotely through the API). Unlike the step 3
puzzles, the learner is not told the goal: it sees frames, the actions
available, and when a level is completed or the game is lost.

- cell_grid(frame): the game's cell grid. A game draws each cell as a k x k
  block of pixels (camera upscaling), possibly with a letterbox or a status
  bar drawn over the outer edge. The scale and offset chosen are the largest
  k (and its offset) at which every block away from the outer ring of blocks
  holds one value; each block becomes the value most of its pixels hold.
- Arc3Agent: the effect learner (effect_learner.py) with
  - goal learning: when an action completes a level, the agent model's key
    for that move (the values at the agent and the cells ahead of it, k=2)
    is recorded as a winning move; plans then search for a state where a
    winning move can be made, as well as for unexplored actions
  - transitions that complete a level or lose the game are not learned as
    effects (the next frame is another level or the same level restarted)
  - after a loss the agent resets the level and drops its plan.
Clicks (ACTION6) are not used yet.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from effect_learner import EffectLearner

SIMPLE_ACTIONS = (1, 2, 3, 4, 5)
MAX_SCALE = 16
MARK_TOLERANCE = 0.1
MIN_REFINED = 3
RING_GAIN = 0.25
HISTORY = 8
WIN_KEY_CELLS = 2


def _scale_fit(frame: np.ndarray, k: int, oy: int, ox: int, tolerance: float = 0.0) -> Optional[np.ndarray]:
    """
    The k x k blocks of the frame from offset (oy, ox) if the blocks off the
    outer ring are uniform (all but a share `tolerance` of them)
    """
    h, w = frame.shape
    rows, cols = (h - oy) // k, (w - ox) // k
    if rows < 5 or cols < 5:
        return None             # (at least 3 x 3 blocks off the ring, so a plain background does not pass)
    blocks = frame[oy:oy + rows * k, ox:ox + cols * k].reshape(rows, k, cols, k).swapaxes(1, 2).reshape(rows, cols, k * k)
    inner = blocks[1:-1, 1:-1]
    mixed = (inner != inner[..., :1]).any(axis=2)
    if mixed.sum() > tolerance * mixed.size:
        return None
    return blocks


def _modes(blocks: np.ndarray) -> np.ndarray:
    out = np.empty(blocks.shape[:2], dtype=int)
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1]):
            vals, counts = np.unique(blocks[r, c], return_counts=True)
            out[r, c] = vals[np.argmax(counts)]
    return out


def _ring_mixed(blocks) -> float:
    """The share of the blocks on the outer ring that are not uniform"""
    rows, cols = blocks.shape[:2]
    ring = np.ones((rows, cols), bool)
    ring[1:-1, 1:-1] = False
    mixed = (blocks != blocks[..., :1]).any(axis=2)
    return float(mixed[ring].mean())


def find_scale(frame: np.ndarray, also=()) -> Tuple[int, int, int]:
    """
    (k, oy, ox): a block size (and offset) at which the frame (and each of the frames in `also`) is
    made of uniform blocks off the outer ring. The largest such k is taken, unless a smaller one (of at
    least MIN_REFINED) leaves at least RING_GAIN less of its ring mixed (a large k can hide content in
    its ring, as on an empty board); for each k, the offset with the least mixed ring.
    """
    fits = {}
    for k in range(MAX_SCALE, 1, -1):
        for oy in range(k):
            for ox in range(k):
                blocks = _scale_fit(frame, k, oy, ox)
                if blocks is not None and all(_scale_fit(f, k, oy, ox) is not None for f in also):
                    mixed = _ring_mixed(blocks)
                    if k not in fits or mixed < fits[k][0]:
                        fits[k] = (mixed, oy, ox)
        if k in fits and fits[k][0] == 0:
            break                               # (nothing smaller can do better)
    if not fits:
        return 1, 0, 0
    best = max(fits)
    for k in sorted(fits, reverse=True):
        if k >= MIN_REFINED and fits[k][0] <= fits[best][0] - RING_GAIN:
            best = k
    return best, fits[best][1], fits[best][2]


def cell_grid(frame: np.ndarray, scale: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    k, oy, ox = scale or find_scale(frame)
    blocks = _scale_fit(frame, k, oy, ox, tolerance=1.0)
    if blocks is None:
        return np.asarray(frame, dtype=int)
    return _modes(blocks)


class GridReader:
    """
    Cell grids for one game: the scale is kept while a frame fits it, else
    found again (a new level). A frame that fits only with MARK_TOLERANCE
    (all but that share of the blocks off the ring uniform) is either one
    with marks drawn at the pixel level over a few cells, or one whose
    first frames looked uniform at too large a scale (an empty board): if
    a scale of at least MIN_REFINED fits this frame and the frames read at
    the current scale exactly, it is taken, else the current one is kept.
    The same is tried for any change in the first HISTORY frames at a scale
    (a change in the ring of blocks, which is not checked, may be content).
    """

    def __init__(self):
        self.scale: Optional[Tuple[int, int, int]] = None
        self.history: List[np.ndarray] = []         # frames read at the current scale (the last HISTORY)

    def __call__(self, frame) -> np.ndarray:
        frame = np.asarray(frame)
        if self.scale is None:
            self._rescale(find_scale(frame))
        elif _scale_fit(frame, *self.scale, tolerance=MARK_TOLERANCE) is None:
            self._rescale(find_scale(frame))
        elif _scale_fit(frame, *self.scale) is None or (
                len(self.history) < HISTORY and not np.array_equal(frame, self.history[-1])):
            finer = find_scale(frame, self.history)
            if MIN_REFINED <= finer[0] < self.scale[0]:
                self._rescale(finer)
        self.history = (self.history + [frame])[-HISTORY:]
        return cell_grid(frame, self.scale)

    def _rescale(self, scale):
        self.scale = scale
        self.history = []

    def pixel(self, cell) -> Tuple[int, int]:
        """(x, y) display coordinates of the centre of a cell of the last grid (for ACTION6)"""
        k, oy, ox = self.scale or (1, 0, 0)
        return ox + cell[1] * k + k // 2, oy + cell[0] * k + k // 2


class Arc3Agent(EffectLearner):
    def __init__(self, rng: Optional[random.Random] = None):
        super().__init__(goal_absent=(), rng=rng)
        self.win_keys: set = set()
        self.wins = self.losses = 0

    # the goal is unknown until a level is won: no state counts as the goal by itself
    def _goal(self, grid) -> bool:
        return False

    def _win_key(self, grid, action) -> Optional[tuple]:
        agent, d = self._agent(grid), self.directions.get(action)
        if agent is None or d is None:
            return None
        cells = [(agent[0] + i * d[0], agent[1] + i * d[1]) for i in range(WIN_KEY_CELLS)]
        return tuple(int(grid[p]) if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] else -9
                     for p in cells)

    def wins_with(self, grid, action) -> bool:
        return self._win_key(grid, action) in self.win_keys

    def level_won(self, grid, action):
        """The action completed the level from this grid"""
        self.wins += 1
        key = self._win_key(grid, action)
        if key is not None:
            self.win_keys.add(key)
        self.plan = []

    def lost(self, grid=None, action=None):
        self.losses += 1
        self.plan = []

    def describe(self) -> str:
        return (f'values {self.agent_values}, directions {self.directions}, '
                f'model {getattr(self.model(), "name", None)}, winning moves {sorted(self.win_keys)}')
