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
from typing import Dict, Optional, Tuple

import numpy as np

from effect_learner import EffectLearner

SIMPLE_ACTIONS = (1, 2, 3, 4, 5)
MAX_SCALE = 16
WIN_KEY_CELLS = 2


def _scale_fit(frame: np.ndarray, k: int, oy: int, ox: int) -> Optional[np.ndarray]:
    """The k x k blocks of the frame from offset (oy, ox) if every block off the outer ring is uniform"""
    h, w = frame.shape
    rows, cols = (h - oy) // k, (w - ox) // k
    if rows < 5 or cols < 5:
        return None             # (at least 3 x 3 blocks off the ring, so a plain background does not pass)
    blocks = frame[oy:oy + rows * k, ox:ox + cols * k].reshape(rows, k, cols, k).swapaxes(1, 2).reshape(rows, cols, k * k)
    inner = blocks[1:-1, 1:-1]
    if (inner != inner[..., :1]).any():
        return None
    return blocks


def _modes(blocks: np.ndarray) -> np.ndarray:
    out = np.empty(blocks.shape[:2], dtype=int)
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1]):
            vals, counts = np.unique(blocks[r, c], return_counts=True)
            out[r, c] = vals[np.argmax(counts)]
    return out


def find_scale(frame: np.ndarray) -> Tuple[int, int, int]:
    """(k, oy, ox): the largest block size (and offset) at which the frame is made of uniform blocks"""
    for k in range(MAX_SCALE, 1, -1):
        for oy in range(k):
            for ox in range(k):
                if _scale_fit(frame, k, oy, ox) is not None:
                    return k, oy, ox
    return 1, 0, 0


def cell_grid(frame: np.ndarray, scale: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    k, oy, ox = scale or find_scale(frame)
    blocks = _scale_fit(frame, k, oy, ox)
    if blocks is None:
        return np.asarray(frame, dtype=int)
    return _modes(blocks)


class GridReader:
    """Cell grids for one game: the scale is found again when a frame no longer fits it (a new level)"""

    def __init__(self):
        self.scale: Optional[Tuple[int, int, int]] = None

    def __call__(self, frame) -> np.ndarray:
        frame = np.asarray(frame)
        if self.scale is None or _scale_fit(frame, *self.scale) is None:
            self.scale = find_scale(frame)
        return cell_grid(frame, self.scale)


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

    def lost(self):
        self.losses += 1
        self.plan = []
