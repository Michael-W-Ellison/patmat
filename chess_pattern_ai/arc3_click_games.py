#!/usr/bin/env python3
"""
Small click games (ACTION6) for the ARC-AGI-3 agents, written for this
project with the ARC engine (package arcengine): the public example games
have no click actions. The goal is not shown to the player.

- LightsOut (3 levels, 5x5): clicking a cell toggles it and its four
  neighbours between off (1) and on (11); a level is won when no cell is
  on. Levels are made by random clicks from all off, so each is solvable.
- ClearRed (3 levels, 8x8): background 0, walls 5, red objects (2) and
  blue objects (1). Clicking a red object removes all of it, clicking
  anything else does nothing; a level is won when no red is left.
- TicTacToe (3 levels, a 3x3 board inside a border of 5): the player
  clicks an empty cell (0) to place its mark (2); the opponent then places
  its mark (1), shown as a second frame of the same action. The opponent
  completes three in a row if it can, else blocks the player's three,
  else picks a random empty cell. Three of the player's marks in a row win
  the level (the next level is a new board); three of the opponent's lose
  the game; a full board with no line is cleared and played again.

GAMES maps names to game classes; local_env(game) gives the reset/step
interface of the ARC-AGI toolkit's environment wrappers.
"""

import random

import numpy as np
from arcengine import ARCBaseGame, ActionInput, Camera, GameAction, Level, Sprite

OFF, ON = 1, 11
LIGHTS_SIDES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _lights_level(seed: int, n: int, presses: int) -> Level:
    rng = random.Random(seed)
    board = np.full((n, n), OFF, dtype=np.int8)
    for _ in range(presses):
        r, c = rng.randrange(n), rng.randrange(n)
        for dr, dc in LIGHTS_SIDES:
            if 0 <= r + dr < n and 0 <= c + dc < n:
                board[r + dr, c + dc] = ON if board[r + dr, c + dc] == OFF else OFF
    if not (board == ON).any():
        board[n // 2, n // 2] = ON          # (the presses cancelled out: one press on the centre instead)
        for dr, dc in LIGHTS_SIDES[1:]:
            board[n // 2 + dr, n // 2 + dc] = ON
    return Level(sprites=[Sprite(pixels=board, name='board')], grid_size=(n, n))


class LightsOut(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_lights_level(seed * 10 + i, 5, p) for i, p in enumerate((3, 5, 7))]
        super().__init__(game_id='lights', levels=levels, camera=Camera(width=5, height=5, background=OFF,
                                                                         letter_box=3))

    def step(self):
        if self.action.id == GameAction.ACTION6:
            at = self.camera.display_to_grid(self.action.data.get('x', 0), self.action.data.get('y', 0))
            if at is not None:
                board = self.current_level.get_sprites_by_name('board')[0]
                c, r = at
                px = board.pixels.copy()
                for dr, dc in LIGHTS_SIDES:
                    if 0 <= r + dr < px.shape[0] and 0 <= c + dc < px.shape[1]:
                        px[r + dr, c + dc] = ON if px[r + dr, c + dc] == OFF else OFF
                board.pixels = px
                if not (px == ON).any():
                    self.next_level()
        self.complete_action()


CLEAR_LEVELS = [
    ['55555555',
     '50000005',
     '50220005',
     '50200105',
     '50000115',
     '50002005',
     '50000005',
     '55555555'],
    ['55555555',
     '52200015',
     '50000015',
     '50111005',
     '50000225',
     '50220025',
     '50000005',
     '55555555'],
    ['55555555',
     '50002205',
     '51100205',
     '50000005',
     '52021105',
     '52020005',
     '50000225',
     '55555555'],
]


def _clear_level(rows) -> Level:
    grid = np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)
    sprites = [Sprite(pixels=np.where(grid == 5, 5, -1).astype(np.int8), name='walls', layer=-1)]
    seen = np.zeros(grid.shape, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = int(grid[r, c])
            if v in (1, 2) and not seen[r, c]:
                cells, stack = [], [(r, c)]
                seen[r, c] = True
                while stack:
                    p = stack.pop()
                    cells.append(p)
                    for dr, dc in LIGHTS_SIDES[1:]:
                        q = (p[0] + dr, p[1] + dc)
                        if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and not seen[q] and grid[q] == v:
                            seen[q] = True
                            stack.append(q)
                r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
                h, w = max(p[0] for p in cells) - r0 + 1, max(p[1] for p in cells) - c0 + 1
                px = np.full((h, w), -1, dtype=np.int8)
                for p in cells:
                    px[p[0] - r0, p[1] - c0] = v
                sprites.append(Sprite(pixels=px, name='red' if v == 2 else 'blue').set_position(c0, r0))
    return Level(sprites=sprites, grid_size=grid.shape[::-1])


class ClearRed(ARCBaseGame):
    def __init__(self):
        super().__init__(game_id='clear', levels=[_clear_level(rows) for rows in CLEAR_LEVELS],
                         camera=Camera(width=8, height=8, background=0, letter_box=3))

    def step(self):
        if self.action.id == GameAction.ACTION6:
            at = self.camera.display_to_grid(self.action.data.get('x', 0), self.action.data.get('y', 0))
            if at is not None:
                hit = self.current_level.get_sprite_at(*at)
                if hit is not None and hit.name == 'red':
                    self.current_level.remove_sprite(hit)
                    if not self.current_level.get_sprites_by_name('red'):
                        self.next_level()
        self.complete_action()


EMPTY, MINE, THEIRS, BORDER = 0, 2, 1, 5
LINES = [[(r, c) for c in range(3)] for r in range(3)] + [[(r, c) for r in range(3)] for c in range(3)] + \
        [[(i, i) for i in range(3)], [(i, 2 - i) for i in range(3)]]


def _line(board, v) -> bool:
    return any(all(board[p] == v for p in line) for line in LINES)


def _ttt_level() -> Level:
    px = np.full((5, 5), BORDER, dtype=np.int8)
    px[1:4, 1:4] = EMPTY
    return Level(sprites=[Sprite(pixels=px, name='board')], grid_size=(5, 5))


class TicTacToe(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._reply = False
        super().__init__(game_id='tictactoe', levels=[_ttt_level() for _ in range(3)],
                         camera=Camera(width=5, height=5, background=EMPTY, letter_box=3))

    def _board(self):
        return self.current_level.get_sprites_by_name('board')[0]

    def _opponent_move(self, b):
        """b: the 3x3 board; the opponent's cell (win, else block, else random)"""
        empty = [(r, c) for r in range(3) for c in range(3) if b[r, c] == EMPTY]
        for v in (THEIRS, MINE):
            for p in empty:
                trial = b.copy()
                trial[p] = v
                if _line(trial, v):
                    return p
        return self.rng.choice(empty)

    def step(self):
        sprite = self._board()
        px = sprite.pixels.copy()
        b = px[1:4, 1:4]
        if self._reply:                         # the second frame of a move: the opponent's turn
            self._reply = False
            if not (b == EMPTY).any():
                b[:] = EMPTY                    # a draw: the board is cleared
            else:
                b[self._opponent_move(b)] = THEIRS
                if _line(b, THEIRS):
                    sprite.pixels = px
                    self.lose()
                    self.complete_action()
                    return
            sprite.pixels = px
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION6:
            at = self.camera.display_to_grid(self.action.data.get('x', 0), self.action.data.get('y', 0))
            if at is not None:
                c, r = at
                if 1 <= r <= 3 and 1 <= c <= 3 and px[r, c] == EMPTY:
                    px[r, c] = MINE
                    sprite.pixels = px
                    if _line(px[1:4, 1:4], MINE):
                        self.next_level()
                        self.complete_action()
                        return
                    self._reply = True          # the action goes on: one more frame, for the opponent
                    return
        self.complete_action()


GAMES = {'lights': LightsOut, 'clear': ClearRed, 'tictactoe': TicTacToe}


class _LocalEnv:
    def __init__(self, game):
        self.game = game

    def reset(self):
        return self.game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    def step(self, action, data=None):
        return self.game.perform_action(ActionInput(id=action, data=data or {}), raw=True)


def local_env(game):
    return _LocalEnv(game)
