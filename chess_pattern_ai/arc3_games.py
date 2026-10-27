#!/usr/bin/env python3
"""
Small games for the ARC-AGI-3 agents, written for this project with the
ARC engine (package arcengine): the public example games have no click
actions, no other players and no falling pieces. The goal is not shown to
the player.

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

- Tetris (an 8 x 14 well inside a border of 5, background 0): a falling
  piece (one of the seven tetrominoes, each its own colour) moves left (3),
  right (4), down (2) or turns clockwise (1) when the cells are free; then,
  in a second frame, it falls one row. A piece that cannot fall further is
  locked; in a third frame full rows are removed (the rows above move
  down), the score (levels_completed) grows by the rows removed, and the
  next piece (random) appears at the top. The game is lost when it has no
  room to appear. win_levels is 200 (a game ends by a loss in practice).

- KInARow: k marks in a row on a board inside a border of 5 (empty 0,
  the player's mark 2, the opponent's 1), one game per episode (one
  level): the player clicks a cell (Gomoku: an empty cell gets its mark;
  Connect Four, with drop: a cell of a column, and the mark drops to the
  lowest empty cell of that column); the opponent's move is a second frame.
  The opponent is 'random' (as in the project's earlier board-game
  trainers) or 'block' (completes its own k if it can, else blocks the
  player's, else random). Who moves first alternates with the seed (the
  opponent's first move is on the board at the start). k in a row of the
  player's wins (WIN), of the opponent's loses (GAME_OVER), a full board is
  a draw (also GAME_OVER); `result` says which. Games: 'connect4' (6 x 7,
  k 4, drop), 'gomoku' (15 x 15, k 5), and '-block' versions.

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


TETROMINOES = [
    ([(0, 0), (0, 1), (0, 2), (0, 3)], 9),      # I
    ([(0, 0), (0, 1), (1, 0), (1, 1)], 11),     # O
    ([(0, 0), (0, 1), (0, 2), (1, 1)], 6),      # T
    ([(0, 0), (0, 1), (0, 2), (1, 0)], 12),     # L
    ([(0, 0), (0, 1), (0, 2), (1, 2)], 8),      # J
    ([(0, 1), (0, 2), (1, 0), (1, 1)], 3),      # S
    ([(0, 0), (0, 1), (1, 1), (1, 2)], 2),      # Z
]
WELL_W, WELL_H = 8, 14


class Tetris(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._phase = 0
        self.piece = None                       # (cells, colour), cells as (row, col) on the grid
        px = np.full((WELL_H + 2, WELL_W + 2), 5, dtype=np.int8)
        px[1:-1, 1:-1] = 0
        level = Level(sprites=[Sprite(pixels=px, name='board')], grid_size=(WELL_W + 2, WELL_H + 2))
        super().__init__(game_id='tetris', levels=[level], win_score=200,
                         camera=Camera(width=WELL_W + 2, height=WELL_H + 2, background=0, letter_box=3))

    def _px(self):
        return self.current_level.get_sprites_by_name('board')[0]

    def _free(self, px, cells):
        own = set(self.piece[0])
        return all(1 <= r <= WELL_H and 1 <= c <= WELL_W and (px[r, c] == 0 or (r, c) in own) for r, c in cells)

    def _put(self, px, cells, colour):
        for p in self.piece[0]:
            px[p] = 0
        for p in cells:
            px[p] = colour
        self.piece = (cells, colour)

    def _spawn(self, px) -> bool:
        shape, colour = TETROMINOES[self.rng.randrange(len(TETROMINOES))]
        cells = [(1 + r, 3 + c) for r, c in shape]
        if any(px[p] != 0 for p in cells):
            return False
        self.piece = (cells, colour)
        for p in cells:
            px[p] = colour
        return True

    def on_set_level(self, level):
        self.piece = None
        self._phase = 0

    def step(self):
        sprite = self._px()
        px = sprite.pixels.copy()
        if self.piece is None:                  # the first action of a game: the first piece appears
            self._spawn(px)
            sprite.pixels = px
            self.complete_action()
            return
        cells, colour = self.piece
        if self._phase == 0:                    # the player's move
            if self.action.id in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                if self.action.id == GameAction.ACTION1:
                    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
                    h = max(r for r, _ in cells) - r0
                    new = [(r0 + (c - c0), c0 + (h - (r - r0))) for r, c in cells]
                else:
                    dr, dc = {GameAction.ACTION2: (1, 0), GameAction.ACTION3: (0, -1),
                              GameAction.ACTION4: (0, 1)}[self.action.id]
                    new = [(r + dr, c + dc) for r, c in cells]
                if self._free(px, new):
                    self._put(px, new, colour)
            self._phase = 1
            sprite.pixels = px
            return
        if self._phase == 1:                    # gravity
            down = [(r + 1, c) for r, c in cells]
            if self._free(px, down):
                self._put(px, down, colour)
                cells = down
            sprite.pixels = px
            if self._free(px, [(r + 1, c) for r, c in cells]):
                self._phase = 0
                self.complete_action()
            else:
                self._phase = 2                 # locked: rows and the next piece in one more frame
            return
        full = [r for r in range(1, WELL_H + 1) if all(px[r, c] != 0 for c in range(1, WELL_W + 1))]
        if full:
            keep = [px[r, 1:-1].copy() for r in range(1, WELL_H + 1) if r not in full]
            rows = [np.zeros(WELL_W, dtype=np.int8) for _ in full] + keep
            for i, row in enumerate(rows):
                px[1 + i, 1:-1] = row
            self._score += len(full)
        self.piece = ([], 0)
        self._phase = 0
        if not self._spawn(px):
            sprite.pixels = px
            self.lose()
            self.complete_action()
            return
        sprite.pixels = px
        if self._score >= self._win_score:
            self.win()
        self.complete_action()


class KInARow(ARCBaseGame):
    def __init__(self, seed: int = 0, rows: int = 6, cols: int = 7, k: int = 4, drop: bool = True,
                 opponent: str = 'random'):
        self.rng = random.Random(seed)
        self.rows, self.cols, self.k, self.drop, self.opponent = rows, cols, k, drop, opponent
        self.agent_first = seed % 2 == 0
        self.result = None
        self._reply = False
        px = np.full((rows + 2, cols + 2), BORDER, dtype=np.int8)
        px[1:-1, 1:-1] = EMPTY
        level = Level(sprites=[Sprite(pixels=px, name='board')], grid_size=(cols + 2, rows + 2))
        super().__init__(game_id='kinarow', levels=[level],
                         camera=Camera(width=cols + 2, height=rows + 2, background=EMPTY, letter_box=3))

    def _board(self):
        return self.current_level.get_sprites_by_name('board')[0]

    def _legal(self, b):
        if self.drop:
            return [(max(r for r in range(self.rows) if b[r, c] == EMPTY), c)
                    for c in range(self.cols) if b[0, c] == EMPTY]
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if b[r, c] == EMPTY]

    def _line(self, b, v, p) -> bool:
        """k of v in a line through p"""
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                r, c = p[0] + s * dr, p[1] + s * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and b[r, c] == v:
                    n += 1
                    r, c = r + s * dr, c + s * dc
            if n >= self.k:
                return True
        return False

    def _opponent_move(self, b):
        legal = self._legal(b)
        if self.opponent == 'block':
            for v in (THEIRS, MINE):
                for p in legal:
                    trial = b.copy()
                    trial[p] = v
                    if self._line(trial, v, p):
                        return p
        return self.rng.choice(legal)

    def on_set_level(self, level):
        self.result = None
        self._reply = False
        if not self.agent_first:
            sprite = level.get_sprites_by_name('board')[0]
            px = sprite.pixels.copy()
            px[1:-1, 1:-1][self._opponent_move(px[1:-1, 1:-1])] = THEIRS
            sprite.pixels = px

    def step(self):
        sprite = self._board()
        px = sprite.pixels.copy()
        b = px[1:-1, 1:-1]
        if self._reply:                         # the opponent's move, in a second frame
            self._reply = False
            p = self._opponent_move(b)
            b[p] = THEIRS
            sprite.pixels = px
            if self._line(b, THEIRS, p):
                self.result = 'loss'
                self.lose()
            elif not self._legal(b):
                self.result = 'draw'
                self.lose()
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION6:
            at = self.camera.display_to_grid(self.action.data.get('x', 0), self.action.data.get('y', 0))
            if at is not None:
                c, r = at[0] - 1, at[1] - 1
                target = None
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.drop:
                        target = next((q for q in self._legal(b) if q[1] == c), None)
                    elif b[r, c] == EMPTY:
                        target = (r, c)
                if target is not None:
                    b[target] = MINE
                    sprite.pixels = px
                    if self._line(b, MINE, target):
                        self.result = 'win'
                        self.win()
                    elif not self._legal(b):
                        self.result = 'draw'
                        self.lose()
                    else:
                        self._reply = True      # the action goes on: one more frame, for the opponent
                        return
        self.complete_action()


def _k_game(**kw):
    def make(seed: int = 0):
        return KInARow(seed=seed, **kw)
    return make


GAMES = {'lights': LightsOut, 'clear': ClearRed, 'tictactoe': TicTacToe, 'tetris': Tetris,
         'connect4': _k_game(), 'connect4-block': _k_game(opponent='block'),
         'gomoku': _k_game(rows=15, cols=15, k=5, drop=False),
         'gomoku-block': _k_game(rows=15, cols=15, k=5, drop=False, opponent='block')}
BOARD_GAMES = ('connect4', 'connect4-block', 'gomoku', 'gomoku-block')


class _LocalEnv:
    def __init__(self, game):
        self.game = game

    def reset(self):
        return self.game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    def step(self, action, data=None):
        return self.game.perform_action(ActionInput(id=action, data=data or {}), raw=True)


def local_env(game):
    return _LocalEnv(game)
