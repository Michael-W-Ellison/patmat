#!/usr/bin/env python3
"""
Board Screens - learning to read the screens of the other board games from examples

As screen_reading does for chess, for the twelve other board games here
(Othello, Go, Gomoku, Hex, Connect Four, Checkers, Breakthrough, Lines of
Action, Arimaa, Pentago, Nine Men's Morris, Dots and Boxes). None of these
games has a picture screen of its own (each prints its board as text), so the
screens are drawn by this module's renderer (render): each game's board in
its usual look (discs on a green board, stones on the points of a wooden
board, hexagons, a blue Connect Four frame with holes, squares, Arimaa's
lettered pieces, the lines of a Morris board, dots and the lines drawn
between them), in a choice of colours and sizes.

What the program knows beforehand, as it knew the alphabet before reading
words: each game's rules and starting position (the game's adapter: legal
moves, the position after a move, the position as a grid of values,
adapter.to_grid), and the shape of its board: how many rows and columns of
places it has and how they are laid out (squares or points in rows and
columns, or hexagons with every other row shifted by half a hexagon); a
board is drawn as a rectangle unlike the page around it, its places evenly
spaced in it. What a place looks like with each value on it, it learns:

Lessons:
- the starting screen (learn_start): every place of the screen at the start
  of a game is learned as what the starting position has there. (Go, Gomoku,
  Hex and Dots and Boxes start empty: only empty places are learned.)
- a screen in play (follow): the position on a screen shown after a move is
  one legal move from the position known before. Of those positions, the one
  the screen's places fit best is taken (read in context); when it misfits by
  at least CLEAR less than the next (moves that give the same picture count
  as one), the places it was not read right on, and the places that looked
  unlike every example of what they are (KEEP), are learned as that position
  has them. In play a place also fits what it was on the screen before as
  near as it looks to itself there; a place unlike anything known, or changed
  since that screen, may have come to hold something not seen yet (fit). So
  the first stones, kings, drawn lines and boxes, which no starting screen
  shows, are learned in play.

Reading (read): the board is the region of the screen unlike the page around
it (the commonest colour of the screen's edge, what the region encloses
counted in); each place is cut out around its centre (a place's width and
height, and the same moved a little each way, SHIFT), scaled to SIZE x SIZE
and compared (mean absolute difference of its colours) with the examples: the
value of the nearest is read. Which way up a board is shown (games with two
sides facing, SIDED: the agent's side at the bottom) is given, or found in
context when following (both ways tried).

Playing from the screen (ScreenView): the agent keeps its own position, made
only from the screens it is shown (one after every move, the last move's
places outlined), decides from it, and its move is played on the true game;
each game's headless trainer takes screen=True (and screen_side=True for the
agent's side at the bottom), and --screen, --screen-side on the command line.

Results (evaluate_board_screens; as committed):
- following, after the starting screen (400 pixels): 5 random games per game
  (both sides random), then 2 in other colours (no lesson on them), and for
  the 3 SIDED games 1 from the other side with the way round not given:
  6,292 screens, every one followed, the position found the true one on
  every screen.
- reading alone, 20 positions of other random games per game, with what was
  learned so: every place of every position right, in each style (the same;
  the last move outlined; 300 and 600 pixels; from the other side; the other
  colours), in 11 of the 12 games. Arimaa (the pieces told apart by small
  letters): 1278/1280 places in the same style, at 600 pixels and from the
  other side; 1276/1280 in the other colours; 1264/1280 at 300 pixels.
- playing from the screens (each trainer, 4 games, the opponent its own; and
  the 3 SIDED games from the agent's side): 3,856 screens; on 3,854 the
  agent's position was the true one; 2 Arimaa screens were lost (a piece
  moved for the first time with the outline on it looked more like other
  pieces' examples than its own, and the move that loses it in a trap came
  within CLEAR); no move illegal.
The thresholds (NOVEL, NEW, CHANGED, KEEP, SIZE, SHIFT) were chosen on 3 to
10 other random games per game (not those above; after the fixes below, 6
per game were followed again: 5,891 screens, none lost). What was fixed on the way:
an edge of the board only partly its colour counted in (the places then cut a
pixel off, enough to tell an Arimaa letter wrong: find_board's EDGE), and
cut-outs of unequal sizes, and halves rounded to even (round()).
"""

import copy
import importlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SIZE = 16               # a place is compared at SIZE x SIZE pixels (RGB)
SHIFT = 1 / 32          # ... and also cut out this share of a place off (the nearest counts)
PAGE = 30               # find_board: a pixel this far (RGB distance) from the page's colour is the board's
EDGE = 0.5              # ... an edge row (column) less than this share as far from the page as the next is not
SAME = 0.01             # an example nearer than this to one of the same value is not kept
KEEP = 0.03             # follow: a place read right is learned too when this far from its value's examples
CLEAR = 0.1             # follow: the best fitting position must misfit by this much less than the next
UNKNOWN = 1.0           # the distance of a value with no example yet
NOVEL = 0.15            # follow: a place further than this from every example looks like nothing known (fit)
NEW = 0.1               # ... the distance at which such a place fits any value other than its value before
CHANGED = 0.1           # ... a place this far from its look on the screen before has changed
SUPERSAMPLE = 2         # the renderer draws at this many times the size, then scales down (smooth edges)

GAMES = {
    # name: (layout, {value: what it is}) - the values are the adapter's to_grid values
    'othello': ('squares', {0: 'empty', 1: 'black disc', 2: 'white disc'}),
    'go': ('points', {0: 'empty', 1: 'black stone', 2: 'white stone'}),
    'gomoku': ('points', {0: 'empty', 1: 'black stone', 2: 'white stone'}),
    'hex': ('hex', {0: 'empty', 1: 'red stone', 2: 'blue stone'}),
    'connect4': ('squares', {0: 'empty', 1: 'red disc', 2: 'yellow disc'}),
    'checkers': ('squares', {0: 'empty', 1: 'red man', 2: 'red king', 3: 'black man', 4: 'black king'}),
    'breakthrough': ('squares', {0: 'empty', 1: 'white piece', 2: 'black piece'}),
    'loa': ('squares', {0: 'empty', 1: 'black disc', 2: 'white disc'}),
    'arimaa': ('squares', {0: 'empty', **{k: f'gold {n}' for k, n in enumerate(
        ['rabbit', 'cat', 'dog', 'horse', 'camel', 'elephant'], 1)}, **{k + 6: f'silver {n}' for k, n in enumerate(
        ['rabbit', 'cat', 'dog', 'horse', 'camel', 'elephant'], 1)}}),
    'pentago': ('squares', {0: 'empty', 1: 'white marble', 2: 'black marble'}),
    'morris': ('points', {0: 'empty', 1: 'white piece', 2: 'black piece'}),
    'dots_boxes': ('points', {0: 'nothing (no line, or a box no one has)', 1: 'a dot or a line drawn',
                              2: "red's box", 3: "blue's box"}),
}
# games with two sides facing each other: the colour whose pieces start at the bottom of the board
SIDED = {'checkers': 'RED', 'breakthrough': 'BLACK', 'arimaa': 'SILVER'}
ARIMAA_LETTERS = ' RCDHME'


# ------------------------------------------------------------------ the games

def game_parts(name: str):
    """(adapter class, game class) of a game"""
    adapter = importlib.import_module(f'{name}.{name}_adapter')
    game = importlib.import_module(f'{name}.{name}_game')
    a = next(getattr(adapter, n) for n in dir(adapter) if n.endswith('Adapter') and isinstance(getattr(adapter, n), type))
    g = next(getattr(game, n) for n in dir(game) if n.endswith('Game') and isinstance(getattr(game, n), type))
    return a, g


def new_game(name: str):
    """(adapter, a game at its starting position)"""
    a, g = game_parts(name)
    return a(), g()


def grid_of(adapter, state) -> np.ndarray:
    return np.asarray(adapter.to_grid(state).cells, dtype=int)


def valid_places(adapter, state) -> np.ndarray:
    return np.asarray(adapter.to_grid(state).valid, dtype=bool)


# ------------------------------------------------------------------ the layout of a board

def centres(layout: str, rows: int, cols: int, top: float, left: float, height: float,
            width: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    (y, x of each place's centre (rows x cols each), a place's height, its width) on a board whose rectangle
    is at (top, left, height, width): rows and columns evenly spaced, half a spacing in from the board's edges;
    hexagons (pointy-topped, touching) every other row (odd rows) shifted right by half a hexagon, the board
    reaching one hexagon's half-height above the top row's centres and below the bottom row's
    """
    r = np.arange(rows).reshape(-1, 1) * np.ones((1, cols))
    c = np.ones((rows, 1)) * np.arange(cols).reshape(1, -1)
    if layout != 'hex':
        sy, sx = height / rows, width / cols
        return top + (r + 0.5) * sy, left + (c + 0.5) * sx, sy, sx
    sx = width / (cols + 0.5)
    rad = sx / math.sqrt(3)
    sy = (height - 2 * rad) / max(1, rows - 1)
    return top + rad + r * sy, left + (c + 0.5 + 0.5 * (r % 2)) * sx, sy, sx


def board_height(layout: str, rows: int, cols: int, width: float) -> float:
    if layout != 'hex':
        return width / cols * rows
    sx = width / (cols + 0.5)
    rad = sx / math.sqrt(3)
    return 2 * rad + (rows - 1) * 1.5 * rad


def find_board(screen: np.ndarray) -> Tuple[int, int, int, int]:
    """(top, left, height, width) of the board: the largest region of pixels unlike the page (the commonest
    colour of the screen's edge), what it encloses counted in (holes the page's colour, as Connect Four's)"""
    from scipy import ndimage
    rgb = screen[..., :3].astype(int)
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    keys, counts = np.unique(edge // 8, axis=0, return_counts=True)
    page = keys[int(np.argmax(counts))] * 8 + 4
    far = np.sqrt(((rgb - page) ** 2).sum(axis=2))
    board = ndimage.binary_fill_holes(far > PAGE)
    lab, n = ndimage.label(board)
    if not n:
        return 0, 0, rgb.shape[0], rgb.shape[1]
    k = int(np.argmax(ndimage.sum(board, lab, range(1, n + 1)))) + 1
    rs, cs = ndimage.find_objects(lab)[k - 1]
    r0, r1, c0, c1 = rs.start, rs.stop, cs.start, cs.stop
    # an edge row (column) only partly of the board's colours (the board's edge smoothed) is not the board's
    rows, cols = far[r0:r1, c0:c1].mean(axis=1), far[r0:r1, c0:c1].mean(axis=0)
    if len(rows) > 2 and rows[0] < EDGE * rows[1]:
        r0 += 1
    if len(rows) > 2 and rows[-1] < EDGE * rows[-2]:
        r1 -= 1
    if len(cols) > 2 and cols[0] < EDGE * cols[1]:
        c0 += 1
    if len(cols) > 2 and cols[-1] < EDGE * cols[-2]:
        c1 -= 1
    return r0, c0, r1 - r0, c1 - c0


def places(screen: np.ndarray, layout: str, shape: Tuple[int, int], flipped: bool = False,
           shifts: bool = False) -> List[List[np.ndarray]]:
    """
    The board's places, row by row of the board (row 0 first), each cut out around its centre (a place's
    height and width): per place, that cut-out, and (shifts) the same moved by SHIFT of a place (at least a
    pixel) in each of the 8 directions. flipped: the board is seen from the other side (its places turned
    half round)
    """
    top, left, h, w = find_board(screen)
    ys, xs, sy, sx = centres(layout, shape[0], shape[1], top, left, h, w)
    my, mx = max(1, round(SHIFT * sy)), max(1, round(SHIFT * sx))
    moves = [(0, 0)] + ([(a * my, b * mx) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b] if shifts else [])
    ph, pw = max(1, int(round(sy))), max(1, int(round(sx)))        # every cut-out the same size
    out = []
    for y, x in zip(ys.ravel(), xs.ravel()):
        cuts = []
        for dy, dx in moves:
            r0 = min(max(0, math.floor(y + dy - sy / 2 + 0.5)), screen.shape[0] - ph)     # (halves up: round()
            c0 = min(max(0, math.floor(x + dx - sx / 2 + 0.5)), screen.shape[1] - pw)     # takes them to even)
            cuts.append(screen[r0:r0 + ph, c0:c0 + pw, :3])
        out.append(cuts)
    return out[::-1] if flipped else out


def features(place: np.ndarray) -> np.ndarray:
    """A place as SIZE x SIZE x 3 colours in 0..1 (its pixels averaged over each part)"""
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(place.astype(np.uint8)))
    return np.asarray(img.resize((SIZE, SIZE), Image.BOX), dtype=float).ravel() / 255


# ------------------------------------------------------------------ the renderer

def _rgb(c) -> Tuple[int, int, int]:
    return tuple(int(v) for v in c)


STYLE = {        # each game's colours (the names in BACKGROUND are the board's, the rest the pieces')
    'othello': {'page': (200, 205, 215), 'board': (20, 120, 60), 'line': (10, 50, 25),
                1: (25, 25, 25), 2: (240, 240, 240), 'edge': (40, 40, 40)},
    'go': {'page': (200, 205, 215), 'board': (222, 184, 110), 'line': (40, 30, 20),
           1: (25, 25, 25), 2: (245, 245, 245), 'edge': (40, 40, 40)},
    'gomoku': {'page': (200, 205, 215), 'board': (230, 196, 130), 'line': (60, 40, 20),
               1: (25, 25, 25), 2: (245, 245, 245), 'edge': (40, 40, 40)},
    'hex': {'page': (200, 205, 215), 'board': (200, 200, 190), 'cell': (240, 232, 210), 'line': (90, 90, 90),
            1: (210, 40, 40), 2: (40, 80, 210), 'edge': (40, 40, 40)},
    'connect4': {'page': (250, 250, 250), 'board': (30, 80, 200), 'hole': (250, 250, 250),
                 1: (220, 35, 35), 2: (245, 205, 30), 'edge': (60, 60, 60)},
    'checkers': {'page': (200, 205, 215), 'light': (240, 217, 181), 'dark': (160, 110, 70),
                 1: (200, 30, 30), 3: (35, 35, 35), 'edge': (20, 20, 20), 'crown': (240, 200, 40)},
    'breakthrough': {'page': (200, 205, 215), 'light': (238, 238, 210), 'dark': (118, 150, 86),
                     1: (248, 248, 248), 2: (30, 30, 30), 'edge': (20, 20, 20)},
    'loa': {'page': (200, 205, 215), 'light': (225, 210, 175), 'dark': (175, 135, 95),
            1: (25, 25, 25), 2: (245, 245, 245), 'edge': (40, 40, 40)},
    'arimaa': {'page': (200, 205, 215), 'board': (232, 222, 190), 'trap': (200, 150, 110), 'line': (120, 100, 70),
               'gold': (230, 180, 40), 'silver': (175, 178, 190), 'edge': (30, 30, 30), 'letter': (20, 20, 20)},
    'pentago': {'page': (200, 205, 215), 'board': (205, 60, 50), 'hole': (140, 35, 30), 'line': (70, 15, 10),
                1: (245, 245, 245), 2: (25, 25, 25), 'edge': (40, 40, 40)},
    'morris': {'page': (200, 205, 215), 'board': (225, 195, 135), 'line': (60, 40, 20),
               1: (245, 245, 245), 2: (25, 25, 25), 'edge': (40, 40, 40)},
    'dots_boxes': {'page': (200, 205, 215), 'board': (252, 252, 248), 'dot': (20, 20, 20), 'drawn': (40, 60, 140),
                   2: (230, 110, 110), 3: (110, 140, 230)},
}
BACKGROUND = ('page', 'board', 'light', 'dark', 'line', 'hole', 'cell', 'trap')


def other_colours(name: str) -> Dict:
    """The game's board in other colours (the board's colours' red, green and blue turned round), pieces kept"""
    return {k: (v[2], v[0], v[1]) for k, v in STYLE[name].items() if k in BACKGROUND}


def render(name: str, cells, size: int = 400, colours: Optional[Dict] = None, flipped: bool = False,
           marks: Sequence[Tuple[int, int]] = ()) -> np.ndarray:
    """
    The screen of a position of a game (cells: its to_grid values, rows x cols) as RGB pixels, size wide.
    colours: changes to the game's colours (STYLE); flipped: turned half round (the other side at the bottom);
    marks: places outlined in orange (as a last move is shown). flipped only for games whose board looks the
    same turned half round (SIDED)
    """
    from PIL import Image, ImageDraw
    layout = GAMES[name][0]
    st = dict(STYLE[name])
    st.update(colours or {})
    cells = np.asarray(cells)
    rows, cols = cells.shape
    if flipped:              # seen from the other side: the places turned half round, the pieces upright
        cells = cells[::-1, ::-1]
        marks = [(rows - 1 - r, cols - 1 - c) for r, c in marks]
    k = SUPERSAMPLE
    W = size * k
    margin = round(0.05 * W)
    bw = W - 2 * margin
    bh = board_height(layout, rows, cols, bw)
    H = int(round(bh + 2 * margin))
    img = Image.new('RGB', (W, H), _rgb(st['page']))
    d = ImageDraw.Draw(img)
    top, left = margin, margin
    ys, xs, sy, sx = centres(layout, rows, cols, top, left, bh, bw)
    s = min(sy, sx)

    def disc(y, x, rad, fill, edge=None, width=0.0):
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=_rgb(fill),
                  outline=_rgb(edge) if edge is not None else None, width=max(1, int(round(width))) if edge else 0)

    def square(r, c, fill):
        d.rectangle([left + c * sx, top + r * sy, left + (c + 1) * sx, top + (r + 1) * sy], fill=_rgb(fill))

    if 'board' in st:
        d.rectangle([left, top, left + bw, top + bh], fill=_rgb(st['board']))
    if 'light' in st:
        for r in range(rows):
            for c in range(cols):
                square(r, c, st['dark'] if (r + c) % 2 else st['light'])

    if name in ('go', 'gomoku'):
        lw = max(1, round(0.04 * s))
        for r in range(rows):
            d.line([xs[r, 0], ys[r, 0], xs[r, -1], ys[r, -1]], fill=_rgb(st['line']), width=lw)
        for c in range(cols):
            d.line([xs[0, c], ys[0, c], xs[-1, c], ys[-1, c]], fill=_rgb(st['line']), width=lw)
        n = rows
        stars = [2, n // 2, n - 3] if n < 13 else [3, n // 2, n - 4]
        for r in stars:
            for c in stars:
                if n >= 13 or (r == c or r + c == n - 1):
                    disc(ys[r, c], xs[r, c], 0.1 * s, st['line'])
    elif name in ('othello', 'arimaa'):
        if name == 'arimaa':
            for r, c in ((2, 2), (2, 5), (5, 2), (5, 5)):
                square(r, c, st['trap'])
        lw = max(1, round((0.03 if name == 'othello' else 0.02) * s))
        inside = lambda p, lo, hi: min(max(p, lo + lw / 2), hi - lw / 2)      # the edge lines on the board
        for r in range(rows + 1):
            y = inside(top + r * sy, top, top + bh)
            d.line([left, y, left + bw, y], fill=_rgb(st['line']), width=lw)
        for c in range(cols + 1):
            x = inside(left + c * sx, left, left + bw)
            d.line([x, top, x, top + bh], fill=_rgb(st['line']), width=lw)
    elif name == 'hex':
        lw = max(1, round(0.03 * s))
        rad = sx / math.sqrt(3) - lw / 2          # the outlines on the board
        for y, x in zip(ys.ravel(), xs.ravel()):
            pts = [(x + rad * math.cos(math.radians(90 + 60 * i)), y + rad * math.sin(math.radians(90 + 60 * i)))
                   for i in range(6)]
            d.polygon(pts, fill=_rgb(st['cell']), outline=_rgb(st['line']), width=lw)
    elif name == 'connect4':
        for y, x in zip(ys.ravel(), xs.ravel()):
            disc(y, x, 0.4 * s, st['hole'])
    elif name == 'pentago':
        for y, x in zip(ys.ravel(), xs.ravel()):
            disc(y, x, 0.25 * s, st['hole'])
        lw = max(2, round(0.08 * s))
        d.line([left + 3 * sx, top, left + 3 * sx, top + bh], fill=_rgb(st['line']), width=lw)
        d.line([left, top + 3 * sy, left + bw, top + 3 * sy], fill=_rgb(st['line']), width=lw)
    elif name == 'morris':
        from morris.morris_adapter import POINT_CELLS
        from morris.morris_board import MorrisBoard
        lw = max(1, round(0.05 * s))
        for p, nbs in MorrisBoard.ADJACENCIES.items():
            for q in nbs:
                (r1, c1), (r2, c2) = POINT_CELLS[p], POINT_CELLS[q]
                d.line([xs[r1, c1], ys[r1, c1], xs[r2, c2], ys[r2, c2]], fill=_rgb(st['line']), width=lw)
        for r, c in POINT_CELLS:
            disc(ys[r, c], xs[r, c], 0.12 * s, st['line'])

    # what is on each place
    sprites = {}
    for r in range(rows):
        for c in range(cols):
            v = int(cells[r, c])
            y, x = ys[r, c], xs[r, c]
            if name == 'dots_boxes':
                if r % 2 == 0 and c % 2 == 0:
                    disc(y, x, 0.16 * s, st['dot'])
                elif v == 1 and r % 2 == 0:            # a line drawn between dots (r, c - 1) and (r, c + 1)
                    d.rectangle([x - sx, y - 0.1 * s, x + sx, y + 0.1 * s], fill=_rgb(st['drawn']))
                elif v == 1:
                    d.rectangle([x - 0.1 * s, y - sy, x + 0.1 * s, y + sy], fill=_rgb(st['drawn']))
                elif v in (2, 3):
                    d.rectangle([x - 0.4 * s, y - 0.4 * s, x + 0.4 * s, y + 0.4 * s], fill=_rgb(st[v]))
                continue
            if v == 0:
                continue
            if name == 'checkers':
                own = 1 if v in (1, 2) else 3
                disc(y, x, 0.4 * s, st[own], st['edge'], 0.04 * s)
                disc(y, x, 0.28 * s, st[own], st['edge'], 0.025 * s)
                if v in (2, 4):
                    disc(y, x, 0.16 * s, st['crown'], st['edge'], 0.02 * s)
            elif name == 'breakthrough':
                fill, edge = st[v], st['edge'] if v == 1 else (120, 120, 120)
                w = max(1, round(0.035 * s))
                d.polygon([(x - 0.3 * s, y + 0.38 * s), (x + 0.3 * s, y + 0.38 * s), (x + 0.12 * s, y - 0.05 * s),
                           (x - 0.12 * s, y - 0.05 * s)], fill=_rgb(fill), outline=_rgb(edge), width=w)
                disc(y - 0.17 * s, x, 0.16 * s, fill, edge, w)
            elif name == 'arimaa':           # one picture of each piece, put on each of its places
                if v not in sprites:
                    sprites[v] = _arimaa_piece(v, s, st)
                sprite = sprites[v]
                img.paste(sprite, (int(round(x - sprite.width / 2)), int(round(y - sprite.height / 2))), sprite)
            elif name == 'hex':
                disc(y, x, 0.36 * s, st[v], st['edge'], 0.03 * s)
            elif name == 'go' or name == 'gomoku':
                disc(y, x, 0.47 * s, st[v], st['edge'], 0.03 * s)
            elif name == 'morris':
                disc(y, x, 0.38 * s, st[v], st['edge'], 0.04 * s)
            elif name == 'pentago':
                disc(y, x, 0.44 * s, st[v], st['edge'], 0.03 * s)
            else:                                   # othello, loa, connect4
                disc(y, x, 0.4 * s, st[v], st['edge'], 0.03 * s)
    for r, c in marks:
        y, x = ys[r, c], xs[r, c]
        d.rectangle([x - 0.47 * sx, y - 0.47 * sy, x + 0.47 * sx, y + 0.47 * sy], outline=(255, 140, 0),
                    width=max(1, round(0.06 * s)))
    img = img.resize((size, int(round(H / k))), Image.BOX)
    return np.asarray(img)


def _arimaa_piece(v: int, s: float, st: Dict):
    """An Arimaa piece's picture (RGBA): a gold or silver disc with the piece's letter"""
    from PIL import Image, ImageDraw, ImageFont
    side = int(round(0.9 * s)) + 2
    sprite = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    d = ImageDraw.Draw(sprite)
    gold = v <= 6
    c, rad = side / 2, 0.42 * s
    d.ellipse([c - rad, c - rad, c + rad, c + rad], fill=_rgb(st['gold' if gold else 'silver']),
              outline=_rgb(st['edge']), width=max(1, int(round(0.04 * s))))
    font = ImageFont.load_default(size=max(6, int(0.5 * s)))
    d.text((c, c), ARIMAA_LETTERS[v if gold else v - 6], fill=_rgb(st['letter']), font=font, anchor='mm')
    return sprite


def render_state(name: str, adapter, state, **kw) -> np.ndarray:
    return render(name, grid_of(adapter, state), **kw)


# ------------------------------------------------------------------ the memory

class BoardScreenMemory:
    """For one game: examples of what its places look like (features) and what is on them (a to_grid value)"""

    def __init__(self, name: str, start=None):
        """start: the game at its starting position (default: the game's class with its default board size)"""
        self.name = name
        self.layout, self.meanings = GAMES[name]
        self.values = sorted(self.meanings)
        adapter, default = new_game(name)
        self.start = copy.deepcopy(start) if start is not None else default
        start = self.start
        self.shape = grid_of(adapter, start).shape
        self.valid = valid_places(adapter, start).ravel()
        self.examples: List[Tuple[np.ndarray, int, str]] = []
        self._matrix = None

    def learn(self, f: np.ndarray, value: int, source: str = '') -> bool:
        """Keep an example (unless one of that value is already nearer than SAME)"""
        if self.examples:
            same = [e[0] for e in self.examples if e[1] == value]
            if same and np.abs(np.array(same) - f).mean(axis=1).min() < SAME:
                return False
        self.examples.append((f, int(value), source))
        self._matrix = None
        return True

    def look(self, screen: np.ndarray, flipped: bool = False) -> np.ndarray:
        """
        The features of the board's places: places (row by row of the board; places not on the board too) x
        cut-outs (the place's own first, then shifted ones: places()) x features; flipped: the screen shows the
        board from the other side
        """
        return np.array([[features(c) for c in cuts]
                         for cuts in places(screen, self.layout, self.shape, flipped, shifts=SHIFT > 0)])

    def distances(self, feats: np.ndarray) -> np.ndarray:
        """Per place and value (self.values), the distance of the value's nearest example (UNKNOWN if none): of
        a place's cut-outs (look()), the nearest (mean absolute difference)"""
        out = np.full((len(feats), len(self.values)), UNKNOWN)
        if not self.examples:
            return out
        if self._matrix is None:
            self._matrix = np.array([e[0] for e in self.examples])
            self._labels = np.array([self.values.index(e[1]) for e in self.examples])
        n, k, size = feats.shape
        flat = feats.reshape(-1, size)
        d = np.empty((len(flat), len(self._matrix)))
        for i, e in enumerate(self._matrix):
            d[:, i] = np.abs(flat - e).mean(axis=1)
        d = d.reshape(n, k, -1).min(axis=1)
        for j in range(len(self.values)):
            m = self._labels == j
            if m.any():
                out[:, j] = d[:, m].min(axis=1)
        return out

    def read(self, screen: np.ndarray, flipped: bool = False) -> np.ndarray:
        """The screen's position read alone, place by place: rows x cols of values (-1: nothing learned)"""
        if not self.examples:
            return np.full(self.shape, -1)
        dist = self.distances(self.look(screen, flipped))
        values = np.array(self.values)[dist.argmin(axis=1)]
        values[~self.valid] = 0
        return values.reshape(self.shape)

    def misfit(self, dist: np.ndarray, cells: np.ndarray) -> float:
        """How badly the places fit a position: the sum, over the board's places, of the distance of the
        position's value there"""
        idx = np.searchsorted(self.values, cells.ravel())
        return float(dist[np.arange(len(idx)), idx][self.valid].sum())

    # -------------------------------------------------------------- lessons

    def learn_start(self, screen: np.ndarray, source: str = 'starting screen',
                    flipped: Optional[bool] = None) -> Dict:
        """
        The starting screen: each place is what the starting position has there. flipped: whether the board is
        shown turned half round (None: when the starting position is the same either way round, it does not
        matter; else the way the examples known fit better; with none known, not a lesson)
        """
        adapter, _ = new_game(self.name)
        cells = grid_of(adapter, self.start)
        if flipped is None:
            if np.array_equal(cells, cells[::-1, ::-1]) or self.name not in SIDED:
                flipped = False
            elif self.examples:
                fits = {f: self.misfit(self.distances(self.look(screen, f)), cells) for f in (False, True)}
                flipped = fits[True] < fits[False]
            else:
                return {'outcome': 'not a lesson', 'why': 'which side is at the bottom is not known'}
        learned = 0
        for f, v, ok in zip(self.look(screen, flipped), cells.ravel(), self.valid):
            if ok:
                learned += self.learn(f[0], v, source)
        return {'outcome': 'learned', 'examples': learned, 'flipped': flipped}

    def fit(self, feats: np.ndarray, before: Optional[np.ndarray] = None,
            before_cells: Optional[np.ndarray] = None) -> np.ndarray:
        """
        distances(), in play: a place also fits the value it had on the screen before (before_cells) as near as
        it looks to itself there (before: that screen's features, in the same order); a place unlike anything
        known (all its distances over NOVEL), so changed, fits any other value at NEW; and a place changed
        (more than CHANGED) fits a value of which no example is known at NEW
        """
        dist = self.distances(feats)
        if before is None:
            return dist
        idx = np.searchsorted(self.values, before_cells.ravel())
        rows = np.arange(len(idx))
        moved = np.abs(feats[:, 0] - before[:, 0]).mean(axis=1)          # how far from its look before
        dist[rows, idx] = np.minimum(dist[rows, idx], moved)
        novel = dist.min(axis=1) > NOVEL
        keep = dist[rows, idx].copy()
        dist[novel] = np.minimum(dist[novel], NEW)
        changed = moved > CHANGED                     # a value not seen yet may have come
        unseen = [j for j, v in enumerate(self.values) if not any(e[1] == v for e in self.examples)]
        if unseen:
            sub = dist[np.ix_(changed, unseen)]
            dist[np.ix_(changed, unseen)] = np.minimum(sub, NEW)
        dist[rows, idx] = keep
        return dist

    def follow(self, adapter, state, screen: np.ndarray, learn: bool = True, source: str = 'play',
               flipped: Optional[bool] = None, before: Optional[np.ndarray] = None):
        """
        The move that leads from state (the position known) to the position on the screen, read in context
        (module docstring): (move, the position after it, the screen's features), or None when no move fits
        clearly better than the next. before: the features of the screen before (state's), if seen, in the order
        of the screen's places (look(..., flipped=False)); the third returned is this screen's so. The places
        read wrong are learned (learn) as the position found has them. flipped: None, tried both ways round
        (games in SIDED)
        """
        ways = [flipped] if flipped is not None else ([False, True] if self.name in SIDED else [False])
        tried = [self._follow(adapter, state, screen, f, before) for f in ways]
        found = [(t[1], k) for k, t in enumerate(tried) if t[0] is not None]
        if not found:
            return None
        k = min(found)[1]
        move, after, cells, feats = tried[k][0]
        if learn:           # the places read wrong, and new looks (further than KEEP from their value's examples)
            dist = self.distances(feats)
            read = np.array(self.values)[dist.argmin(axis=1)]
            near = dist[np.arange(len(dist)), np.searchsorted(self.values, cells.ravel())]
            for f, v, r, d, ok in zip(feats, cells.ravel(), read, near, self.valid):
                if ok and (v != r or d > KEEP):
                    self.learn(f[0], int(v), source)
        return move, after, feats[::-1] if ways[k] else feats

    def _follow(self, adapter, state, screen, flipped, before):
        feats = self.look(screen, flipped)
        if before is not None and flipped:
            before = before[::-1]
        dist = self.fit(feats, before, grid_of(adapter, state))
        options = {}
        if not adapter.is_over(state):
            for m in adapter.legal_moves(state, adapter.to_move(state)):
                after = adapter.apply(state, m)
                cells = grid_of(adapter, after)
                key = cells.tobytes()
                if key not in options:
                    options[key] = (self.misfit(dist, cells), m, after, cells)
        if not options:
            return None, float('inf')
        ranked = sorted(options.values(), key=lambda o: o[0])
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][0] - best[0] < CLEAR:
            return None, best[0]
        return (best[1], best[2], best[3], feats), best[0]


# ------------------------------------------------------------------ playing from the screen

class ScreenView:
    """
    What an agent knows of a game it sees only on screens: its own position, started from the starting
    position (the starting screen a lesson, if nothing is known yet) and moved on by every screen it is shown
    (one after each move, followed in context); it decides from that position, and its move is played on the
    true game as the true game's legal move that gives the same position (screen_stats counts each screen,
    whether the agent's position was then the true one, screens it could not follow, and moves it could not
    play). A screen it cannot follow is read alone, place by place; if that does not give the true position
    either, the agent is lost: it is given the true position (counted) so the game can go on.
    """

    def __init__(self, name: str, memory: Optional[BoardScreenMemory] = None, side: bool = False,
                 size: int = 400, colours: Optional[Dict] = None, mark_last: bool = True):
        self.name = name
        self.memory = memory            # made at the first game (its board size), if not given
        self.side = side
        self.size = size
        self.colours = colours
        self.mark_last = mark_last
        self.flipped = False
        self.stats = {'screens': 0, 'true position': 0, 'followed': 0, 'read alone': 0, 'lost': 0, 'illegal': 0}

    def new_game(self, adapter, game, agent_colour=None) -> None:
        """A game starts (game: its starting position, known to the agent); agent_colour: the agent's side"""
        self.adapter = adapter
        if self.memory is None:
            self.memory = BoardScreenMemory(self.name, game)
        self.own = copy.deepcopy(game)
        self.last = grid_of(adapter, game)
        bottom = SIDED.get(self.name)
        self.flipped = bool(self.side and bottom and agent_colour is not None
                            and getattr(agent_colour, 'name', str(agent_colour)) != bottom)
        shown = self.screen(game)
        if not self.memory.examples:
            self.memory.learn_start(shown, flipped=self.flipped)
        self.before = self.memory.look(shown)

    def screen(self, game) -> np.ndarray:
        cells = grid_of(self.adapter, game)
        marks = [tuple(p) for p in np.argwhere(cells != self.last)] if self.mark_last else []
        return render(self.name, cells, self.size, self.colours, self.flipped, marks)

    def see(self, game) -> None:
        """A move has been played on the true game: follow the screen now shown"""
        shown = self.screen(game)
        self.last = grid_of(self.adapter, game)
        self.stats['screens'] += 1
        found = self.memory.follow(self.adapter, self.own, shown, flipped=self.flipped, before=self.before)
        if found is not None:
            self.stats['followed'] += 1
            self.own, self.before = found[1], found[2]
            self.stats['true position'] += np.array_equal(grid_of(self.adapter, self.own), self.last)
            return
        self.before = self.memory.look(shown)
        self.own = copy.deepcopy(game)
        if np.array_equal(self.memory.read(shown, self.flipped), self.last):
            self.stats['read alone'] += 1          # read alone: the position known, though not how it came
            self.stats['true position'] += 1
        else:
            self.stats['lost'] += 1                # (given the true position, not counted as found)

    def true_move(self, game, move):
        """The true game's legal move that gives the position the agent's move gives on its own position"""
        want = grid_of(self.adapter, self.adapter.apply(self.own, move))
        for m in self.adapter.legal_moves(game, self.adapter.to_move(game)):
            if np.array_equal(grid_of(self.adapter, self.adapter.apply(game, m)), want):
                return m
        self.stats['illegal'] += 1
        return None


def screen_view(name: str, screen=None, side: bool = False) -> Optional[ScreenView]:
    """A trainer's screen option: None or False (no screen), True (a new ScreenView) or a ScreenView"""
    if screen is None or screen is False:
        return None
    return ScreenView(name, side=side) if screen is True else screen


def agent_move(view: Optional[ScreenView], game, choose):
    """
    The agent's move on the true game: choose(state) picks a move on a position; without a screen view on the
    true game itself, with one on the agent's own position (played as the true game's move that gives the
    same position; None if there is none)
    """
    if view is None:
        return choose(game)
    move = choose(view.own)
    return None if move is None else view.true_move(game, move)
