#!/usr/bin/env python3
"""
Screen Reading - learning to read a chess screen from examples

The program sees a chess board only as a picture (an RGB image: the
repository's own board, python-chess's SVG drawn to pixels as the GUI does,
or a board image from elsewhere) and learns to read it the way it learned
digits and letters (symbol_learning): from lessons whose meaning it can
work out itself. What it knows beforehand, as it knew the alphabet: the
starting position of chess, and the rules (python-chess), so it knows which
positions can follow a position it knows.

Lessons:
- the starting screen (learn_start): the screen at the start of a game is
  the starting position: each of its 64 squares is learned as what the
  starting position has there.
- a screen in play (follow): the position on a screen after a move is one
  legal move (or, between the player's own turns, two: the player's and
  the opponent's) from the position known before. Of those positions, the
  one the screen's squares fit best is taken (read in context, as
  sequence_puzzles reads a row of numbers); when it fits clearly better
  than the next, the squares it was not read right on are learned as that
  position has them.

Reading (read): the board is found on the screen (the rows and columns
mostly of the two square colours), cut into 8 x 8 squares; in each square the ink is
what differs from the square's own colour (its border), its outline filled
in (a white piece's inside may be close to a light square's colour), split
into dark and light ink (a black piece is mostly dark, a white piece light inside a dark
outline); its two masks, cropped to the ink, scaled to SIZE x SIZE and
blurred, are compared with the examples (nearest). A square with little ink
is empty. Small ink in a square's corner (the coordinates some boards print)
is not counted.

Results (evaluate_screen_reading; headless_trainer --screen):
- the repository's own screens, after the starting screen only (400 pixels,
  default colours): 60 positions of random games in each of six styles
  (the same; the last move highlighted; 260 and 600 pixels; green and blue
  squares): every square of every board read right (3840/3840 each).
- Lichess board images, 8 piece sets, 10 positions each: with the
  repository's starting screen only, cburnett (the set python-chess draws)
  and merida 640/640; the other six 519 to 632 of 640 squares (0 to 3 of 10
  boards); after that set's starting screen as one more lesson, every set
  640/640.
- playing from the screen (the chess trainer, the opponent's moves found
  from the screens in context): 20 games against the random opponent and 10
  against Stockfish (level 0), 913 turns: on every turn the board the agent
  decided from was the true board; no screen was lost, no move illegal.

Playing from the screen (ScreenPlayer): a player that knows only what it
has read: it keeps its own board, follows each screen it is shown, and
plays from that board; HeadlessTrainer(screen=True) plays its games this
way (headless_trainer).
"""

import io
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SIZE = 16               # piece images compared at SIZE x SIZE (each of the two masks)
DIFFERENT = 60          # a pixel is ink when this far (RGB distance) from its square's colour
EMPTY = 0.03            # a square with less ink than this share of its area is empty
DARK = 100              # ink darker than this (mean of RGB) is dark ink, the rest light
CORNER = 0.3            # ink touching a corner region this share of the square wide, and small, is a label
CLEAR = 0.6             # follow: the best fitting position must fit at most this share of the next's misfit
PIECES = 'PNBRQKpnbrqk'
SQUARE_SHARE = 0.15     # find_board: a board row (column) has at least this share of square colours


# ------------------------------------------------------------------ screens

def render(board, size: int = 400, colors: Optional[Dict] = None, lastmove=None, flipped: bool = False) -> np.ndarray:
    """The repository's own chess screen: python-chess's SVG board (as ai_gui draws it) to RGB pixels"""
    import cairosvg
    import chess.svg
    from PIL import Image
    svg = chess.svg.board(board, size=size, colors=colors or {}, lastmove=lastmove, flipped=flipped)
    png = cairosvg.svg2png(bytestring=svg.encode())
    return np.array(Image.open(io.BytesIO(png)).convert('RGB'))


def find_board(screen: np.ndarray) -> Tuple[int, int, int, int]:
    """
    (top, left, height, width) of the board: the two commonest colours of the screen are the squares'; the
    board's rows are the longest run of rows a good share of whose pixels are of those colours (pieces and
    highlights take the rest), and its columns likewise
    """
    rgb = screen[..., :3].astype(int)
    keys, counts = np.unique(rgb.reshape(-1, 3) // 8, axis=0, return_counts=True)
    top2 = keys[np.argsort(-counts)[:2]]
    near = np.zeros(rgb.shape[:2], bool)
    for k in top2:
        near |= np.all(np.abs(rgb // 8 - k) <= 1, axis=2)

    def longest(flags, gap: int = 3):
        """the longest run of True, gaps of at most gap False (the blended edges between squares) inside"""
        runs, start, last = [], None, None
        for i, f in enumerate(flags):
            if f:
                if start is None or i - last > gap + 1:
                    if start is not None:
                        runs.append((start, last + 1))
                    start = i
                last = i
        if start is not None:
            runs.append((start, last + 1))
        return max(runs, key=lambda r: r[1] - r[0], default=(0, len(flags)))
    r0, r1 = longest(near.mean(axis=1) > SQUARE_SHARE)
    c0, c1 = longest(near[r0:r1].mean(axis=0) > SQUARE_SHARE)
    r0, r1 = longest(near[:, c0:c1].mean(axis=1) > SQUARE_SHARE)
    return r0, c0, r1 - r0, c1 - c0


def squares(screen: np.ndarray) -> List[np.ndarray]:
    """The 64 squares of the board, a8, b8, ... h8, a7, ... h1 (the screen's top row first)"""
    top, left, h, w = find_board(screen)
    out = []
    for r in range(8):
        for c in range(8):
            r0, r1 = top + round(r * h / 8), top + round((r + 1) * h / 8)
            c0, c1 = left + round(c * w / 8), left + round((c + 1) * w / 8)
            out.append(screen[r0:r1, c0:c1, :3])
    return out


def _resize(mask: np.ndarray, box) -> np.ndarray:
    r0, r1, c0, c1 = box
    m = mask[r0:r1, c0:c1].astype(float)
    side = max(m.shape)
    sq = np.zeros((side, side))
    a, b = (side - m.shape[0]) // 2, (side - m.shape[1]) // 2
    sq[a:a + m.shape[0], b:b + m.shape[1]] = m
    edges = np.linspace(0, side, SIZE + 1).astype(int)
    out = np.array([[sq[edges[i]:max(edges[i] + 1, edges[i + 1]), edges[j]:max(edges[j] + 1, edges[j + 1])].mean()
                     for j in range(SIZE)] for i in range(SIZE)])
    from scipy import ndimage
    return ndimage.gaussian_filter(out, 0.7)


def square_features(square: np.ndarray) -> Optional[np.ndarray]:
    """A square's piece as a feature vector (its dark and light ink, see the module docstring); None: empty"""
    from scipy import ndimage
    rgb = square.astype(int)
    h, w = rgb.shape[:2]
    m = max(2, min(h, w) // 12)
    border = np.concatenate([rgb[:m].reshape(-1, 3), rgb[-m:].reshape(-1, 3), rgb[:, :m].reshape(-1, 3),
                             rgb[:, -m:].reshape(-1, 3)])
    keys, counts = np.unique(border // 8, axis=0, return_counts=True)
    colour = keys[int(np.argmax(counts))] * 8 + 4
    ink = np.sqrt(((rgb - colour) ** 2).sum(axis=2)) > DIFFERENT
    ink[:m // 2], ink[-(m // 2):], ink[:, :m // 2], ink[:, -(m // 2):] = False, False, False, False
    # labels printed in a corner (coordinates): small pieces of ink within a corner region
    labels, n = ndimage.label(ink)
    cr, cc = int(CORNER * h), int(CORNER * w)
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        part = labels[sl] == i
        small = part.sum() < 0.04 * h * w
        in_corner = (sl[0].stop <= cr or sl[0].start >= h - cr) and (sl[1].stop <= cc or sl[1].start >= w - cc)
        if small and in_corner:
            ink[sl] &= ~part
    if ink.mean() < EMPTY:
        return None
    ink = ndimage.binary_fill_holes(ink)          # a white piece's inside, close to a light square's colour, too
    rows, cols = np.nonzero(ink)
    box = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
    lum = rgb.mean(axis=2)
    dark = ink & (lum < DARK)
    light = ink & (lum >= DARK)
    return np.concatenate([_resize(dark, box).ravel(), _resize(light, box).ravel()])


def placement(board) -> List[str]:
    """What a python-chess board has on each square, a8 ... h1 ('.': empty), as on the screen (white below)"""
    import chess
    out = []
    for rank in range(7, -1, -1):
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            out.append(p.symbol() if p else '.')
    return out


def board_fen(read: List[str]) -> str:
    """The board part of a FEN from 64 squares read (a8 ... h1; '.' empty, '?' unknown taken as empty)"""
    rows = []
    for r in range(8):
        row, empty = '', 0
        for piece in read[r * 8:(r + 1) * 8]:
            if piece in ('.', '?'):
                empty += 1
            else:
                row += (str(empty) if empty else '') + piece
                empty = 0
        rows.append(row + (str(empty) if empty else ''))
    return '/'.join(rows)


# ------------------------------------------------------------------ the memory

class ScreenMemory:
    """Examples of what squares look like (feature vectors) and what they show (a piece letter)"""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.examples: List[Tuple[np.ndarray, str, str]] = []           # (features, piece, source)
        if path and os.path.exists(path):
            with open(path) as f:
                for e in json.load(f)['examples']:
                    self.examples.append((np.frombuffer(bytes.fromhex(e['f']), np.uint8).astype(float) / 255,
                                          e['piece'], e['source']))
        self._matrix = None

    def save(self, path: Optional[str] = None) -> None:
        with open(path or self.path, 'w') as f:
            json.dump({'about': 'chess screen squares learned by screen_reading', 'size': SIZE,
                       'examples': [{'f': np.clip(np.round(v * 255), 0, 255).astype(np.uint8).tobytes().hex(),
                                     'piece': p, 'source': s} for v, p, s in self.examples]}, f)

    def learn(self, features: np.ndarray, piece: str, source: str = '') -> None:
        self.examples.append((features, piece, source))
        self._matrix = None

    def distances(self, features: np.ndarray) -> Dict[str, float]:
        """Per piece, the distance of its nearest example (mean absolute difference)"""
        if not self.examples:
            return {}
        if self._matrix is None:
            self._matrix = np.array([e[0] for e in self.examples])
        d = np.abs(self._matrix - features.reshape(1, -1)).mean(axis=1)
        out: Dict[str, float] = {}
        for (_, piece, _), x in zip(self.examples, d):
            out[piece] = min(out.get(piece, 1.0), float(x))
        return out

    def read_square(self, square: np.ndarray) -> str:
        f = square_features(square)
        if f is None:
            return '.'
        d = self.distances(f)
        return min(d, key=d.get) if d else '?'

    def read(self, screen: np.ndarray) -> List[str]:
        """The screen's 64 squares read, a8 ... h1"""
        return [self.read_square(s) for s in squares(screen)]

    # -------------------------------------------------------------- lessons

    def learn_start(self, screen: np.ndarray, source: str = 'starting screen') -> Dict:
        """The starting screen: each square is what the starting position has there"""
        import chess
        learned = 0
        for sq, piece in zip(squares(screen), placement(chess.Board())):
            f = square_features(sq)
            if piece != '.' and f is not None:
                self.learn(f, piece, source)
                learned += 1
        return {'outcome': 'learned', 'squares': learned}

    def misfit(self, dists: List[Optional[Dict[str, float]]], expected: List[str]) -> float:
        """How badly squares fit a placement (dists: per square, distances() of its piece, None if empty): per
        square, 1 for ink where it has none (or none where it has a piece), else the distance to the piece's
        nearest example (1 if no example)"""
        total = 0.0
        for d, piece in zip(dists, expected):
            if piece == '.':
                total += 0.0 if d is None else 1.0
            elif d is None:
                total += 1.0
            else:
                total += d.get(piece, 1.0)
        return total

    def follow(self, board, screen: np.ndarray, plies: int = 1, learn: bool = True,
               source: str = 'play') -> Optional[List]:
        """
        The moves (plies of them, 1 or 2) that lead from board (the position known) to the position on the
        screen, read in context (module docstring); None when no sequence of moves fits clearly better than
        the next. The squares read wrong are learned (learn) as the position found has them
        """
        feats = [square_features(s) for s in squares(screen)]
        dists = [None if f is None else self.distances(f) for f in feats]
        options = []
        for m1 in list(board.legal_moves):
            board.push(m1)
            if plies == 1 or board.is_game_over():
                options.append(([m1], self.misfit(dists, placement(board))))
            else:
                for m2 in list(board.legal_moves):
                    board.push(m2)
                    options.append(([m1, m2], self.misfit(dists, placement(board))))
                    board.pop()
            board.pop()
        if plies == 2 and board.is_game_over():
            return None
        options.sort(key=lambda e: e[1])
        if not options:
            return None
        # several move orders can give one position: they count as one
        best_moves, best = options[0]
        after = board.copy()
        for m in best_moves:
            after.push(m)
        target = placement(after)
        rival = next((cost for moves, cost in options[1:]
                      if placement(_after(board, moves)) != target), None)
        if rival is not None and best > CLEAR * rival:
            return None
        if learn:
            for f, d, piece in zip(feats, dists, target):
                if piece != '.' and d and min(d, key=d.get) != piece:
                    self.learn(f, piece, source)
        return best_moves


def _after(board, moves):
    b = board.copy()
    for m in moves:
        b.push(m)
    return b


# ------------------------------------------------------------------ playing from the screen

class ScreenPlayer:
    """
    A player that knows the game only from screens: its own board starts from the starting screen (a
    lesson) and follows every screen shown (read in context); it decides from its own board
    """

    def __init__(self, memory: ScreenMemory, learn: bool = True):
        import chess
        self.memory = memory
        self.learn = learn
        self.board = chess.Board()
        self.stats = {'screens': 0, 'followed': 0, 'lost': 0, 'squares read right': 0, 'squares': 0}

    def new_game(self, screen: Optional[np.ndarray] = None) -> None:
        import chess
        self.board = chess.Board()
        if screen is not None and not self.memory.examples:
            self.memory.learn_start(screen)

    def see(self, screen: np.ndarray, truth=None, plies: int = 2) -> bool:
        """Follow a screen shown after plies moves (the player's own and the opponent's); False when lost"""
        self.stats['screens'] += 1
        if truth is not None:
            read = self.memory.read(screen)
            self.stats['squares'] += 64
            self.stats['squares read right'] += sum(a == b for a, b in zip(read, placement(truth)))
        moves = self.memory.follow(self.board, screen, plies=plies, learn=self.learn)
        if moves is None:
            self.stats['lost'] += 1
            return False
        for m in moves:
            self.board.push(m)
        self.stats['followed'] += 1
        return True
