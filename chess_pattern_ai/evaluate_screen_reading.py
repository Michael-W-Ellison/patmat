#!/usr/bin/env python3
"""
Evaluate screen_reading: chess screens read after learning from the starting screen only.

    python3 evaluate_screen_reading.py [cache directory]

1. The repository's own screens (screen_reading.render): the program is shown one starting screen (400
   pixels, the default colours) and then reads 60 positions of random games in six styles: the same, the
   last move highlighted, 260 and 600 pixels, green squares, blue squares.
2. Lichess board images (https://lichess1.org/export/fen.gif, downloaded into the cache once): 8 piece
   sets on several board themes, 10 positions each: read with what the program learned from the
   repository's starting screen only, and after one more lesson, that piece set's starting screen.
"""

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

import chess
import numpy as np

import screen_reading as S

STYLES = {'same': {}, 'last move shown': {'lastmove': True}, 'size 260': {'size': 260}, 'size 600': {'size': 600},
          'green squares': {'colors': {'square light': '#eeeed2', 'square dark': '#769656'}},
          'blue squares': {'colors': {'square light': '#dee3e6', 'square dark': '#8ca2ad'}}}
LICHESS_SETS = [('cburnett', 'brown'), ('merida', 'blue'), ('alpha', 'green'), ('chessnut', 'purple'),
                ('leipzig', 'ic'), ('companion', 'brown'), ('kosal', 'blue'), ('fantasy', 'green')]
LICHESS = 'https://lichess1.org/export/fen.gif?'


def positions(n: int, seed: int, low: int = 4, high: int = 80):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b, last = chess.Board(), None
        for _ in range(rng.randint(low, high)):
            moves = list(b.legal_moves)
            if not moves:
                break
            last = rng.choice(moves)
            b.push(last)
        out.append((b.copy(), last))
    return out


def score(memory, screens_and_boards):
    right = boards = 0
    for screen, b in screens_and_boards:
        ok = sum(x == y for x, y in zip(memory.read(screen), S.placement(b)))
        right += ok
        boards += ok == 64
    return right, boards


def own_screens() -> None:
    memory = S.ScreenMemory()
    memory.learn_start(S.render(chess.Board(), 400))
    tests = positions(60, 0)
    for name, kw in STYLES.items():
        shown = []
        for b, last in tests:
            k = dict(kw)
            size = k.pop('size', 400)
            if k.pop('lastmove', False):
                k['lastmove'] = last
            shown.append((S.render(b, size, **k), b))
        right, boards = score(memory, shown)
        print(f'own screens, {name:16s} squares {right}/{64 * len(tests)}, boards all right {boards}/{len(tests)}')


def lichess(cache: str) -> None:
    from PIL import Image
    os.makedirs(cache, exist_ok=True)
    fens = [chess.STARTING_FEN] + [b.fen() for b, _ in positions(10, 7, 6, 70)]

    def image(piece, theme, k):
        path = os.path.join(cache, f'{piece}_{k}.gif')
        if not os.path.exists(path):
            url = LICHESS + urllib.parse.urlencode({'fen': fens[k], 'color': 'white', 'theme': theme, 'piece': piece})
            req = urllib.request.Request(url, headers={'User-Agent': 'patmat-screen-test'})
            with urllib.request.urlopen(req) as r, open(path, 'wb') as f:
                f.write(r.read())
            time.sleep(0.5)
        return np.array(Image.open(path).convert('RGB'))
    own = S.ScreenMemory()
    own.learn_start(S.render(chess.Board(), 400))
    for piece, theme in LICHESS_SETS:
        tests = [(image(piece, theme, k), chess.Board(fens[k])) for k in range(1, len(fens))]
        line = f'lichess {piece:10s} ({theme:6s})'
        for lesson in (False, True):
            memory = S.ScreenMemory()
            memory.examples = list(own.examples)
            if lesson:
                memory.learn_start(image(piece, theme, 0), f'lichess {piece} start')
            right, boards = score(memory, tests)
            line += f' | {"+ its starting screen" if lesson else "own starting screen only"}: squares {right}/640, boards {boards}/10'
        print(line)


if __name__ == '__main__':
    own_screens()
    lichess(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_lichess'))
