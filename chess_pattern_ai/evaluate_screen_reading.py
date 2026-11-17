#!/usr/bin/env python3
"""
Evaluate screen_reading: chess screens read after learning from the starting screen only.

    python3 evaluate_screen_reading.py [cache directory]

1. The repository's own screens (screen_reading.render): the program is shown one starting screen (400
   pixels, the default colours) and then reads 60 positions of random games in six styles: the same, the
   last move highlighted, 260 and 600 pixels, green squares, blue squares.
   The same flipped (black at the bottom: the program finds which way from the rank labels, read with
   the digits it has learned); how often the labels tell the way at four sizes; and a program shown only
   a flipped starting screen.
2. Lichess board images (https://lichess1.org/export/fen.gif, downloaded into the cache once): 8 piece
   sets on several board themes, 10 positions each: read with what the program learned from the
   repository's starting screen only, and after one more lesson, that piece set's starting screen; and the
   same positions seen from black's side, after a lesson on its starting screen from black's side.
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
    for flipped in (False, True):
        for name, kw in STYLES.items():
            shown = []
            for b, last in tests:
                k = dict(kw)
                size = k.pop('size', 400)
                if k.pop('lastmove', False):
                    k['lastmove'] = last
                shown.append((S.render(b, size, flipped=flipped, **k), b))
            right, boards = score(memory, shown)
            print(f'own screens{", flipped" if flipped else ""}, {name:16s} squares {right}/{64 * len(tests)}, '
                  f'boards all right {boards}/{len(tests)}')
    # orientation by the rank labels, at sizes where they are readable and where they are not
    for size in (200, 260, 320, 400):
        found = [S.orientation(S.render(b, size, flipped=k % 2 == 1)) for k, (b, _) in enumerate(tests)]
        right = sum(f == (k % 2 == 1) for k, f in enumerate(found))
        wrong = sum(f is not None and f != (k % 2 == 1) for k, f in enumerate(found))
        print(f'orientation from the labels, size {size}: right {right}, wrong {wrong}, unknown '
              f'{sum(f is None for f in found)} of {len(found)}')
    # a program that has seen only a flipped starting screen
    flipped_first = S.ScreenMemory()
    print('starting screen shown flipped:', flipped_first.learn_start(S.render(chess.Board(), 400, flipped=True)))
    right, boards = score(flipped_first, [(S.render(b, 400, flipped=k % 2 == 1), b) for k, (b, _) in enumerate(tests)])
    print(f'  then reading both ways round: squares {right}/{64 * len(tests)}, boards all right {boards}/{len(tests)}')


def lichess(cache: str) -> None:
    from PIL import Image
    os.makedirs(cache, exist_ok=True)
    fens = [chess.STARTING_FEN] + [b.fen() for b, _ in positions(10, 7, 6, 70)]

    def image(piece, theme, k, side='white'):
        path = os.path.join(cache, f'{piece}_{k}.gif' if side == 'white' else f'{piece}_{k}_black.gif')
        if not os.path.exists(path):
            url = LICHESS + urllib.parse.urlencode({'fen': fens[k], 'color': side, 'theme': theme, 'piece': piece})
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
        # the same positions from black's side, after a lesson on that set's starting screen from black's side
        flipped = [(image(piece, theme, k, 'black'), chess.Board(fens[k])) for k in range(1, len(fens))]
        memory = S.ScreenMemory()
        memory.examples = list(own.examples)
        memory.learn_start(image(piece, theme, 0, 'black'), f'lichess {piece} start, black side')
        right, boards = score(memory, flipped)
        line += f' | from black\'s side, + its starting screen so: squares {right}/640, boards {boards}/10'
        print(line)


if __name__ == '__main__':
    own_screens()
    lichess(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.cache', 'patmat_lichess'))
