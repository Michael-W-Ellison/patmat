#!/usr/bin/env python3
"""
Evaluate board_screens: the screens of the twelve other board games, read after learning from the starting
screen and from play.

    python3 evaluate_board_screens.py [game ...]

For each game (all by default):
1. Play: the program is shown the starting screen (400 pixels, the game's own colours; a lesson), then
   follows the screens of 5 random games (both sides random; one screen after every move, the last move's
   places outlined): screens followed, of them the position found the true one, and screens lost.
2. Reading alone, with what it has learned so: 20 positions of other random games (4 to 80 moves in),
   place by place, in five styles: the same; the last move outlined; 300 and 600 pixels; and seen from the
   other side (games with two sides facing, board_screens.SIDED). Places read right (of the board's
   places), and positions with every place right.
3. Other colours (board_screens.other_colours: the board's colours turned round, the pieces kept): the
   memory of 1. follows 2 more random games shown so (no lesson on them), then reads the 20 positions in
   them.
4. Which way round (SIDED games): a game shown from the other side, followed with the way round not given
   (tried both ways in context): screens followed, positions true.
Results: the board_screens docstring.
"""

import random
import sys
import time

import numpy as np

import board_screens as B


def random_game(name, seed, plies=100):
    """The positions of a random game (both sides random), the starting position first"""
    rng = random.Random(seed)
    adapter, state = B.new_game(name)
    out = [state]
    for _ in range(plies):
        if adapter.is_over(state):
            break
        moves = adapter.legal_moves(state, adapter.to_move(state))
        if not moves:
            break
        # the positions after the moves, in a fixed order (the moves' order may differ between runs), so a seed
        # gives the same game each run
        after = sorted({B.grid_of(adapter, a).tobytes(): a for a in (adapter.apply(state, m) for m in moves)}.items())
        state = rng.choice(after)[1]
        out.append(state)
    return adapter, out


def follow_game(memory, name, seed, colours=None, flipped=False, given=True):
    """Follow a random game's screens; (screens, followed, true, lost)"""
    adapter, states = random_game(name, seed)
    own = states[0]
    last = B.grid_of(adapter, own)
    before = memory.look(B.render(name, last, 400, colours, flipped))
    n = followed = true = lost = 0
    for state in states[1:]:
        cells = B.grid_of(adapter, state)
        shown = B.render(name, cells, 400, colours, flipped, [tuple(p) for p in np.argwhere(cells != last)])
        last = cells
        n += 1
        found = memory.follow(adapter, own, shown, flipped=flipped if given else None, before=before)
        if found is None:
            lost += 1
            own, before = state, memory.look(shown)
        else:
            followed += 1
            own, before = found[1], found[2]
            true += np.array_equal(B.grid_of(adapter, own), cells)
    return n, followed, true, lost


def held_out(name, n=20, seed=1000):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        adapter, states = random_game(name, rng.randrange(10 ** 6), rng.randint(4, 80))
        if len(states) > 1:
            out.append((B.grid_of(adapter, states[-1]), B.grid_of(adapter, states[-2])))
    return out


def read_score(memory, tests, size=400, colours=None, flipped=False, marked=False):
    right = boards = 0
    for cells, before in tests:
        marks = [tuple(p) for p in np.argwhere(cells != before)] if marked else []
        read = memory.read(B.render(memory.name, cells, size, colours, flipped, marks), flipped)
        ok = int(((read == cells) | ~memory.valid.reshape(cells.shape)).sum() - (~memory.valid).sum())
        right += ok
        boards += ok == memory.valid.sum()
    return right, boards


def evaluate(name):
    t = time.time()
    memory = B.BoardScreenMemory(name)
    adapter, start = B.new_game(name)
    first = memory.learn_start(B.render_state(name, adapter, start), 'the starting screen', flipped=False)
    totals = np.zeros(4, int)
    for seed in range(5):
        totals += follow_game(memory, name, seed)
    n, followed, true, lost = totals
    print(f'{name}: starting screen {first["examples"]} examples; 5 games, {n} screens: followed {followed} '
          f'(true position {true}), lost {lost}; {len(memory.examples)} examples')
    tests = held_out(name)
    places = int(memory.valid.sum()) * len(tests)
    styles = {'same': {}, 'last move outlined': {'marked': True}, '300 pixels': {'size': 300},
              '600 pixels': {'size': 600}}
    if name in B.SIDED:
        styles['from the other side'] = {'flipped': True}
    line = '  read alone:'
    for style, kw in styles.items():
        right, boards = read_score(memory, tests, **kw)
        line += f' {style} {right}/{places} ({boards}/{len(tests)});'
    print(line)
    colours = B.other_colours(name)
    totals = np.zeros(4, int)
    for seed in (5, 6):
        totals += follow_game(memory, name, seed, colours)
    n, followed, true, lost = totals
    right, boards = read_score(memory, tests, colours=colours)
    print(f'  other colours, 2 games, {n} screens: followed {followed} (true position {true}), lost {lost}; '
          f'then read alone {right}/{places} ({boards}/{len(tests)})')
    if name in B.SIDED:
        n, followed, true, lost = follow_game(memory, name, 7, flipped=True, given=False)
        print(f'  from the other side, the way round not given, {n} screens: followed {followed} '
              f'(true position {true}), lost {lost}')
    print(f'  ({time.time() - t:.0f} s)', flush=True)


if __name__ == '__main__':
    for game in sys.argv[1:] or list(B.GAMES):
        evaluate(game)
