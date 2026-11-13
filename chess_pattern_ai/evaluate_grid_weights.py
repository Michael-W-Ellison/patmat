#!/usr/bin/env python3
"""
Evaluate grid_weights: do learned weights of game-independent position features improve the board games'
trainers, as the evaluation weights did in chess?

    python3 evaluate_grid_weights.py learn <work dir> <game> <games>
    python3 evaluate_grid_weights.py play <work dir> <game> <arm> <games> [--seed S] [--opponent NAME]
    python3 evaluate_grid_weights.py report <work dir> [game ...]

learn: the game's trainer (against its usual opponent) plays games with use_weights on, drawing and tuning
the weights per game, and learning everything else as usual, into <work dir>/<game>/learned/ (its training
database and the shared general-pattern database next to it). play: copies of those databases
(<work dir>/<game>/<arm>/) play games with the arm's setting: 'plain' (no weights) or 'weights' (the learned
mean, not tuned); both keep learning everything else as usual. Each game's result and score margin is
appended to <work dir>/<game>/<arm>/games.jsonl as it ends; play carries on from that file if it exists (a
run stopped part-way is not lost). report: per game, each arm's games, wins, draws, losses and mean margin;
the difference in win rate with a test of two proportions; the learned weights.

--opponent: the trainer's opponent in play (default: its usual one, random
moves); OPPONENTS lists the others ('greedy' for breakthrough:
breakthrough_opponent.greedy).

GAMES: the board games whose trainers take part (not minesweeper: one player, hidden cells).
"""

import argparse
import importlib
import inspect
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import time

GAMES = {
    'arimaa': 'ArimaaHeadlessTrainer', 'breakthrough': 'BreakthroughHeadlessTrainer',
    'checkers': 'CheckersHeadlessTrainer', 'connect4': 'Connect4HeadlessTrainer',
    'dots_boxes': 'DotsBoxesHeadlessTrainer', 'go': 'GoHeadlessTrainer', 'gomoku': 'GomokuHeadlessTrainer',
    'hex': 'HexHeadlessTrainer', 'loa': 'LOAHeadlessTrainer', 'morris': 'MorrisHeadlessTrainer',
    'othello': 'OthelloHeadlessTrainer', 'pentago': 'PentagoHeadlessTrainer',
}
OPPONENTS = {('breakthrough', 'greedy'): ('breakthrough.breakthrough_opponent', 'greedy')}
DB = 'training.db'
ARMS = ('plain', 'weights')


def trainer(game, folder, weights, tune, opponent=None):
    logging.disable(logging.INFO)
    module = importlib.import_module(f'{game}.{game}_headless_trainer')
    cls = getattr(module, GAMES[game])
    cls.use_weights, cls.tune_weights = weights, tune
    if opponent is not None:
        where, name = OPPONENTS[(game, opponent)]
        cls.opponent = staticmethod(getattr(importlib.import_module(where), name))
    return cls(os.path.join(folder, DB)), module


def play_one(t, module, i):
    """One game, the agent's colour alternating where the trainer takes one"""
    params = inspect.signature(t.play_game).parameters
    out = io_quiet()
    try:
        if 'ai_color' in params:
            colours = list(module.Color)
            return t.play_game(colours[i % 2])
        return t.play_game()
    finally:
        out.close()


class io_quiet:
    """Trainers print as they play; silence stdout during a game"""
    def __init__(self):
        self.saved = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def close(self):
        sys.stdout.close()
        sys.stdout = self.saved


def learn(work, game, games):
    folder = os.path.join(work, game, 'learned')
    os.makedirs(folder, exist_ok=True)
    t, module = trainer(game, folder, True, True)
    log = os.path.join(folder, 'learn.jsonl')
    done = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    random.seed(f'learn {game} {done}')
    with open(log, 'a') as f:
        for i in range(done, games):
            start = time.time()
            result, score, _ = play_one(t, module, i)
            f.write(json.dumps({'game': i + 1, 'result': result, 'margin': score,
                                'mean': t.weights.tuner.mean(), 'seconds': time.time() - start}) + '\n')
            f.flush()
            print(f'{game} learn {i + 1}: {result} {score:+.0f} (generation {t.weights.tuner.generations})',
                  file=sys.stderr, flush=True)
    t.weights.close()


def play(work, game, arm, games, seed=None, opponent=None):
    src = os.path.join(work, game, 'learned')
    folder = os.path.join(work, game, arm + (f'_{seed}' if seed else ''))
    log = os.path.join(folder, 'games.jsonl')
    if not os.path.exists(log):
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(src):
            if name.endswith('.db'):
                shutil.copy(os.path.join(src, name), folder)
        open(log, 'w').close()
    done = sum(1 for _ in open(log))
    t, module = trainer(game, folder, arm == 'weights', False, opponent)
    random.seed(f'{game} {arm} {seed} {done}')
    with open(log, 'a') as f:
        for i in range(done, games):
            start = time.time()
            result, score, _ = play_one(t, module, i)
            f.write(json.dumps({'game': i + 1, 'result': result, 'margin': score,
                                'seconds': time.time() - start}) + '\n')
            f.flush()
            print(f'{game} {arm} {i + 1}: {result} {score:+.0f}', file=sys.stderr, flush=True)


def _rows(folder):
    path = os.path.join(folder, 'games.jsonl')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def two_proportions(w1, n1, w2, n2):
    """Two-sided p-value of a difference in proportions (normal approximation)"""
    if not n1 or not n2:
        return float('nan')
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    if se == 0:
        return 1.0
    z = abs(w1 / n1 - w2 / n2) / se
    return math.erfc(z / math.sqrt(2))


def report(work, games=None):
    from grid_weights import FEATURES
    for game in games or sorted(os.listdir(work)):
        if game not in GAMES:
            continue
        print(f'== {game}')
        base = os.path.join(work, game)
        arms = {}
        for arm in ARMS:
            rows = []
            for name in sorted(os.listdir(base)):
                if name == arm or name.startswith(arm + '_'):
                    rows += _rows(os.path.join(base, name))
            arms[arm] = rows
            if rows:
                won = sum(r['result'] == 'win' for r in rows)
                drawn = sum(r['result'] == 'draw' for r in rows)
                print(f'  {arm:8s} {len(rows):4d} games: won {won} ({won / len(rows) * 100:.1f}%), drawn {drawn}, '
                      f'lost {len(rows) - won - drawn}; mean margin {statistics.mean(r["margin"] for r in rows):+.1f}; '
                      f'{statistics.mean(r["seconds"] for r in rows):.2f} s per game')
        p, w = arms['plain'], arms['weights']
        if p and w:
            print(f'  difference in win rate: p = {two_proportions(sum(r["result"] == "win" for r in p), len(p), sum(r["result"] == "win" for r in w), len(w)):.3f}')
        log = os.path.join(base, 'learned', 'learn.jsonl')
        if os.path.exists(log):
            with open(log) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            print(f'  learned in {len(rows)} games; weights ' +
                  ', '.join(f'{n} {m:+.2f}' for n, m in zip(FEATURES, rows[-1]['mean'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('learn', 'play', 'report'))
    ap.add_argument('work')
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--seed', default=None)
    ap.add_argument('--opponent', default=None)
    a = ap.parse_args()
    if a.what == 'learn':
        learn(a.work, a.rest[0], int(a.rest[1]))
    elif a.what == 'play':
        play(a.work, a.rest[0], a.rest[1], int(a.rest[2]), a.seed, a.opponent)
    else:
        report(a.work, a.rest or None)
