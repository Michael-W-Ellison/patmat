#!/usr/bin/env python3
"""
Why do the learned evaluation weights (evaluation_weights) help less at 0.5 s per move than with time for
the 3-ply search to finish?

    python3 weights_depth_analysis.py play <work dir> <games> [--seed S] [--name N] [--level L]
    python3 weights_depth_analysis.py report <work dir>

When the search runs out of time it plays the best move of the last depth it finished (pattern_search.choose),
so the two conditions differ only where the 3-ply search takes longer than 0.5 s, and there the move comes from
the 2-ply search. play: the agent (the weights' learned mean, nothing learned) plays games against Stockfish
(skill L, default 0) with time for its 3-ply search to finish; at each of its moves the position is also searched
four ways, the plain evaluation and the weights, each to 2 and to 3 plies (no time limit), the 3-ply searches
timed, and each move chosen is judged by Stockfish (depth blunder_analysis.JUDGE_DEPTH: centipawns lost, as in
evaluate_lookahead). One line per position in <work dir>/<name>.positions.jsonl (the database is
<work dir>/learned.db, which must hold the learned weights; it is not changed).
report: per variant, the mean centipawn loss and blunders (blunder_analysis.BLUNDER or more), over all
positions and split by whether the 3-ply search took longer than SLOW seconds.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import chess
import chess.engine

import blunder_analysis as BA

SLOW = 0.5
VARIANTS = ('plain2', 'plain3', 'weights2', 'weights3')


def play(work, games, level, path, seed, name):
    import logging
    logging.disable(logging.INFO)
    from headless_trainer import HeadlessTrainer
    HeadlessTrainer.use_weights = True
    HeadlessTrainer.tune_weights = False
    HeadlessTrainer.SEARCH_SECONDS = 30.0
    t = HeadlessTrainer(os.path.join(work, 'learned.db'), opponent='stockfish', stockfish_level=level,
                        stockfish_path=path)
    t.learn = False
    weights = t.weights.mean()
    s = t.search
    judge = chess.engine.SimpleEngine.popen_uci(path)
    limit = chess.engine.Limit(depth=BA.JUDGE_DEPTH)
    out = open(os.path.join(work, f'{name}.positions.jsonl'), 'a')

    def search(board, w, depth):
        s.weights, s.max_depth = w, depth
        start = time.time()
        move = s.choose(board)
        return move, time.time() - start, s.nodes

    def analysed(board, colour):
        before = BA.score_of(judge.analyse(board, limit), colour)
        rec = {'fen': board.fen()}
        losses = {}
        for v in VARIANTS:
            move, took, nodes = search(board, weights if v.startswith('weights') else None, int(v[-1]))
            if move not in losses:
                board.push(move)
                after = BA.score_of(judge.analyse(board, limit), colour)
                board.pop()
                losses[move] = max(0, min(before, 1000) - max(after, -1000))
            rec[v] = {'move': move.uci(), 'loss': losses[move], 'seconds': took, 'nodes': nodes}
        out.write(json.dumps(rec) + '\n')
        out.flush()
        return chess.Move.from_uci(rec['weights3']['move'])

    t.play_ai_move = lambda board, colour: None if board.is_game_over() else analysed(board, colour)
    random.seed(f'{name} {seed}')
    for i in range(games):
        result, _, _ = t.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        print(f'{name} game {i + 1}: {result}', file=sys.stderr, flush=True)
    judge.quit()
    t.close()
    out.close()


def report(work):
    rows = []
    for f in sorted(os.listdir(work)):
        if f.endswith('.positions.jsonl'):
            with open(os.path.join(work, f)) as fh:
                rows += [json.loads(line) for line in fh if line.strip()]

    def line(label, rs):
        if not rs:
            return
        parts = []
        for v in VARIANTS:
            losses = [r[v]['loss'] for r in rs]
            b = sum(x >= BA.BLUNDER for x in losses)
            parts.append(f'{v} {statistics.mean(losses):5.1f} {b:3d} ({b / len(rs) * 100:4.1f}%)')
        print(f'{label:44s} {len(rs):5d}  ' + '   '.join(parts))
    print(f'{"positions":44s} {"n":>5s}  per variant: mean centipawn loss, blunders')
    line('all', rows)
    for v in ('plain3', 'weights3'):
        line(f'{v} search took <= {SLOW} s', [r for r in rows if r[v]['seconds'] <= SLOW])
        line(f'{v} search took > {SLOW} s', [r for r in rows if r[v]['seconds'] > SLOW])
    for v in ('plain', 'weights'):
        same = sum(r[v + '2']['move'] == r[v + '3']['move'] for r in rows)
        print(f'{v}: the 2-ply and 3-ply searches chose the same move in {same} of {len(rows)} positions')
    for v in ('plain3', 'weights3'):
        print(f'{v}: mean seconds {statistics.mean(r[v]["seconds"] for r in rows):.2f}, '
              f'mean positions {statistics.mean(r[v]["nodes"] for r in rows):.0f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('play', 'report'))
    ap.add_argument('work')
    ap.add_argument('games', nargs='?', type=int, default=10)
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--stockfish-path', default='/usr/games/stockfish')
    ap.add_argument('--seed', default='0')
    ap.add_argument('--name', default='run')
    a = ap.parse_args()
    if a.what == 'play':
        play(a.work, a.games, a.level, a.stockfish_path, a.seed, a.name)
    else:
        report(a.work)
