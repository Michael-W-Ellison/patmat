#!/usr/bin/env python3
"""
Evaluate lookahead_patterns: does learning which positions hide a tactic, and reading them further, cut
the agent's blunders?

    python3 evaluate_lookahead.py learn <work dir> <games> [--level N]
    python3 evaluate_lookahead.py play <work dir> <arm> <games> [--level N] [--out file.jsonl]

learn: the chess trainer with use_lookahead plays games against Stockfish (skill level N, default 0),
learning as usual and, after each game, the look-ahead patterns; its database is <work dir>/learned.db.
play: a copy of that database (<work dir>/<arm>.db) plays games against the same opponent, arm 'plain'
(the search as before) or 'lookahead' (reading further where the patterns call for it, and still learning
them); both keep learning everything else as usual. Each of the agent's moves is judged by Stockfish (depth
blunder_analysis.JUDGE_DEPTH): centipawns lost, and the blunders (blunder_analysis.BLUNDER or more) as
blunder_analysis describes them. Printed: results, mean centipawn loss, blunders, the search's mean depth
reached and positions examined per move, and how often it read a position further.

Results: the lookahead_patterns docstring.
(For the results there, learn was stopped after 70 of 80 games by a time limit; its patterns are saved
after every game.)
"""

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time

import chess
import chess.engine

import blunder_analysis as BA


def trainer(db, lookahead, level, path):
    import logging
    logging.disable(logging.INFO)
    from headless_trainer import HeadlessTrainer
    HeadlessTrainer.use_lookahead = lookahead
    return HeadlessTrainer(db, opponent='stockfish', stockfish_level=level, stockfish_path=path)


def learn(work, games, level, path):
    os.makedirs(work, exist_ok=True)
    t = trainer(os.path.join(work, 'learned.db'), True, level, path)
    for i in range(games):
        start = time.time()
        result, _, _ = t.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        print(f'learn game {i + 1}: {result} ({time.time() - start:.0f} s; {t.lookahead.lessons} lessons, '
              f'{t.lookahead.overall():.2f} of positions hid a swing)', file=sys.stderr, flush=True)
    print('signatures (seen >= 20), by rate:')
    rows = [(t.lookahead.rate(s), s, n, h) for s, (n, h) in t.lookahead.counts.items() if n >= 20]
    for r, s, n, h in sorted(rows, reverse=True):
        print(f'  {s:24s} seen {n:5d} hidden {h:4d} rate {r:.2f}')
    print(f'{sum(n for n, _ in t.lookahead.counts.values())} lessons, overall rate {t.lookahead.overall():.3f}')
    t.close()


def play(work, arm, games, level, path, out=None):
    db = os.path.join(work, f'{arm}.db')
    shutil.copy(os.path.join(work, 'learned.db'), db)
    t = trainer(db, arm == 'lookahead', level, path)
    judge = chess.engine.SimpleEngine.popen_uci(path)
    limit = chess.engine.Limit(depth=BA.JUDGE_DEPTH)
    losses, records, depths, nodes, extended = [], [], [], [], []
    sink = open(out, 'w') if out else None
    play_move = t.play_ai_move

    def judged(board, colour):
        move = play_move(board, colour)
        if move is None:
            return move
        depths.append(t.search.depth_reached)
        nodes.append(t.search.nodes)
        extended.append(t.search.extensions)
        before = BA.score_of(judge.analyse(board, limit), colour)
        board.push(move)
        after = BA.score_of(judge.analyse(board, limit), colour)
        board.pop()
        loss = max(0, min(before, 1000) - max(after, -1000))
        losses.append(loss)
        if loss >= BA.BLUNDER:
            rec = BA.analyse_move(judge, board, move, t.search, colour)
            if rec is not None:
                records.append(rec)
                if sink:
                    sink.write(json.dumps(rec) + '\n')
                    sink.flush()
        return move

    t.play_ai_move = judged
    results = {}
    random.seed(arm)
    for i in range(games):
        result, _, _ = t.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        results[result] = results.get(result, 0) + 1
        print(f'{arm} game {i + 1}: {result} ({len(records)} blunders so far)', file=sys.stderr, flush=True)
    print(f'arm {arm}: results {results}')
    print(f'mean centipawn loss {statistics.mean(losses):.1f} over {len(losses)} moves; '
          f'search depth reached {statistics.mean(depths):.2f}, positions per move {statistics.mean(nodes):.0f}, '
          f'positions read further per move {statistics.mean(extended):.1f}')
    print(BA.report(records, len(losses)))
    judge.quit()
    t.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('learn', 'play'))
    ap.add_argument('work')
    ap.add_argument('rest', nargs='+')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--stockfish-path', default='/usr/games/stockfish')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.what == 'learn':
        learn(a.work, int(a.rest[0]), a.level, a.stockfish_path)
    else:
        play(a.work, a.rest[0], int(a.rest[1]), a.level, a.stockfish_path, a.out)
