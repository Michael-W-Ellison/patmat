#!/usr/bin/env python3
"""
Evaluate lookahead_patterns: does learning which positions hide a tactic, and reading them further, cut
the agent's blunders?

    python3 evaluate_lookahead.py learn <work dir> <games> [--arm A] [--level N]
    python3 evaluate_lookahead.py play <work dir> <arm> <games> [--level N] [--out file.jsonl]

Arms (ARMS): 'plain' (the search as before), 'lookahead' (lookahead_patterns: reading further where the
look-ahead patterns call for it), 'memory' (episodic_memory: episodic and semantic memory of punished moves),
'memory_first' (the same, remembered punishments read first but not further), 'plain_depth' and
'memory_depth' (as plain and memory, with time enough for the search to finish its depth), 'memory_root'
(the memory used only on the search's best root moves).
learn: the chess trainer with arm A's learning on (default lookahead) plays games against Stockfish (skill
level N, default 0), learning as usual and, after each game, that arm's memory; its database is
<work dir>/learned.db. play: a copy of that database (<work dir>/<arm>.db) plays games against the same
opponent with the arm's setting (still learning its memory); every arm keeps learning everything else as
usual. Each of the agent's moves is judged by Stockfish (depth
blunder_analysis.JUDGE_DEPTH): centipawns lost, and the blunders (blunder_analysis.BLUNDER or more) as
blunder_analysis describes them. Printed: results, mean centipawn loss, blunders, the search's mean depth
reached and positions examined per move, and how often it read a position further.

Results: the lookahead_patterns and episodic_memory docstrings.
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


ARMS = {'plain': {}, 'lookahead': {'use_lookahead': True}, 'memory': {'use_memory': True},
        # memory, the remembered punishments read (first) but not further
        'memory_first': {'use_memory': True, 'extension': 0},
        # equal depth, not equal time: the 3-ply search always finished (up to DEPTH_SECONDS per move)
        'plain_depth': {'seconds': 30.0},
        'memory_depth': {'use_memory': True, 'seconds': 30.0},
        # memory at the root only: the search's best moves checked after it (pattern_search.PatternSearch.check)
        'memory_root': {'use_memory': True, 'mode': 'root'}}


def trainer(db, arm, level, path):
    import logging
    logging.disable(logging.INFO)
    from headless_trainer import HeadlessTrainer
    for flag in ('use_lookahead', 'use_memory'):
        setattr(HeadlessTrainer, flag, ARMS[arm].get(flag, False))
    import episodic_memory
    episodic_memory.EXTENSION = ARMS[arm].get('extension', 2)
    HeadlessTrainer.SEARCH_SECONDS = ARMS[arm].get('seconds', 0.5)
    t = HeadlessTrainer(db, opponent='stockfish', stockfish_level=level, stockfish_path=path)
    t.search.memory_mode = ARMS[arm].get('mode', 'lines')
    return t


def learn(work, games, level, path, arm='lookahead'):
    os.makedirs(work, exist_ok=True)
    t = trainer(os.path.join(work, 'learned.db'), arm, level, path)
    for i in range(games):
        start = time.time()
        result, _, _ = t.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        note = (f'{t.lookahead.lessons} lessons, {t.lookahead.overall():.2f} of positions hid a swing'
                if t.lookahead is not None else f'{len(t.memory)} episodes')
        print(f'learn game {i + 1}: {result} ({time.time() - start:.0f} s; {note})', file=sys.stderr, flush=True)
    if t.memory is not None:
        rows = list(t.memory.conn.execute('SELECT blunderer, COUNT(*), AVG(lost) FROM episodes GROUP BY blunderer'))
        print('episodes by whose move was punished (count, mean material lost):', rows)
        print('semantic memory, reply patterns by how many times more often than average they punished:')
        for f, r in sorted(t.general.refutations.rates.items(), key=lambda e: -e[1]):
            print(f'  {f:32s} {r:.2f}')
        t.close()
        return
    print('signatures (seen >= 20), by rate:')
    rows = [(t.lookahead.rate(s), s, n, h) for s, (n, h) in t.lookahead.counts.items() if n >= 20]
    for r, s, n, h in sorted(rows, reverse=True):
        print(f'  {s:24s} seen {n:5d} hidden {h:4d} rate {r:.2f}')
    print(f'{sum(n for n, _ in t.lookahead.counts.values())} lessons, overall rate {t.lookahead.overall():.3f}')
    t.close()


def play(work, arm, games, level, path, out=None, seed=None, name=None):
    db = os.path.join(work, f'{name or arm}.db')
    shutil.copy(os.path.join(work, 'learned.db'), db)
    t = trainer(db, arm, level, path)
    judge = chess.engine.SimpleEngine.popen_uci(path)
    limit = chess.engine.Limit(depth=BA.JUDGE_DEPTH)
    losses, records, depths, nodes, extended, recalls, seconds = [], [], [], [], [], [], []
    sink = open(out, 'w') if out else None
    play_move = t.play_ai_move

    def judged(board, colour):
        start = time.time()
        move = play_move(board, colour)
        seconds.append(time.time() - start)
        if move is None:
            return move
        depths.append(t.search.depth_reached)
        nodes.append(t.search.nodes)
        extended.append(t.search.extensions)
        recalls.append(t.search.recalls)
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
    random.seed(arm if seed is None else f'{arm} {seed}')
    for i in range(games):
        result, _, _ = t.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        results[result] = results.get(result, 0) + 1
        print(f'{arm} game {i + 1}: {result} ({len(records)} blunders so far)', file=sys.stderr, flush=True)
    print(f'arm {arm}: results {results}')
    print(f'mean centipawn loss {statistics.mean(losses):.1f} over {len(losses)} moves; '
          f'search depth reached {statistics.mean(depths):.2f}, positions per move {statistics.mean(nodes):.0f}, '
          f'positions read further per move {statistics.mean(extended):.1f}, '
          f'remembered moves matched per move {statistics.mean(recalls):.1f}, '
          f'seconds per move {statistics.mean(seconds):.2f} (most {max(seconds):.1f})')
    if t.search.root_flagged:
        print(f'moves whose best candidates met a remembered situation: {t.search.root_flagged}')
    if t.memory is not None:
        print(f'episodes remembered at the end: {len(t.memory)}')
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
    ap.add_argument('--arm', default='lookahead', choices=sorted(ARMS))
    ap.add_argument('--seed', default=None, help='play: a different seed, to split an arm over processes')
    ap.add_argument('--name', default=None, help='play: the database copy\'s name (default: the arm)')
    a = ap.parse_args()
    if a.what == 'learn':
        learn(a.work, int(a.rest[0]), a.level, a.stockfish_path, a.arm)
    else:
        play(a.work, a.rest[0], int(a.rest[1]), a.level, a.stockfish_path, a.out, a.seed, a.name)
