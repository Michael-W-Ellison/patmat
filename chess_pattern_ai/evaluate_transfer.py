#!/usr/bin/env python3
"""
Does generic pattern data speed up learning a new game? (pattern_store.py)

    python3 evaluate_transfer.py fresh <dir> <game> [games] [seed]
    python3 evaluate_transfer.py transfer <dir> <game> [games] [seed] <source game> ...
    python3 evaluate_transfer.py report <dir>

fresh: a new agent plays the game (of arc3_games.py) `games` times, with no
stored patterns; the game's specific entry is saved in <dir>/fresh-<game>.
transfer: a new agent plays the game the same number of times (the same
seeds), using the generic store built from the specific entries of the
source games' fresh sessions (copied to <dir>/to-<game>, which then also
gets the game's own entry). Each run's results go to <dir>/results.
report: per game, the fresh and the transfer results in blocks of BLOCK
games (board games: wins, losses, draws; Tetris games: the mean score),
the first game won (board games), and the generic patterns the transfer
run used, with how each turned out.

Board games are one game per episode (KInARow: 'result'); Tetris games
(names starting with 'tetris') end at the first loss and score the rows
cleared.
"""

import glob
import json
import logging
import os
import random
import shutil
import sys
import time

from arc3_games import GAMES, local_env
from evaluate_arc3 import play
from object_agent import ObjectAgent
from pattern_store import PatternStore

BLOCK = 10
MAX_ACTIONS = {'board': 500, 'tetris': 5000}


def _kind(game) -> str:
    return 'tetris' if game.startswith('tetris') else 'board'


def session(game, games, seed, store: PatternStore, sources=None) -> dict:
    agent = ObjectAgent(random.Random(seed))
    if sources is not None:
        agent.use_generic(store.generic())
    out = []
    t0 = time.time()
    for i in range(games):
        env = local_env(GAMES[game](seed=seed * 1000 + i))
        agent.new_episode()
        _, _, _, _, score = play(env, agent.act, MAX_ACTIONS[_kind(game)], agent, stop_on_loss=True)
        out.append(score if _kind(game) == 'tetris' else (getattr(env.game, 'result', None) or 'unfinished'))
        print(f'{game} {i + 1}: {out[-1]}', flush=True)
    store.save(game, agent, games)
    use = {}
    if sources is not None:
        with open(os.path.join(store.path, 'generic.json')) as f:
            for kj, games_use in json.load(f)['use'].items():
                if game in games_use:
                    use[kj] = games_use[game]
    return {'game': game, 'seed': seed, 'results': out, 'seconds': round(time.time() - t0),
            'sources': sources, 'generic offered': len(agent.generic), 'used': use,
            'learned': sorted(str(k) for k in agent.facts())}


def _blocks(game, results) -> str:
    parts = []
    for i in range(0, len(results), BLOCK):
        b = results[i:i + BLOCK]
        span = f'{i + 1}-{i + len(b)}'
        if _kind(game) == 'tetris':
            parts.append(f'{span}: {sum(b) / len(b):.1f}')
        else:
            parts.append(f'{span}: {b.count("win")}-{b.count("loss")}-{b.count("draw")}'
                         + (f' ({b.count("unfinished")} unfinished)' if b.count('unfinished') else ''))
    return '; '.join(parts)


def report(path):
    runs = {}
    for f in glob.glob(os.path.join(path, 'results', '*.json')):
        with open(f) as fh:
            r = json.load(fh)
        runs.setdefault(r['game'], {})['transfer' if r['sources'] is not None else 'fresh'] = r
    for game, by in sorted(runs.items()):
        print(f'== {game}' + (' (mean score per block)' if _kind(game) == 'tetris' else ' (wins-losses-draws per block)'))
        for name in ('fresh', 'transfer'):
            r = by.get(name)
            if r is None:
                continue
            res = r['results']
            extra = ''
            if _kind(game) == 'board':
                first = next((i + 1 for i, x in enumerate(res) if x == 'win'), None)
                extra = f', first win: game {first}'
            else:
                extra = f', all {len(res)}: {sum(res) / len(res):.1f}'
            print(f'   {name:8}: {_blocks(game, res)}{extra} ({r["seconds"]} s)')
            if name == 'transfer':
                print(f'      from {", ".join(r["sources"])}: {r["generic offered"]} generic patterns offered; used:')
                for kj, outcome in sorted(r['used'].items()):
                    print(f'         {tuple(json.loads(kj))}: {outcome}')


def main():
    logging.disable(logging.INFO)
    phase, path = sys.argv[1], sys.argv[2]
    if phase == 'report':
        report(path)
        return
    game = sys.argv[3]
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    os.makedirs(os.path.join(path, 'results'), exist_ok=True)
    if phase == 'fresh':
        store_dir = os.path.join(path, f'fresh-{game}')
        shutil.rmtree(store_dir, ignore_errors=True)
        r = session(game, games, seed, PatternStore(store_dir))
    else:
        sources = sys.argv[6:]
        store_dir = os.path.join(path, f'to-{game}')
        shutil.rmtree(store_dir, ignore_errors=True)
        store = PatternStore(store_dir)
        for s in sources:
            shutil.copy(os.path.join(path, f'fresh-{s}', 'specific', f'{s}.json'),
                        os.path.join(store_dir, 'specific', f'{s}.json'))
        store.refresh()
        r = session(game, games, seed, store, sources)
    with open(os.path.join(path, 'results', f'{phase}-{game}.json'), 'w') as f:
        json.dump(r, f, indent=1)


if __name__ == '__main__':
    main()
