#!/usr/bin/env python3
"""
Does generic pattern data speed up learning a new game? (pattern_store.py)

    python3 evaluate_transfer.py fresh <dir> <game> [games] [seed]
    python3 evaluate_transfer.py transfer <dir> <game> [games] [seed] <source game> ...
    python3 evaluate_transfer.py meta <dir> <game> [games] [seed] <meta dir> [excluded game ...]
    python3 evaluate_transfer.py report <dir>

fresh: a new agent plays the game (of arc3_games.py) `games` times, with no
stored patterns; the game's specific entry is saved in <dir>/fresh-<game>.
transfer: a new agent plays the game the same number of times (the same
seeds), using the generic store built from the specific entries of the
source games' fresh sessions (copied to <dir>/to-<game>, which then also
gets the game's own entry). Each run's results go to <dir>/results.
meta: a new agent plays the game the same number of times using only the
universal elements (meta_patterns) of the meta store in <meta dir>, with
the instances of the game itself and of the excluded games (the game's
own family) left out: transfer across kinds of game (and from ARC tasks).
report: per game, the fresh, transfer and meta results in blocks of BLOCK
games (board games: wins, losses, draws; Tetris games: the mean score),
the first game won (board games), and the generic patterns the transfer
run used, with how each turned out.

Board games are one game per episode (KInARow: 'result'); Tetris games
(names starting with 'tetris') end at the first loss and score the rows
cleared. Other games ('maze', 'maze-b', or a game of the ARC-AGI toolkit
in the directory named by the environment variable ARC3_ENVS, such as
the example games) are played for at most MAX_ACTIONS['levels'] actions
per episode, a lost level reset; each episode reports the levels
completed, the actions taken and the levels lost.
"""

import glob
import json
import logging
import os
import random
import shutil
import sys
import time

from arc3_games import BOARD_GAMES, GAMES, local_env
from evaluate_arc3 import play
from object_agent import ObjectAgent
from pattern_store import PatternStore

BLOCK = 10
MAX_ACTIONS = {'board': 500, 'tetris': 5000, 'levels': 2000}


def _kind(game) -> str:
    return 'tetris' if game.startswith('tetris') else 'board' if game in BOARD_GAMES else 'levels'


def _env(game, seed):
    if game in GAMES:
        return local_env(GAMES[game](seed=seed))
    from arc_agi import Arcade, OperationMode
    return Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ['ARC3_ENVS']).make(game)


def session(game, games, seed, store: PatternStore, sources=None, meta=None) -> dict:
    agent = ObjectAgent(random.Random(seed))
    if sources is not None and meta is None:
        agent.use_generic(store.generic())
    if meta is not None:
        agent.use_meta(meta)
    out = []
    t0 = time.time()
    for i in range(games):
        env = _env(game, seed * 1000 + i)
        agent.new_episode()
        kind = _kind(game)
        done, total, per_level, losses, score = play(env, agent.act, MAX_ACTIONS[kind], agent,
                                                     stop_on_loss=kind != 'levels')
        if kind == 'levels':
            out.append({'levels': done, 'of': total, 'actions': sum(per_level), 'losses': losses})
        else:
            out.append(score if kind == 'tetris' else (getattr(env.game, 'result', None) or 'unfinished'))
        print(f'{game} {i + 1}: {out[-1]}', flush=True)
    store.save(game, agent, games)
    use = {}
    if meta is not None:
        use = {json.dumps([e]): outcome[game] for e, outcome in store.meta.data['use'].items() if game in outcome}
    elif sources is not None:
        with open(os.path.join(store.path, 'generic.json')) as f:
            for kj, games_use in json.load(f)['use'].items():
                if game in games_use:
                    use[kj] = games_use[game]
    return {'game': game, 'seed': seed, 'results': out, 'seconds': round(time.time() - t0),
            'sources': sources, 'generic offered': len(agent.generic), 'used': use,
            'phase': 'meta' if meta is not None else 'transfer' if sources is not None else 'fresh',
            'meta offered': sorted(meta or {}),
            'learned': sorted(str(k) for k in agent.facts())}


def _blocks(game, results) -> str:
    if _kind(game) == 'levels':
        return '; '.join(f'{i + 1}: {r["levels"]}/{r["of"]} levels in {r["actions"]} actions, {r["losses"]} lost'
                         for i, r in enumerate(results))
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
        runs.setdefault(r['game'], {})[r.get('phase') or ('transfer' if r['sources'] is not None else 'fresh')] = r
    for game, by in sorted(runs.items()):
        print(f'== {game}' + {'tetris': ' (mean score per block)', 'board': ' (wins-losses-draws per block)',
                              'levels': ' (per episode)'}[_kind(game)])
        for name in ('fresh', 'transfer', 'meta'):
            r = by.get(name)
            if r is None:
                continue
            res = r['results']
            extra = ''
            if _kind(game) == 'board':
                first = next((i + 1 for i, x in enumerate(res) if x == 'win'), None)
                extra = f', first win: game {first}'
            elif _kind(game) == 'levels':
                extra = ''
            else:
                extra = f', all {len(res)}: {sum(res) / len(res):.1f}'
            print(f'   {name:8}: {_blocks(game, res)}{extra} ({r["seconds"]} s)')
            if name == 'meta':
                print(f'      universal elements only, from the meta store without {", ".join(r["sources"])}: '
                      f'{", ".join(r["meta offered"])} offered; used:')
                for kj, outcome in sorted(r['used'].items()):
                    print(f'         {json.loads(kj)[0]}: {outcome}')
            elif name == 'transfer':
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
    elif phase == 'meta':
        from meta_patterns import MetaStore
        excluded = [game] + sys.argv[7:]
        store_dir = os.path.join(path, f'meta-{game}')
        shutil.rmtree(store_dir, ignore_errors=True)
        elements = MetaStore(sys.argv[6]).elements(exclude={f'game:{g}' for g in excluded})
        r = session(game, games, seed, PatternStore(store_dir), excluded, elements)
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
