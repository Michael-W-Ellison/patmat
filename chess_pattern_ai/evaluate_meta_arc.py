#!/usr/bin/env python3
"""
Universal elements (meta_patterns.py) and the ARC pattern solver

    python3 evaluate_meta_arc.py profile <data_dir> <split> <out.json> [part] [parts]
    python3 evaluate_meta_arc.py build <meta_dir> <pattern store dir ...> -- <training profile ...>
    python3 evaluate_meta_arc.py report <meta_dir> <training profile ...> -- <evaluation profile ...>

profile: for every task of the split (or every parts-th task from part),
each family of arc_pattern_solver searched alone (single patterns, no
combinations): seconds, patterns proposed, and the consistent patterns
(name, complexity, whether the test prediction is right).

build: two meta stores (meta_patterns.MetaStore): <meta_dir>/games with
the instances of the games in the pattern stores given (their specific
entries), and <meta_dir>/all with, in addition, the elements of the
simplest right pattern of every training task some single family solves.

report: the solver searches its families in an order and, in the anytime
mode compared here, stops at the first family that gives a consistent
pattern, answering with that family's simplest patterns (2 attempts).
For the evaluation tasks it compares the family orders
    fixed        the solver's own listing (A, B, C, ...)
    games        by the weight of each family's elements in a meta store
                 built from games only (<meta_dir>/games)
    games+arc    the same with the elements of the training tasks the
                 solver solved added (<meta_dir>/all)
    arc          the training tasks' elements alone
    random       the mean over 20 random orders
and reports: tasks answered right (attempt 1, attempts 1-2), and the
patterns proposed and seconds spent before the first consistent family
(over the tasks some family fits). It also reports the full search (all
families, the solver's normal mode) with ties among the simplest patterns
broken by the elements' weights instead of in listing order.
"""

import json
import os
import random
import signal
import sys
import time

import numpy as np


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def profile(data_dir, split, out_path, part=0, parts=1):
    from arc_pattern_solver import FAMILIES, find_patterns
    signal.signal(signal.SIGALRM, _alarm)
    files = sorted(f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json'))[part::parts]
    out = []
    for f in files:
        task = json.load(open(os.path.join(data_dir, split, f)))
        truth = [np.array(t['output']) for t in task['test']]
        rec = {'task': f[:-5], 'families': {}}
        for fam in FAMILIES:
            t0 = time.time()
            try:
                signal.alarm(60)
                consistent, proposed = find_patterns(task, families=(fam,), combinations=False)
                signal.alarm(0)
            except Timeout:
                rec['families'][fam] = {'timeout': True, 'seconds': 60.0, 'proposed': 0, 'consistent': []}
                continue
            rec['families'][fam] = {
                'seconds': round(time.time() - t0, 3), 'proposed': proposed,
                'consistent': [{'name': e['name'], 'complexity': e['complexity'],
                                'right': all(p.shape == t.shape and np.array_equal(p, t)
                                             for p, t in zip(e['preds'], truth)),
                                'key': [p.tobytes().hex()[:64] + str(p.shape) for p in e['preds']]}
                               for e in consistent]}
        out.append(rec)
        print(f[:-5], {k: len(v['consistent']) for k, v in rec['families'].items() if v['consistent']}, flush=True)
    with open(out_path, 'w') as fh:
        json.dump(out, fh)


def build(meta_dir, store_dirs, train_files):
    from arc_pattern_solver import FAMILIES
    from meta_patterns import MetaStore
    from object_agent import ObjectAgent
    from pattern_store import PatternStore
    stores = {name: MetaStore(os.path.join(meta_dir, name)) for name in ('games', 'all')}
    for d in store_dirs:
        ps = PatternStore(d)
        for game in ps.games():
            agent = ObjectAgent()
            agent.import_state(ps.load(game)['state'])
            for m in stores.values():
                m.add_game(game, agent.facts())
    solved = 0
    for f in train_files:
        for rec in json.load(open(f)):
            right = [(e['complexity'], list(FAMILIES).index(fam), fam, e['name'])
                     for fam, v in rec['families'].items() for e in v['consistent'] if e['right']]
            if right:
                _, _, fam, name = min(right)
                stores['all'].add_arc(rec['task'], fam, name)
                solved += 1
    for name, m in stores.items():
        print(f'{name}: {len(m.data["instances"])} sources ({solved} ARC tasks in "all")')
        for element, e in sorted(m.elements().items()):
            print(f'   {element}: mechanisms {e["mechanisms"]}, {e["games"]} games, {e["arc_tasks"]} ARC tasks')


def _answers(entries, weights=None, k=2):
    """Up to k distinct predictions: simplest first; ties by element weight (if given), then listing order"""
    from meta_patterns import arc_elements
    def key(i_e):
        i, e = i_e
        w = sum(weights.get(x, 0.0) for x in arc_elements(e['family'], e['name'])) if weights else 0.0
        return (e['complexity'], -w, i)
    out, seen = [], set()
    for _, e in sorted(enumerate(entries), key=key):
        kk = tuple(e['key'])
        if kk not in seen:
            seen.add(kk)
            out.append(e)
        if len(out) == k:
            break
    return out


def anytime(rec, order, weights=None):
    """(attempt 1 right, attempts 1-2 right, proposed before the first fit, seconds before it, fitted)"""
    proposed = seconds = 0.0
    for fam in order:
        f = rec['families'][fam]
        proposed += f['proposed']
        seconds += f['seconds']
        if f['consistent']:
            ans = _answers([dict(e, family=fam) for e in f['consistent']], weights)
            return bool(ans and ans[0]['right']), any(e['right'] for e in ans), proposed, seconds, True
    return False, False, proposed, seconds, False


def full(rec, weights=None):
    entries = [dict(e, family=fam) for fam, f in rec['families'].items() for e in f['consistent']]
    ans = _answers(entries, weights)
    return bool(ans and ans[0]['right']), any(e['right'] for e in ans)


def report(meta_dir, train_files, eval_files):
    from arc_pattern_solver import FAMILIES
    from meta_patterns import MetaStore, arc_elements
    load = lambda files: [r for f in files for r in json.load(open(f))]
    train, ev = load(train_files), load(eval_files)
    orders = {'fixed': list(FAMILIES)}
    weights = {}
    for name in ('games', 'all'):
        store = MetaStore(os.path.join(meta_dir, name))
        weights[name] = store.weights()
        orders['games' if name == 'games' else 'games+arc'] = store.family_order(list(FAMILIES))
    games = {s for s in store.data['instances'] if s.startswith('game:')}
    weights['arc'] = store.weights(exclude=games)
    orders['arc'] = store.family_order(list(FAMILIES), exclude=games)
    print(f'training tasks {len(train)}, evaluation tasks {len(ev)}')
    for name, w in weights.items():
        print(f'element weights ({name}): ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(w.items(), key=lambda t: -t[1])))
    for name, order in orders.items():
        print(f'order {name}: {"".join(order)}')
    print('\nanytime search (stop at the first family that fits), evaluation tasks:')
    print(f'   {"order":10} {"right 1":>8} {"right 1-2":>9} {"fitted":>7} {"proposed before fit":>20} {"seconds before fit":>19}')
    rows = {}
    for name, order in orders.items():
        w = {'fixed': None, 'games': weights['games'], 'games+arc': weights['all'], 'arc': weights['arc']}[name]
        rows[name] = [anytime(r, order, w) for r in ev]
    rng = random.Random(0)
    rand = []
    for _ in range(20):
        order = list(FAMILIES)
        rng.shuffle(order)
        rand.append([anytime(r, order) for r in ev])
    for name, res in list(rows.items()) + [('random', None)]:
        if res is None:
            n1 = np.mean([sum(x[0] for x in res) for res in rand])
            n2 = np.mean([sum(x[1] for x in res) for res in rand])
            fit = [x for res in rand for x in res if x[4]]
            print(f'   {name:10} {n1:8.1f} {n2:9.1f} {len(fit) / 20:7.0f} {np.mean([x[2] for x in fit]):20.0f}'
                  f' {np.mean([x[3] for x in fit]):19.2f}')
            continue
        fit = [x for x in res if x[4]]
        print(f'   {name:10} {sum(x[0] for x in res):8d} {sum(x[1] for x in res):9d} {len(fit):7d}'
              f' {np.mean([x[2] for x in fit]):20.0f} {np.mean([x[3] for x in fit]):19.2f}')
    print('\nfull search (all families), ties among the simplest broken by element weight, evaluation tasks:')
    for name, w in (('listing order', None), ('games', weights['games']), ('games+arc', weights['all']),
                    ('arc', weights['arc'])):
        res = [full(r, w) for r in ev]
        print(f'   {name:14} right 1: {sum(x[0] for x in res)}, right 1-2: {sum(x[1] for x in res)}')


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == 'build':
        i = sys.argv.index('--')
        build(sys.argv[2], sys.argv[3:i], sys.argv[i + 1:])
    elif sys.argv[1] == 'profile':
        part = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        parts = int(sys.argv[6]) if len(sys.argv) > 6 else 1
        profile(sys.argv[2], sys.argv[3], sys.argv[4], part, parts)
    else:
        i = sys.argv.index('--')
        report(sys.argv[2], sys.argv[3:i], sys.argv[i + 1:])


if __name__ == '__main__':
    main()
