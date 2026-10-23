#!/usr/bin/env python3
"""Summarise evaluate_arc_relation_solver.py output: usage: analyze_arc_relation_solver.py <out.json>..."""
import json, sys, collections
for path in sys.argv[1:]:
    d = json.load(open(path)); r = d['results']; n = len(r)
    ok = [x for x in r if x['status'] == 'ok']
    def cnt(key): return sum(bool(x.get(key)) for x in r)
    found = sum(x.get('found', 0) > 0 for x in r)
    amb = [x for x in ok if x.get('distinct_predictions', 0) > 1]
    print(f"\n=== {d['split']}: {n} tasks, {d['seconds']}s, timeouts {sum(x['status']=='timeout' for x in r)}")
    print(f"tasks with a consistent candidate: {found}   upper bound (some consistent candidate correct): {cnt('oracle')} = {100*cnt('oracle')/n:.1f}%")
    for o in ('game', 'fixed', 'reverse_game'):
        print(f"  {o:<13} 1 attempt: {cnt(o):>3} = {100*cnt(o)/n:4.1f}%   2 attempts: {cnt(o+'_top2'):>3} = {100*cnt(o+'_top2')/n:4.1f}%")
    rm = sum(x.get('random_mean', 0) for x in r)
    print(f"  random order  1 attempt: {rm:5.1f} = {100*rm/n:4.1f}% (mean of 20 orders per task)")
    print(f"ambiguous tasks (consistent candidates disagree on the test): {len(amb)}; solved there by game {sum(bool(x['game']) for x in amb)}, fixed {sum(bool(x['fixed']) for x in amb)}, reverse {sum(bool(x['reverse_game']) for x in amb)}, random {sum(x['random_mean'] for x in amb):.1f}, upper bound {sum(bool(x['oracle']) for x in amb)}")
    old = sum(bool(x.get('old_solved')) for x in r)
    combined = sum((bool(x.get('game')) if x.get('found', 0) > 0 else bool(x.get('old_solved'))) for x in r)
    either = sum(bool(x.get('game')) or bool(x.get('old_solved')) for x in r)
    print(f"existing solver: {old} = {100*old/n:.1f}%   combined (new answer if any, else existing): {combined} = {100*combined/n:.1f}%   solved by either: {either}")
    by = collections.Counter(x['game_name'].split(':')[0].split('+')[0] for x in r if x.get('game'))
    print('solved (game order) by transformation family:', dict(by.most_common()))
    print('slowest task:', max((x.get('seconds', 0), x['task']) for x in r))
