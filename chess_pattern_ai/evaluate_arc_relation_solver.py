#!/usr/bin/env python3
"""
Evaluate arc_relation_solver on an ARC-AGI-1 split (plan step 3).

usage: evaluate_arc_relation_solver.py <code_dir> <data_dir> <split> <out.json> [old_solver.csv]
  data_dir: the ARC-AGI repository's data/ directory (training/, evaluation/)
  old_solver.csv: optional per-task results of the existing ARCSolver
                  (columns task, solved) to report the combination

Per task: candidates tried, consistent candidates, distinct test
predictions, whether any consistent candidate is right (upper bound),
and whether the game-learned / fixed / reversed / random orderings pick
a right one (1 and 2 attempts). Summarise with analyze_arc_relation_solver.py.
"""
import sys, os, json, time, random, signal
code_dir, data_dir, split, out_path = sys.argv[1:5]
old_csv = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, code_dir)
import numpy as np
from arc_relation_solver import consistent_candidates, solves, choose, load_relation_lifts

class Timeout(Exception): pass
def alarm(*_): raise Timeout()
signal.signal(signal.SIGALRM, alarm)

lifts = load_relation_lifts(os.path.join(code_dir, 'learned_relations.json'))
old = {}
if old_csv:
    import csv
    old = {r['task']: r['solved'] == '1' for r in csv.DictReader(open(old_csv))}
results = []
files = sorted(f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json'))
t_all = time.time()
for f in files:
    task = json.load(open(os.path.join(data_dir, split, f)))
    tid = f[:-5]
    t0 = time.time()
    rec = {'task': tid, 'old_solved': old.get(tid)}
    try:
        signal.alarm(120)
        found, n_cands = consistent_candidates(task, lifts)
        signal.alarm(0)
    except Timeout:
        rec.update(status='timeout', found=0)
        results.append(rec); continue
    rec['status'] = 'ok'
    rec['candidates'] = n_cands
    rec['found'] = len(found)
    distinct = choose(found, 'fixed', k=100)
    rec['distinct_predictions'] = len(distinct)
    rec['oracle'] = any(solves(e, task) for e in found)
    for ordering in ('game', 'fixed', 'reverse_game'):
        top = choose(found, ordering, k=2)
        rec[ordering] = bool(top) and solves(top[0], task)
        rec[ordering + '_top2'] = any(solves(e, task) for e in top)
        rec[ordering + '_name'] = top[0]['name'] if top else None
    rng = random.Random(f'{split}-{tid}')
    rand = [ (lambda t: bool(t) and solves(t[0], task))(choose(found, 'random', rng=rng, k=1)) for _ in range(20)]
    rec['random_mean'] = sum(rand) / len(rand)
    rec['seconds'] = round(time.time() - t0, 2)
    results.append(rec)
json.dump({'split': split, 'seconds': round(time.time() - t_all), 'results': results}, open(out_path, 'w'), indent=0)
