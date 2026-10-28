#!/usr/bin/env python3
"""
Evaluate arc_pattern_solver on an ARC-AGI-1 split.

usage: evaluate_arc_pattern_solver.py <code_dir> <data_dir> <split> <out.json> [limit] [part parts]
  part parts: only every parts-th task from the part-th (to run a split in parallel processes)
  data_dir: the ARC-AGI repository's data/ directory (training/, evaluation/)

Per task: patterns proposed, consistent patterns, distinct predictions,
whether attempt 1 / attempts 1-2 are exactly right, whether any consistent
pattern is right, and the families of the right patterns. Prints a summary.
"""
import sys, os, json, time, signal
code_dir, data_dir, split, out_path = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'all' else None
part, parts = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (0, 1)
sys.path.insert(0, code_dir)
import numpy as np
from arc_pattern_solver import find_patterns, rank
from arc_relation_solver import load_relation_lifts
class Timeout(Exception): pass
def alarm(*_): raise Timeout()
signal.signal(signal.SIGALRM, alarm)
lifts = load_relation_lifts(os.path.join(code_dir, 'learned_relations.json'))
files = sorted(f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json'))[:limit][part::parts]
res = []; t_all = time.time()
for f in files:
    task = json.load(open(os.path.join(data_dir, split, f)))
    rec = {'task': f[:-5]}; t0 = time.time()
    try:
        signal.alarm(180)
        consistent, proposed = find_patterns(task)
        ranked = rank(consistent, task, lifts)
        signal.alarm(0)
    except Timeout:
        rec.update(status='timeout'); res.append(rec); continue
    truth = [np.array(t['output']) for t in task['test']]
    right = lambda e: all(p.shape == t.shape and np.array_equal(p, t) for p, t in zip(e['preds'], truth))
    rec.update(status='ok', proposed=proposed, consistent=len(consistent), distinct=len(ranked),
               solved1=bool(ranked) and right(ranked[0]), solved2=any(right(e) for e in ranked[:2]),
               oracle=any(right(e) for e in ranked),
               family=ranked[0]['family'] if ranked else None, pattern=ranked[0]['name'] if ranked else None,
               right_families=sorted({e['family'] for e in consistent if right(e)}),
               seconds=round(time.time() - t0, 2))
    res.append(rec)
json.dump({'split': split, 'seconds': round(time.time() - t_all), 'results': res}, open(out_path, 'w'), indent=0)
r = res; n = len(r)
c = lambda k: sum(bool(x.get(k)) for x in r)
print(f"{split}: {n} tasks, {round(time.time()-t_all)}s, timeouts {sum(x['status']=='timeout' for x in r)}; "
      f"solved 1 attempt {c('solved1')} ({100*c('solved1')/n:.1f}%), 2 attempts {c('solved2')} ({100*c('solved2')/n:.1f}%), "
      f"upper bound {c('oracle')}; slowest {max((x.get('seconds',0), x['task']) for x in r)}")
