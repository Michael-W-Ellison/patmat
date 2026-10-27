#!/usr/bin/env python3
"""
Can the rule learner infer the update rules of cellular automata? (rule_inference.py)

For each automaton and each amount of training (transitions observed, in
runs of up to RUN steps from random grids), a fresh learner is trained and
then asked to predict new random grids: the next frame, and ROLLOUT frames
ahead (predicting from its own predictions). Reported: the hypothesis used,
the share of exact predictions, of wrong predictions, and of "unknown".

Usage: python3 evaluate_rule_inference.py [seed]
"""

import sys

import numpy as np

from rule_inference import AUTOMATA, RuleLearner, random_grid

SIZE = (24, 24)
TRAIN = (1, 2, 5, 10, 20, 50, 100)
RUN = 5
TESTS = 50
ROLLOUT = 10


def evaluate(name, seed):
    step, states, in_family = AUTOMATA[name]
    rows = []
    for n in TRAIN:
        rng = np.random.default_rng(seed)
        learner = RuleLearner()
        g, t = None, 0
        for i in range(n):
            if i % RUN == 0:
                g, t = random_grid(rng, SIZE, states), 0
            after = step(g, t)
            learner.observe(g, after)
            g, t = after, t + 1
        one = roll = unknown = wrong = 0
        used = set()
        for _ in range(TESTS):
            g = random_grid(rng, SIZE, states)
            pred, h = learner.predict(g)
            if pred is None:
                unknown += 1
                continue
            used.add(f'{h[0]}, {"wrapped" if h[1] else "zero edge"}')
            right = np.array_equal(pred, step(g, 0))
            one += right
            wrong += not right
            truth, guess, ok = g, g, True
            for t in range(ROLLOUT):
                truth = step(truth, t)
                guess, _ = learner.predict(guess)
                if guess is None or not np.array_equal(guess, truth):
                    ok = False
                    break
            roll += ok
        rows.append((n, one / TESTS, wrong / TESTS, roll / TESTS, unknown / TESTS, sorted(used),
                     len(learner.consistent())))
    return in_family, rows


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for name in AUTOMATA:
        in_family, rows = evaluate(name, seed)
        print(f'== {name} ({"in" if in_family else "outside"} the hypothesis family)')
        for n, one, wrong, roll, unknown, used, alive in rows:
            print(f'   {n:2d} frames: next frame exact {100 * one:5.1f}% wrong {100 * wrong:5.1f}% '
                  f'unknown {100 * unknown:5.1f}%; {ROLLOUT} frames exact {100 * roll:5.1f}%; '
                  f'consistent hypotheses {alive:2d}, used {used}')


if __name__ == '__main__':
    main()
