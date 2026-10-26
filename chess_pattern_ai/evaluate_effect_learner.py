#!/usr/bin/env python3
"""
Does the effect learner learn what actions do and then solve the puzzles?
(action_puzzles.py, effect_learner.py)

For each environment a fresh learner plays a sequence of episodes, keeping
what it learned between them; a random player plays the same episodes.
Reported: the model used, the share of episodes solved overall and in the
second half, and in solved second-half episodes the actions used against
the shortest solution.

Usage: python3 evaluate_effect_learner.py [episodes] [seed]
"""

import copy
import random
import sys

from action_puzzles import ENVIRONMENTS
from effect_learner import EffectLearner


def episode(env, learner=None, rng=None):
    while not env.is_over():
        grid = env.observation()
        actions = env.actions()
        action = learner.act(grid, actions) if learner else rng.choice(actions)
        after, _ = env.step(action)
        if learner:
            learner.after_step(grid, action, after)
    return env.won(), env.steps


def run(kind, n, seed):
    rng = random.Random(seed)
    learner = None
    solved = late = r_solved = 0
    ratios = []
    for i in range(n):
        env = ENVIRONMENTS[kind](seed * 100_003 + i)
        if learner is None:
            learner = EffectLearner(env.goal_absent, random.Random(seed))
        learner.new_episode()
        won, steps = episode(copy.deepcopy(env), learner)
        solved += won
        if i >= n // 2:
            late += won
            if won:
                ratios.append(steps / env.optimal)
        won, _ = episode(env, rng=rng)
        r_solved += won
    m = learner.model()
    return (m.name if m else None, solved / n, late / (n - n // 2),
            sum(ratios) / len(ratios) if ratios else None, r_solved / n, learner.moves_by)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for kind in ENVIRONMENTS:
        model, solved, late, ratio, r_solved, moves = run(kind, n, seed)
        print(f'== {kind}: model {model}')
        print(f'   solved {100 * solved:.1f}% (second half {100 * late:.1f}%), '
              f'actions / shortest in solved second-half episodes {ratio if ratio is None else round(ratio, 2)}; '
              f'random solved {100 * r_solved:.1f}%; moves {moves}', flush=True)


if __name__ == '__main__':
    main()
