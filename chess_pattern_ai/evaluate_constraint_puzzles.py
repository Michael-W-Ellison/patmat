#!/usr/bin/env python3
"""
Does the Minesweeper constraint learner, unchanged, discover the rules of
other constraint puzzles? (see constraint_puzzles.py)

For each puzzle type a fresh learner plays a sequence of puzzles, learning
only from the views shown after each attempt, and a random player plays
the same puzzles. Reported per type: the hypotheses accepted (and after
how many puzzles), the share solved, the mean share of empty cells
uncovered, and how many moves were deduced rather than guessed.

Usage: python3 evaluate_constraint_puzzles.py [puzzles per type] [seed]
"""

import random
import sys

from constraint_puzzles import PUZZLES
from minesweeper.constraint_learner import ConstraintLearner


def attempt(puzzle, choose, end=None):
    while not puzzle.is_over():
        puzzle.reveal(choose(puzzle.observation(), puzzle.legal_moves()))
    if end is not None:
        end(puzzle.final_view())
    return puzzle.won(), puzzle.progress()


def run(kind: str, n: int, seed: int):
    make = PUZZLES[kind]
    rng = random.Random(seed)
    learner = ConstraintLearner(random.Random(seed))
    solved = progress = r_solved = r_progress = 0
    late_solved = 0
    for i in range(n):
        won, prog = attempt(make(seed * 100_003 + i), learner.choose, learner.end_game)
        solved += won
        progress += prog
        if i >= n // 2:
            late_solved += won
        won, prog = attempt(make(seed * 100_003 + i), lambda obs, moves: rng.choice(moves))
        r_solved += won
        r_progress += prog
    s = learner.summary()
    return {'kind': kind, 'accepted': s['accepted'], 'solved': solved / n, 'solved_second_half': late_solved / (n - n // 2),
            'progress': progress / n, 'random_solved': r_solved / n, 'random_progress': r_progress / n,
            'deduced': s['moves']['deduced'], 'guessed': s['moves']['guessed'],
            'hypotheses': s['hypotheses']}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for kind in PUZZLES:
        res = run(kind, n, seed)
        print(f"== {kind}: accepted {res['accepted'] or 'nothing'}")
        print(f"   solved {100 * res['solved']:.1f}% (second half {100 * res['solved_second_half']:.1f}%), "
              f"empty cells uncovered {100 * res['progress']:.1f}%; "
              f"random: solved {100 * res['random_solved']:.1f}%, uncovered {100 * res['random_progress']:.1f}%; "
              f"moves deduced {res['deduced']}, guessed {res['guessed']}")
        for h, st in res['hypotheses'].items():
            print(f"     {h:45s} confirmed {st['confirmed']:6d}  violated {st['violated']:6d}")


if __name__ == '__main__':
    main()
