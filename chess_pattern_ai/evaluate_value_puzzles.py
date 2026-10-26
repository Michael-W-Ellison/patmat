#!/usr/bin/env python3
"""
Does the value learner discover the rules of Sudoku and KenKen, and solve
them? (value_puzzles.py, value_learner.py; every puzzle has one solution)

For each puzzle type a fresh learner plays a sequence of puzzles, learning
only from the solutions shown after each attempt, and a random player
(a random value in a random empty cell) plays the same puzzles. Reported:
the rules accepted (and after how many puzzles), the share solved overall
and in the second half, and how many moves were deduced or guessed.

Usage: python3 evaluate_value_puzzles.py [puzzles per type] [seed]
"""

import copy
import random
import sys

from value_learner import ValueLearner
from value_puzzles import PUZZLES


def attempt(puzzle, choose, end=None):
    while not puzzle.is_over():
        pos, value = choose(puzzle.observation(), puzzle.legal_moves())
        puzzle.fill(pos, value)
    if end is not None:
        end(puzzle.final_view())
    return puzzle.won(), puzzle.progress()


def run(kind, n, seed):
    rng = random.Random(seed)
    learner = ValueLearner(random.Random(seed))
    solved = late = r_solved = 0
    progress = 0.0
    for i in range(n):
        puzzle = PUZZLES[kind](seed * 100_003 + i)
        won, prog = attempt(copy.deepcopy(puzzle), learner.choose, learner.end_game)
        solved += won
        progress += prog
        late += won and i >= n // 2
        values = puzzle.n
        won, _ = attempt(puzzle, lambda obs, moves: (rng.choice(moves), rng.randint(1, values)))
        r_solved += won
    return learner.summary(), solved / n, late / (n - n // 2), progress / n, r_solved / n


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for kind in PUZZLES:
        summary, solved, late, progress, r_solved = run(kind, n, seed)
        print(f'== {kind}: accepted {summary["accepted"]}')
        print(f'   solved {100 * solved:.1f}% (second half {100 * late:.1f}%), cells filled {100 * progress:.1f}%; '
              f'random solved {100 * r_solved:.1f}%; moves {summary["moves"]}', flush=True)


if __name__ == '__main__':
    main()
