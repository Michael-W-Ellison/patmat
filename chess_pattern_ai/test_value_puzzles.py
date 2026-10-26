#!/usr/bin/env python3
"""Unit tests for value_puzzles and value_learner (run: python3 -m unittest test_value_puzzles)"""

import random
import unittest

from evaluate_value_puzzles import attempt
from value_learner import ValueLearner
from value_puzzles import OPERATIONS, TRUE_MEANING, count_solutions, kenken, sudoku


class TestPuzzles(unittest.TestCase):
    def test_sudoku_is_valid_and_unique(self):
        p = sudoku(1)
        s = p.solution
        for i in range(9):
            self.assertEqual(sorted(s[i]), list(range(1, 10)))
            self.assertEqual(sorted(s[r][i] for r in range(9)), list(range(1, 10)))
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                self.assertEqual(sorted(s[i + r][j + c] for r in range(3) for c in range(3)), list(range(1, 10)))
        sols = count_solutions(p.constraints, p.values, limit=3)
        self.assertEqual(len(sols), 1)
        self.assertTrue(all(sols[0][(r, c)] == s[r][c] for r in range(9) for c in range(9)))

    def test_kenken_clues_and_uniqueness(self):
        p = kenken(2, 5)
        groups = {}
        for r in range(5):
            for c in range(5):
                groups.setdefault(p.cages[r][c], []).append(p.solution[r][c])
        for cage, (target, symbol) in p.clues.items():
            self.assertEqual(OPERATIONS[TRUE_MEANING[symbol]](groups[cage]), target)
        self.assertEqual(len(count_solutions(p.constraints, p.values, limit=3)), 1)

    def test_wrong_value_ends_the_attempt(self):
        p = kenken(3, 4)
        r, c = p.legal_moves()[0]
        p.fill((r, c), p.solution[r][c] % 4 + 1)
        self.assertTrue(p.is_over() and not p.won())
        self.assertEqual(p.final_view().values, p.solution)


class TestLearner(unittest.TestCase):
    def test_discovers_sudoku_rules(self):
        learner = ValueLearner(random.Random(0))
        for i in range(6):
            attempt(sudoku(i), learner.choose, learner.end_game)
        rules = set(learner.accepted())
        for kind in ('row', 'column', 'box 3x3'):
            self.assertIn(('different', kind), rules)
        for kind in ('diagonal', 'anti-diagonal', 'box 1x9'):
            self.assertNotIn(('different', kind), rules)

    def test_discovers_what_kenken_symbols_mean(self):
        learner = ValueLearner(random.Random(0))
        for i in range(40):
            attempt(kenken(i, 6), learner.choose, learner.end_game)
        rules = set(learner.accepted())
        for symbol, meaning in (('+', 'sum'), ('x', 'product'), ('-', 'difference')):
            self.assertIn(('clue', symbol, meaning), rules)
            self.assertEqual([h for h in rules if h[0] == 'clue' and h[1] == symbol], [('clue', symbol, meaning)])
        self.assertNotIn(('different', 'cage'), rules)

    def test_solves_once_learned(self):
        learner = ValueLearner(random.Random(0))
        for i in range(6):
            attempt(sudoku(i), learner.choose, learner.end_game)
        won = sum(attempt(sudoku(100 + i), learner.choose, learner.end_game)[0] for i in range(5))
        self.assertEqual(won, 5)


if __name__ == '__main__':
    unittest.main()
