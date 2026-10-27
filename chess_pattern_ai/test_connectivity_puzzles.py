#!/usr/bin/env python3
"""Unit tests for global_constraints, connectivity_puzzles and the learner's global rules
(run: python3 -m unittest test_connectivity_puzzles)"""

import random
import unittest

import global_constraints as gc
from connectivity_puzzles import DOT, NURIKABE_RULES, SLITHERLINK_RULES, nurikabe, slitherlink, solve
from constraint_puzzles import BinaryPuzzle, placement
from evaluate_constraint_puzzles import attempt
from minesweeper.constraint_learner import ConstraintLearner


class TestGlobalConstraints(unittest.TestCase):
    def test_holds(self):
        mines = {(0, 0), (0, 1), (1, 1)}
        empty = {(1, 0), (2, 0), (2, 1)}
        self.assertTrue(gc.holds(('connected', 'mine', 'sides'), mines, empty, {}, {}))
        self.assertFalse(gc.holds(('connected', 'mine', 'sides'), {(0, 0), (2, 1)}, empty, {}, {}))
        self.assertTrue(gc.holds(('no 2x2', 'mine', ''), mines, empty, {}, {}))
        self.assertFalse(gc.holds(('no 2x2', 'mine', ''), mines | {(1, 0)}, {(2, 0), (2, 1)}, {}, {}))
        # the one island is (1,0), (2,0), (2,1), with its clue on (2,0): it must read 3
        self.assertTrue(gc.holds(('islands', 'not-mine', 'one clue = size'), mines, {(1, 0), (2, 1)}, {(2, 0): 3}, {}))
        self.assertFalse(gc.holds(('islands', 'not-mine', 'one clue = size'), mines, {(1, 0), (2, 1)}, {(2, 0): 2}, {}))

    def test_consistent_detects_impossible_connection(self):
        # two sea cells walled apart by island cells cannot be connected
        mines, empty = {(0, 0), (0, 2)}, {(0, 1), (1, 1)}
        unknown = {(1, 0), (1, 2)}
        self.assertFalse(gc.consistent(('connected', 'mine', 'sides'), mines, empty, unknown, {}, {}))
        self.assertTrue(gc.consistent(('connected', 'mine', 'sides'), mines, {(0, 1)}, unknown | {(1, 1)}, {}, {}))

    def test_evidence_discounts_rules_that_hold_by_chance(self):
        p = placement(0)
        ok, surprise = gc.evidence(('connected', 'not-mine', 'sides'), p.filled, p.board - p.filled, {}, {},
                                   random.Random(0))
        self.assertTrue(ok)
        self.assertLess(surprise, 0.2)
        n = nurikabe(0)
        ok, surprise = gc.evidence(('connected', 'mine', 'sides'), n.filled, n.board - n.filled, n.clues, {},
                                   random.Random(0))
        self.assertTrue(ok)
        self.assertGreater(surprise, 0.8)


class TestPuzzles(unittest.TestCase):
    def test_unique_and_valid(self):
        for i in range(3):
            p = nurikabe(i)
            empty = p.board - p.filled
            self.assertTrue(all(gc.holds(r, p.filled, empty, p.clues, {}) for r in NURIKABE_RULES))
            known = {q: True for q in p.shown_filled} | {q: False for q in p.shown_empty}
            self.assertEqual(solve(p.board, p.clues, {}, NURIKABE_RULES, {}, known, limit=3), [p.filled])
            s = slitherlink(i)
            self.assertTrue(all(gc.holds(r, s.filled, s.board - s.filled, s.clues, s.markers) for r in SLITHERLINK_RULES))
            counts = {c: [(c[0] + dr, c[1] + dc) for dr, dc in gc.SIDES] for c in s.clues}
            self.assertEqual(solve(s.board, s.clues, s.markers, SLITHERLINK_RULES, counts, {}, limit=3), [s.filled])


class TestLearner(unittest.TestCase):
    def learn(self, make, n):
        learner = ConstraintLearner(random.Random(0))
        results = [attempt(make(i), learner.choose, learner.end_game)[0] for i in range(n)]
        return learner, results

    def test_discovers_nurikabe(self):
        learner, results = self.learn(nurikabe, 30)
        self.assertEqual(set(learner.accepted()), set(NURIKABE_RULES))
        self.assertTrue(all(results[20:]))

    def test_attempt_with_no_moves_keeps_islands_rule(self):
        # every island is one numbered cell: the puzzle is won before any move, and the learner never
        # sees the board during the attempt; the final view must still give the board (not the clue cells)
        sea = {(r, c) for r in range(3) for c in range(3)} - {(0, 0), (2, 2)}
        clues = {(0, 0): 1, (2, 2): 1}
        puzzle = BinaryPuzzle(3, 3, sea, clues, sea)
        learner = ConstraintLearner(random.Random(0))
        attempt(puzzle, learner.choose, learner.end_game)
        rule = ('islands', 'not-mine', 'one clue = size')
        self.assertEqual(learner.hypotheses[rule]['violated'], 0)

    def test_discovers_slitherlink(self):
        learner, results = self.learn(slitherlink, 30)
        accepted = set(learner.accepted())
        self.assertTrue(set(SLITHERLINK_RULES) <= accepted)
        self.assertIn(('count', '4 neighbours', 'mine'), accepted)
        self.assertNotIn(('degree', DOT, frozenset({2})), accepted)
        self.assertTrue(all(results[25:]))


if __name__ == '__main__':
    unittest.main()
