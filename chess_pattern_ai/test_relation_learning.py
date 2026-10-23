#!/usr/bin/env python3
"""Unit tests for relation_learning (run: python3 -m unittest test_relation_learning)"""

import random
import unittest

from grid_vocabulary import Grid
from relation_learning import RELATIONS, move_changes, move_lifts, relation_profile


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([[1, 1, 1, 0],
                          [0, 2, 0, 0],
                          [0, 0, 0, 0],
                          [2, 0, 0, 0]])

    def test_values(self):
        p = relation_profile(self.grid, [1], [2])
        self.assertEqual(p['pieces'], 1)            # 3 - 2
        self.assertEqual(p['largest_region'], 2)    # 3 - 1
        self.assertEqual(p['regions'], -1)          # 1 - 2
        self.assertEqual(p['longest_line'], 2)      # 3 - 1
        self.assertEqual(p['freedom'], -2)          # 3 - 5 empty neighbours
        self.assertEqual(p['capture_threat'], 1)    # (2,1) brackets (1,1); 2 cannot bracket
        self.assertEqual(p['span'], 2)              # 3 - 1
        self.assertEqual(p['sides_touched'], 0)     # top+left vs bottom+left

    def test_antisymmetric(self):
        p = relation_profile(self.grid, [1], [2])
        q = relation_profile(self.grid, [2], [1])
        self.assertTrue(all(abs(p[r] + q[r]) < 1e-12 for r in RELATIONS))

    def test_several_values_per_side(self):
        grid = Grid([[1, 2, 0], [0, 0, 7]])
        self.assertEqual(relation_profile(grid, [1, 2], [7])['pieces'], 1)

    def test_enclosed_empty(self):
        grid = Grid([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]])
        self.assertEqual(relation_profile(grid, [1], [2])['enclosed_empty'], 1)


class TestChangesAndLifts(unittest.TestCase):
    def test_move_changes(self):
        before = dict.fromkeys(RELATIONS, 0.0)
        after = dict(before, pieces=1.0, regions=-2.0)
        c = move_changes(before, after)
        self.assertEqual((c['pieces'], c['regions'], c['span']), ('+', '-', '0'))

    def test_lift_finds_the_winning_change(self):
        # The eventual winner's moves increase 'pieces'; the loser's do not
        rng = random.Random(0)
        games = []
        for g in range(60):
            winner = g % 2
            moves = []
            for ply in range(20):
                idx = ply % 2
                changes = dict.fromkeys(RELATIONS, '0')
                changes['pieces'] = '+' if idx == winner else '-'
                changes['span'] = rng.choice('+0-')     # unrelated to winning
                moves.append((idx, changes))
            games.append((winner, moves))
        lifts = move_lifts(games, random.Random(1), bootstrap=200)
        self.assertAlmostEqual(lifts[('pieces', '+')]['lift'], 0.5)
        self.assertGreater(lifts[('pieces', '+')]['ci_low'], 0)
        self.assertLess(lifts[('pieces', '-')]['ci_high'], 0)
        span = lifts[('span', '+')]
        self.assertLess(span['ci_low'], 0)
        self.assertGreater(span['ci_high'], 0)

    def test_draws_are_ignored(self):
        moves = [(0, dict.fromkeys(RELATIONS, '+'))]
        lifts = move_lifts([(None, moves)], random.Random(0), bootstrap=10)
        self.assertEqual(lifts[('pieces', '+')]['moves'], 0)


if __name__ == '__main__':
    unittest.main()
