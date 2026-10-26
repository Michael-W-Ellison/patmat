#!/usr/bin/env python3
"""Unit tests for action_puzzles and effect_learner (run: python3 -m unittest test_effect_learner)"""

import copy
import random
import unittest

import numpy as np

from action_puzzles import (BLOCK, BOX, FLOOR, PLAYER, PLUS, WALL, Sokoban, _masks, lights_out, shortest,
                            solve_xor, sokoban)
from effect_learner import CellModel, EffectLearner
from evaluate_effect_learner import episode


class TestEnvironments(unittest.TestCase):
    def test_lights_out_press(self):
        env = lights_out(0)
        zero = np.zeros((5, 5), dtype=int)
        after = env.apply(zero, (0, 0))
        self.assertEqual({tuple(p) for p in np.argwhere(after)}, {(0, 0), (1, 0), (0, 1)})
        presses = solve_xor(_masks(env), env.grid.ravel())
        g = env.grid
        for a, bit in zip(env.actions(), presses):
            if bit:
                g = env.apply(g, a)
        self.assertFalse(g.any())
        self.assertEqual(int(presses.sum()), env.optimal)

    def test_sokoban_push_rules(self):
        g = np.full((5, 5), WALL)
        g[1:-1, 1:-1] = FLOOR
        g[2, 1], g[2, 2] = PLAYER, BOX
        env = Sokoban(g)
        after = env.apply(g, 'E')
        self.assertEqual((after[2, 2], after[2, 3]), (PLAYER, BOX))
        self.assertTrue(np.array_equal(env.apply(after, 'E'), after))     # box against the wall: nothing moves
        self.assertTrue(np.array_equal(env.apply(g, 'W'), g))             # player against the wall

    def test_sokoban_levels_are_solvable(self):
        for i in range(3):
            env = sokoban(i)
            self.assertEqual(shortest(env, env.grid), env.optimal)
            self.assertGreaterEqual(env.optimal, 8)


class TestLearner(unittest.TestCase):
    def play(self, make, episodes):
        learner, results = None, []
        for i in range(episodes):
            env = make(i)
            learner = learner or EffectLearner(env.goal_absent, random.Random(0))
            learner.new_episode()
            won, steps = episode(copy.deepcopy(env), learner)
            results.append((won, steps, env.optimal))
        return learner, results

    def test_learns_lights_out_and_plans_fewest_presses(self):
        for pattern in (PLUS, BLOCK):
            learner, results = self.play(lambda i: lights_out(i, pattern=pattern), 10)
            self.assertIsInstance(learner.model(), CellModel)
            self.assertEqual({tuple(map(int, o)) for o in learner.cell.offsets}, set(pattern))
            self.assertTrue(learner.cell.toggles())
            self.assertTrue(all(won and steps == best for won, steps, best in results[5:]))

    def test_learns_sokoban(self):
        learner, results = self.play(sokoban, 15)
        self.assertEqual(learner.agent_values, {5, 6})
        self.assertEqual(learner.model().name, 'agent model (k=3, one table)')
        self.assertEqual(learner.directions, {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)})
        self.assertTrue(all(won for won, _, _ in results[10:]))


if __name__ == '__main__':
    unittest.main()
