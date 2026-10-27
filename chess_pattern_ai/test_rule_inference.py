#!/usr/bin/env python3
"""Unit tests for rule_inference (run: python3 -m unittest test_rule_inference)"""

import unittest

import numpy as np

from rule_inference import AUTOMATA, RuleLearner, describe, random_grid


def trained(name, frames, seed=0, run=5):
    step, states, _ = AUTOMATA[name]
    rng = np.random.default_rng(seed)
    learner = RuleLearner()
    g, t = None, 0
    for i in range(frames):
        if i % run == 0:
            g, t = random_grid(rng, (24, 24), states), 0
        after = step(g, t)
        learner.observe(g, after)
        g, t = after, t + 1
    return learner, step, states, rng


class TestRuleInference(unittest.TestCase):
    def test_learns_life_exactly(self):
        learner, step, states, rng = trained('Life B3/S23', 50)
        for _ in range(10):
            g = random_grid(rng, (24, 24), states)
            pred, h = learner.predict(g)
            self.assertEqual(h, ('own + 8 counts', True))
            self.assertTrue(np.array_equal(pred, step(g, 0)))

    def test_multi_state_rule_uses_one_state_count(self):
        learner, step, states, rng = trained("Brian's Brain (3 states)", 50)
        g = random_grid(rng, (24, 24), states)
        pred, h = learner.predict(g)
        self.assertEqual(h, ('own + 8 count of 1', True))

    def test_rules_outside_the_family_are_rejected(self):
        for name in ('Critters (2x2 blocks, alternating)', 'Life / HighLife alternating by step'):
            learner, step, states, rng = trained(name, 10)
            self.assertEqual(learner.consistent(), [], name)
            self.assertEqual(learner.predict(random_grid(rng, (24, 24), states)), (None, None))

    def test_unseen_or_ambiguous_gives_unknown(self):
        learner = RuleLearner()
        a = np.zeros((4, 4), dtype=int)
        learner.observe(a, a)                       # only empty neighbourhoods seen
        b = a.copy()
        b[1, 1] = 1
        self.assertEqual(learner.predict(b), (None, None))
        pred, _ = learner.predict(a)
        self.assertTrue(np.array_equal(pred, a))

    def test_descriptions(self):
        a = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 2]])
        centre = 4                                   # row-major index of (1, 1)
        self.assertEqual(describe(a, '3x3 counts', False, 3)[centre], (3, 1))
        self.assertEqual(describe(a, 'own + 4 counts', False, 3)[centre], (1, 2, 0))
        self.assertEqual(describe(a, 'own + 8 count of 2', False, 3)[centre], (1, 1))
        self.assertEqual(describe(a, 'own + 4 in order', False, 3)[centre], (1, 1, 0, 1, 0))


if __name__ == '__main__':
    unittest.main()
