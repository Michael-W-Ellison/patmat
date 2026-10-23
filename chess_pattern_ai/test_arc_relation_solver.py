#!/usr/bin/env python3
"""Unit tests for arc_relation_solver (run: python3 -m unittest test_arc_relation_solver)"""

import unittest

import numpy as np

from arc_relation_solver import (choose, connect, consistent_candidates, crop, extend_lines,
                                 fill_enclosed, gravity, halo, learn_color_map, ray,
                                 select_objects, solves, symmetry_complete)

A = np.array


class TestTransforms(unittest.TestCase):
    def test_fill_enclosed(self):
        a = A([[0, 3, 3, 3], [0, 3, 0, 3], [0, 3, 3, 3]])
        self.assertEqual(fill_enclosed(a, 0, 4, None, False).tolist(),
                         [[0, 3, 3, 3], [0, 3, 4, 3], [0, 3, 3, 3]])

    def test_connect(self):
        a = A([[2, 0, 0, 2], [0, 0, 0, 0], [5, 0, 0, 0]])
        self.assertEqual(connect(a, 0, None, False).tolist(),
                         [[2, 2, 2, 2], [0, 0, 0, 0], [5, 0, 0, 0]])

    def test_extend_lines_and_ray(self):
        a = A([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        self.assertEqual(extend_lines(a, 0)[1].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(ray(A([[0, 7, 0, 0]]), 0, 7, 'E').tolist(), [[0, 7, 7, 7]])

    def test_halo(self):
        a = A([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        self.assertEqual(halo(a, 0, 2, False).tolist(), [[0, 2, 0], [2, 5, 2], [0, 2, 0]])

    def test_objects(self):
        a = A([[1, 1, 0, 2], [1, 0, 0, 0]])
        self.assertEqual(select_objects(a, 0, 'keep_largest', False).tolist(), [[1, 1, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(crop(a, 0, 'smallest', False).tolist(), [[2]])

    def test_gravity(self):
        a = A([[1, 0], [0, 0], [0, 2]])
        self.assertEqual(gravity(a, 0, 'S').tolist(), [[0, 0], [0, 0], [1, 2]])
        self.assertEqual(gravity(a, 0, 'E').tolist(), [[0, 1], [0, 0], [0, 2]])

    def test_symmetry_complete(self):
        a = A([[1, 2, 0], [3, 0, 0]])
        self.assertEqual(symmetry_complete(a, 0, 'lr').tolist(), [[1, 2, 1], [3, 0, 3]])

    def test_color_map(self):
        pairs = [(A([[1, 2]]), A([[3, 2]])), (A([[1, 1]]), A([[3, 3]]))]
        self.assertEqual(learn_color_map(pairs), {1: 3, 2: 2})
        self.assertIsNone(learn_color_map([(A([[1, 1]]), A([[3, 4]]))]))


class TestSolve(unittest.TestCase):
    def test_finds_fill_enclosed_task(self):
        def box(n):
            # a wall of 3s around an n x n hole, on an empty 10 x 10 board
            g = np.zeros((10, 10), dtype=int)
            g[1, 1:n + 3] = g[n + 2, 1:n + 3] = 3
            g[1:n + 3, 1] = g[1:n + 3, n + 2] = 3
            return g
        def filled(n):
            g = box(n)
            g[2:n + 2, 2:n + 2] = 4
            return g
        task = {'train': [{'input': box(1).tolist(), 'output': filled(1).tolist()},
                          {'input': box(2).tolist(), 'output': filled(2).tolist()}],
                'test': [{'input': box(3).tolist(), 'output': filled(3).tolist()}]}
        lifts = {}
        found, tried = consistent_candidates(task, lifts)
        self.assertGreater(tried, len(found))
        best = choose(found, 'game')
        self.assertTrue(best and solves(best[0], task))
        self.assertTrue(best[0]['name'].startswith('fill_enclosed'))


if __name__ == '__main__':
    unittest.main()
