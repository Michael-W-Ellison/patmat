#!/usr/bin/env python3
"""Unit tests for arc_elements and arc_pattern_solver (run: python3 -m unittest test_arc_pattern_solver)"""

import unittest

import numpy as np

from arc_elements import segment
from arc_pattern_solver import find_patterns, rank

A = np.array


def task(pairs, test_in, test_out):
    return {'train': [{'input': a, 'output': b} for a, b in pairs],
            'test': [{'input': test_in, 'output': test_out}]}


def best(t):
    found, _ = find_patterns(t)
    ranked = rank(found, t, {})
    return ranked[0] if ranked else None


def right(entry, t):
    return entry is not None and all(np.array_equal(p, A(x['output'])) for p, x in zip(entry['preds'], t['test']))


class TestElements(unittest.TestCase):
    def test_segmentations_and_properties(self):
        a = A([[1, 1, 0, 2],
               [0, 1, 0, 0],
               [0, 0, 3, 0]])
        objs = segment(a, 0, 'color4')
        self.assertEqual(sorted(o.props['size'] for o in objs), [1, 1, 3])
        big = max(objs, key=lambda o: o.props['size'])
        self.assertTrue(big.props['is_largest'] and big.props['unique_shape'])
        self.assertEqual(big.props['color'], 1)
        self.assertEqual(len(segment(a, 0, 'multi8')), 2)    # the 3 touches the 1s diagonally

    def test_holes_and_symmetry(self):
        ring = A([[5, 5, 5], [5, 0, 5], [5, 5, 5]])
        o = segment(ring, 0, 'color4')[0]
        self.assertEqual(o.props['holes'], 1)
        self.assertTrue(o.props['sym_lr'] and o.props['sym_ud'])


class TestFamilies(unittest.TestCase):
    def test_recolor_largest_object(self):           # family A
        pairs = [([[1, 1, 0, 1], [1, 0, 0, 0]], [[2, 2, 0, 1], [2, 0, 0, 0]]),
                 ([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], [[0, 2, 0, 0], [0, 2, 2, 0], [1, 0, 0, 1]])]
        t = task(pairs, [[1, 0, 1], [0, 0, 1]], [[1, 0, 2], [0, 0, 2]])
        self.assertTrue(right(best(t), t))

    def test_crop_unique_colour_object(self):        # family B
        pairs = [([[1, 1, 0, 0], [1, 1, 0, 4], [0, 0, 0, 4]], [[4], [4]]),
                 ([[0, 6, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], [[6]])]
        t = task(pairs, [[1, 0, 0], [0, 0, 7], [0, 7, 7]], [[0, 7], [7, 7]])
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'B')

    def test_slide_down(self):                       # family D (or gravity)
        pairs = [([[3, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [3, 0]]),
                 ([[0, 5, 5], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 5, 5]])]
        t = task(pairs, [[0, 0, 2], [0, 0, 0]], [[0, 0, 0], [0, 0, 2]])
        self.assertTrue(right(best(t), t))

    def test_tile_with_reflection(self):             # family E
        def make(a):
            a = A(a)
            return np.hstack([a, np.fliplr(a)]).tolist()
        pairs = [([[1, 2], [3, 4]], make([[1, 2], [3, 4]])), ([[5, 0], [0, 6]], make([[5, 0], [0, 6]]))]
        t = task(pairs, [[7, 8], [0, 9]], make([[7, 8], [0, 9]]))
        e = best(t)
        self.assertTrue(right(e, t))

    def test_upscale(self):
        pairs = [([[1, 2]], [[1, 1, 2, 2], [1, 1, 2, 2]]), ([[3], [4]], [[3, 3], [3, 3], [4, 4], [4, 4]])]
        t = task(pairs, [[5, 6]], [[5, 5, 6, 6], [5, 5, 6, 6]])
        self.assertTrue(right(best(t), t))

    def test_no_pattern(self):
        pairs = [([[1, 2]], [[7, 7, 7]]), ([[3, 4]], [[0]])]
        t = task(pairs, [[5, 6]], [[1]])
        self.assertIsNone(best(t))


if __name__ == '__main__':
    unittest.main()
