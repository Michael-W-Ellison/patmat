#!/usr/bin/env python3
"""Unit tests for arc_elements and arc_pattern_solver (run: python3 -m unittest test_arc_pattern_solver)"""

import unittest

import numpy as np

from arc_elements import segment
from arc_pattern_solver import exception_patterns, find_patterns, multi_step_patterns, rank

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


class TestNewCapabilities(unittest.TestCase):
    def test_two_property_condition(self):
        # delete objects that are colour 1 AND touch the border; keep everything else
        pairs = [([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]),
                 ([[0, 0, 0, 2], [0, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 0, 2], [0, 1, 0, 0], [0, 0, 0, 0]])]
        t = task(pairs, [[0, 0, 0, 1], [0, 1, 0, 0], [2, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0]])
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertIn('+', e['name'])

    def test_adopt_touching_colour(self):
        pairs = [([[5, 5, 3], [0, 0, 0]], [[3, 3, 3], [0, 0, 0]]),
                 ([[0, 0, 0], [4, 5, 5]], [[0, 0, 0], [4, 4, 4]])]
        t = task(pairs, [[5, 0, 0], [5, 6, 0]], [[6, 0, 0], [6, 6, 0]])
        self.assertTrue(right(best(t), t))

    def test_filter_then_select_crop(self):
        # the largest of the colour-2 objects (a larger colour-1 object is a distractor)
        g1 = [[1, 1, 1, 0, 2, 2], [1, 1, 1, 0, 2, 2], [0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]
        g2 = [[2, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        test = [[0, 0, 0, 1, 1, 1], [2, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [2, 2, 0, 0, 0, 0]]
        t = task([(g1, [[2, 2], [2, 2]]), (g2, [[2, 2, 2]])], test, [[2, 2]])
        e = best(t)
        self.assertTrue(right(e, t))

    def test_approach_until_touching(self):
        pairs = [([[3, 0, 0, 0, 5]], [[0, 0, 0, 3, 5]]),
                 ([[5, 0, 0, 3], [0, 0, 0, 0]], [[5, 3, 0, 0], [0, 0, 0, 0]])]
        t = task(pairs, [[0, 0], [3, 0], [0, 0], [5, 0]], [[0, 0], [0, 0], [3, 0], [5, 0]])
        self.assertTrue(right(best(t), t))

    def test_stamp_template_at_markers(self):
        def board(markers, template_at=(0, 0)):
            g = np.zeros((7, 7), dtype=int)
            r, c = template_at
            g[r:r + 2, c:c + 2] = [[1, 2], [2, 2]]
            for mr, mc in markers:
                g[mr, mc] = 1
            return g
        def stamped(markers, template_at=(0, 0)):
            g = board(markers, template_at)
            for mr, mc in markers:
                g[mr:mr + 2, mc:mc + 2] = [[1, 2], [2, 2]]
            return g
        pairs = [(board([(4, 4)]).tolist(), stamped([(4, 4)]).tolist()),
                 (board([(3, 1), (5, 5)]).tolist(), stamped([(3, 1), (5, 5)]).tolist())]
        t = task(pairs, board([(4, 0), (2, 4)]).tolist(), stamped([(4, 0), (2, 4)]).tolist())
        self.assertTrue(right(best(t), t))

    def test_combination_crop_then_recolor(self):
        # crop to the largest object, then recolour 1 -> 4
        g1 = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [3, 0, 0, 0]]
        g2 = [[1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 3]]
        test = [[0, 0, 0], [1, 0, 3], [1, 0, 0]]
        t = task([(g1, [[4, 4], [4, 4]]), (g2, [[4, 4, 4]])], test, [[4], [4]])
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'G')


class TestCountingAndBuilding(unittest.TestCase):
    def test_count_objects_as_row(self):             # family H
        pairs = [([[1, 0, 1], [0, 0, 0], [1, 0, 0]], [[3, 3, 3]]),
                 ([[0, 2, 0], [0, 0, 0], [2, 0, 0]], [[3, 3]])]
        t = task(pairs, [[4, 0, 4, 0], [0, 0, 0, 0], [4, 0, 4, 0]], [[3, 3, 3, 3]])
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'H')

    def test_colour_with_most_cells(self):           # family H, comparing
        pairs = [([[1, 1, 2], [0, 1, 0]], [[1]]), ([[3, 2, 2], [2, 0, 3]], [[2]]),
                 ([[4, 0, 5], [5, 0, 5]], [[5]])]
        t = task(pairs, [[6, 7, 7], [6, 6, 0]], [[6]])
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'H')

    def test_split_and_combine(self):                # family I
        def make(l, r):
            return [x + [5] + y for x, y in zip(l, r)]
        def both(l, r):
            return [[2 if x and y else 0 for x, y in zip(a, b)] for a, b in zip(l, r)]
        L1, R1 = [[1, 0], [1, 1]], [[1, 1], [0, 1]]
        L2, R2 = [[0, 1], [1, 0]], [[1, 1], [1, 1]]
        L3, R3 = [[1, 1], [0, 0]], [[1, 0], [1, 0]]
        t = task([(make(L1, R1), both(L1, R1)), (make(L2, R2), both(L2, R2))], make(L3, R3), both(L3, R3))
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'I')

    def test_local_rule(self):                       # family I
        # cells with exactly one non-background neighbour (line ends) turn 4
        g1 = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
        o1 = [[0, 0, 0, 0, 0], [0, 4, 1, 4, 0], [0, 0, 0, 0, 0]]
        g2 = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]
        o2 = [[0, 4, 0], [0, 1, 0], [0, 4, 0], [0, 0, 0]]
        test = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
        t = task([(g1, o1), (g2, o2)], test, [[0, 0, 0, 0], [4, 1, 1, 4], [0, 0, 0, 0]])
        found, _ = find_patterns(t)
        self.assertTrue(any(f['family'] == 'I' and right(f, t) for f in found))


class TestRoles(unittest.TestCase):
    def test_panel_fill_by_role(self):               # family J, panels
        def grid(n):
            # n x n panels of 2 x 2 cells separated by lines of 5
            size = 3 * n - 1
            g = np.zeros((size, size), dtype=int)
            for k in range(1, n):
                g[3 * k - 1, :] = 5
                g[:, 3 * k - 1] = 5
            return g
        def filled(n):
            g = grid(n)
            g[0:2, 0:2] = 1
            c = 3 * (n // 2)
            g[c:c + 2, c:c + 2] = 2
            g[-2:, -2:] = 3
            return g
        t = task([(grid(3).tolist(), filled(3).tolist()), (grid(5).tolist(), filled(5).tolist())],
                 grid(7).tolist(), filled(7).tolist())
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'J')

    def test_panel_with_unique_colour(self):         # family J, panels
        g1 = [[8, 8, 3, 8, 8], [4, 8, 3, 8, 8], [3, 3, 3, 3, 3], [8, 8, 3, 8, 8], [8, 8, 3, 8, 8]]
        g2 = [[1, 1, 2, 1, 1], [1, 1, 2, 1, 1], [2, 2, 2, 2, 2], [1, 1, 2, 1, 6], [1, 1, 2, 1, 1]]
        test = [[7, 7, 7, 0, 7, 7, 7], [7, 7, 7, 0, 7, 9, 7], [7, 7, 7, 0, 7, 7, 7], [0, 0, 0, 0, 0, 0, 0],
                [7, 7, 7, 0, 7, 7, 7], [7, 7, 7, 0, 7, 7, 7], [7, 7, 7, 0, 7, 7, 7]]
        t = task([(g1, [[8, 8], [4, 8]]), (g2, [[1, 6], [1, 1]])], test, [[7, 7, 7], [7, 9, 7], [7, 7, 7]])
        e = best(t)
        self.assertTrue(right(e, t))

    def test_block_summary(self):                    # family J, uniform blocks
        b1 = np.vstack([np.hstack([np.full((2, 3), 2), np.full((2, 1), 8)]),
                        np.hstack([np.full((3, 3), 3), np.full((3, 1), 5)])])
        b2 = np.hstack([np.full((4, 2), 4), np.full((4, 3), 1), np.full((4, 1), 6)])
        b3 = np.vstack([np.full((1, 5), 7), np.full((2, 5), 9), np.full((3, 5), 7)])
        t = task([(b1.tolist(), [[2, 8], [3, 5]]), (b2.tolist(), [[4, 1, 6]])], b3.tolist(), [[7], [9], [7]])
        self.assertTrue(right(best(t), t))

    def test_recolour_by_key_shape(self):            # family J, key / legend
        # grey (5) shapes take the colour of the key shape with the same shape
        g1 = [[1, 1, 0, 2, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5, 5],
              [0, 5, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0]]
        o1 = [[1, 1, 0, 2, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1],
              [0, 2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0]]
        g2 = [[3, 0, 4, 4, 4, 0, 0], [3, 0, 0, 0, 0, 0, 0], [3, 0, 0, 5, 5, 5, 0],
              [0, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0]]
        o2 = [[3, 0, 4, 4, 4, 0, 0], [3, 0, 0, 0, 0, 0, 0], [3, 0, 0, 4, 4, 4, 0],
              [0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0]]
        test = [[6, 0, 7, 7, 0, 0], [6, 0, 7, 7, 0, 0], [0, 0, 0, 0, 0, 5], [5, 5, 0, 0, 0, 5], [5, 5, 0, 0, 0, 0]]
        out = [[6, 0, 7, 7, 0, 0], [6, 0, 7, 7, 0, 0], [0, 0, 0, 0, 0, 6], [7, 7, 0, 0, 0, 6], [7, 7, 0, 0, 0, 0]]
        t = task([(g1, o1), (g2, o2)], test, out)
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'J')

    def test_recolour_by_marker_in_column(self):     # family J, markers
        g1 = [[2, 0, 0, 6, 0], [0, 0, 0, 0, 0], [5, 5, 0, 5, 5], [5, 5, 0, 5, 5]]
        o1 = [[2, 0, 0, 6, 0], [0, 0, 0, 0, 0], [2, 2, 0, 6, 6], [2, 2, 0, 6, 6]]
        g2 = [[0, 1, 0, 0, 8], [0, 0, 0, 0, 0], [0, 5, 0, 5, 5], [0, 5, 0, 5, 5], [0, 0, 0, 0, 0]]
        o2 = [[0, 1, 0, 0, 8], [0, 0, 0, 0, 0], [0, 1, 0, 8, 8], [0, 1, 0, 8, 8], [0, 0, 0, 0, 0]]
        test = [[0, 0, 4, 0, 0, 3], [0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 5]]
        out = [[0, 0, 4, 0, 0, 3], [0, 0, 0, 0, 0, 0], [0, 4, 4, 0, 0, 0], [0, 0, 0, 0, 3, 3]]
        t = task([(g1, o1), (g2, o2)], test, out)
        self.assertTrue(right(best(t), t))

    def test_colour_map_from_key(self):              # family J, legend
        g1 = [[1, 2, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]]
        o1 = [[1, 2, 0, 0], [0, 0, 2, 2], [0, 0, 2, 0]]
        g2 = [[3, 0, 3, 0], [0, 0, 0, 0], [3, 4, 0, 3]]
        o2 = [[4, 0, 4, 0], [0, 0, 0, 0], [3, 4, 0, 4]]
        test = [[6, 6, 0, 0], [0, 0, 0, 0], [0, 6, 7, 0]]
        t = task([(g1, o1), (g2, o2)], test, [[7, 7, 0, 0], [0, 0, 0, 0], [0, 6, 7, 0]])
        self.assertTrue(right(best(t), t))

    def test_region_bounded_by_markers(self):        # family J, markers
        g1 = [[0, 0, 0, 0, 0], [0, 4, 0, 4, 0], [0, 0, 3, 0, 0], [0, 4, 0, 4, 0]]
        g2 = [[4, 0, 0, 4], [0, 1, 2, 0], [0, 2, 1, 0], [4, 0, 0, 4], [0, 0, 0, 0]]
        test = [[0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 4, 0], [0, 0, 6, 6, 0, 0], [0, 0, 0, 7, 0, 0], [0, 4, 0, 0, 4, 0]]
        t = task([(g1, [[3]]), (g2, [[1, 2], [2, 1]])],
                 test, [[6, 6], [0, 7]])
        self.assertTrue(right(best(t), t))


class TestRays(unittest.TestCase):
    def test_ray_stops_at_wall(self):                # family K
        pairs = [([[2, 0, 0, 0, 5], [0, 0, 0, 0, 5]], [[2, 2, 2, 2, 5], [0, 0, 0, 0, 5]]),
                 ([[0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 5, 0], [0, 0, 0, 0, 0, 0]],
                  [[0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 5, 0], [0, 0, 0, 0, 0, 0]])]
        t = task(pairs, [[0, 2, 0, 5, 0], [2, 0, 0, 0, 0]], [[0, 2, 2, 5, 0], [2, 2, 2, 2, 2]])
        e = best(t)
        self.assertTrue(right(e, t))

    def test_ray_turns_at_obstacle(self):            # family K
        # rays go up and turn right (east) when blocked
        g1 = [[0, 0, 0, 0], [5, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0]]
        o1 = [[0, 0, 0, 0], [5, 0, 0, 0], [3, 3, 3, 3], [3, 0, 0, 0]]
        g2 = [[0, 5, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 3, 0, 0, 0]]
        o2 = [[0, 5, 0, 0, 0], [0, 3, 3, 3, 3], [0, 3, 0, 0, 0], [0, 3, 0, 0, 0]]
        test = [[0, 0, 0], [0, 0, 5], [0, 0, 0], [0, 0, 3], [0, 0, 0]]
        out = [[0, 0, 0], [0, 0, 5], [0, 0, 3], [0, 0, 3], [0, 0, 0]]
        t = task([(g1, o1), (g2, o2)], test, out)
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'K')

    def test_crossing_lines(self):                   # family K
        def make(points, size):
            g = np.zeros(size, dtype=int)
            o = np.zeros(size, dtype=int)
            for (r, c), v in points:
                g[r, c] = v
            for (r, c), v in points:
                o[r, :] = np.where(o[r, :] == 0, v, 2)
                o[:, c] = np.where(o[:, c] == 0, v, 2)
            for (r, c), v in points:
                o[r, c] = v
            return g.tolist(), o.tolist()
        p1 = make([((1, 1), 8), ((4, 3), 5)], (6, 6))
        p2 = make([((0, 4), 4), ((3, 1), 3)], (5, 6))
        test = make([((2, 2), 6), ((5, 0), 7)], (7, 5))
        t = task([p1, p2], *test)
        self.assertTrue(right(best(t), t))

    def test_diagonal_bounce(self):                  # family K
        def make(r, c, h, w):
            g = np.zeros((h, w), dtype=int)
            g[r, c] = 4
            o = g.copy()
            dr, dc = -1, 1
            for _ in range(2 * (h + w)):
                nr, nc = r + dr, c + dc
                if not 0 <= nr < h:
                    dr = -dr
                    continue
                if not 0 <= nc < w:
                    break
                r, c = nr, nc
                o[r, c] = 4
            return g.tolist(), o.tolist()
        t = task([make(4, 0, 5, 7), make(2, 0, 3, 6)], *make(3, 0, 4, 9))
        e = best(t)
        self.assertTrue(right(e, t))


class TestParametersAndSteps(unittest.TestCase):
    def test_move_up_by_height(self):                # family M
        def make(objs, h, w):
            g, o = np.zeros((h, w), dtype=int), np.zeros((h, w), dtype=int)
            for color, c, height in objs:
                g[h - height:h, c] = color
                o[h - 2 * height:h - height, c] = color
            return g.tolist(), o.tolist()
        t = task([make([(1, 0, 2), (4, 2, 3)], 7, 4), make([(2, 1, 1), (3, 3, 2)], 5, 5)],
                 *make([(5, 0, 3), (6, 2, 1), (7, 4, 2)], 8, 5))
        e = best(t)
        self.assertTrue(right(e, t))

    def test_scale_by_number_of_colours(self):       # family M
        def up(a, k):
            return np.kron(np.array(a), np.ones((k, k), dtype=int)).tolist()
        g1, g2, g3 = [[1, 0], [0, 2]], [[3, 3], [0, 0]], [[4, 5], [6, 0]]
        t = task([(g1, up(g1, 2)), (g2, up(g2, 1))], g3, up(g3, 3))
        e = best(t)
        self.assertTrue(right(e, t))
        self.assertEqual(e['family'], 'M')

    def test_chain_tile_then_rays(self):             # family L, size step first
        def make(g):
            t = np.tile(np.array(g), (2, 2))
            o = t.copy()
            for c in range(t.shape[1]):
                for r in range(t.shape[0]):
                    if t[r, c] == 2:
                        rr = r + 1
                        while rr < t.shape[0] and t[rr, c] == 0:
                            o[rr, c] = 8
                            rr += 1
            return g, o.tolist()
        t = task([make([[2, 0, 0], [0, 0, 0], [0, 0, 2]]), make([[0, 2], [0, 0], [0, 0]])],
                 *make([[0, 0, 0, 2], [2, 0, 0, 0], [0, 0, 0, 0]]))
        e = best(t)
        self.assertTrue(right(e, t))

    def test_chain_from_subgoal(self):               # family L, part of the change first
        import time
        # outline objects in 3 (halo), then fill their holes with 4
        def make(boxes, h, w):
            g = np.zeros((h, w), dtype=int)
            for r, c in boxes:
                g[r:r + 3, c:c + 3] = 6
                g[r + 1, c + 1] = 0
            o = g.copy()
            for r, c in boxes:
                o[r + 1, c + 1] = 4
                for rr in range(r - 1, r + 4):
                    for cc in range(c - 1, c + 4):
                        if 0 <= rr < h and 0 <= cc < w and o[rr, cc] == 0:
                            o[rr, cc] = 3
            return g, o
        pairs = [make([(1, 1)], 6, 6), make([(1, 1), (1, 6)], 6, 11)]
        chains = multi_step_patterns(pairs, 0, time.time() + 60)
        self.assertTrue(chains)
        test_in, test_out = make([(2, 2), (6, 1)], 10, 7)
        self.assertTrue(any(np.array_equal(c.predict(test_in), test_out) for c in chains))


class TestSymmetryAndRelativePositions(unittest.TestCase):
    def test_complete_symmetry_about_object_centre(self):   # family N
        # a shape mirrored left-right about a vertical marker line of 5
        def make(left, col, h, w):
            g = np.zeros((h, w), dtype=int)
            g[:, col] = 5
            for r, c, v in left:
                g[r, c] = v
            o = g.copy()
            for r, c, v in left:
                o[r, 2 * col - c] = v
            return g.tolist(), o.tolist()
        p1 = make([(0, 1, 2), (1, 2, 3)], 3, 3, 7)
        p2 = make([(1, 4, 4), (2, 3, 4), (0, 3, 1)], 5, 4, 9)
        test = make([(0, 0, 6), (2, 1, 7)], 2, 3, 8)
        t = task([p1, p2], *test)
        e = best(t)
        self.assertTrue(right(e, t))

    def test_repair_under_mask(self):                # family N
        def sym(seed):
            rng = np.random.RandomState(seed)
            q = rng.randint(1, 5, (4, 4))
            top = np.hstack([q, np.fliplr(q)])
            return np.vstack([top, np.flipud(top)])
        def masked(g, r, c, h, w):
            m = g.copy()
            m[r:r + h, c:c + w] = 9
            return m
        g1, g2, g3 = sym(1), sym(2), sym(3)
        t = task([(masked(g1, 1, 1, 2, 3).tolist(), g1.tolist()), (masked(g2, 4, 5, 3, 2).tolist(), g2.tolist())],
                 masked(g3, 5, 0, 2, 2).tolist(), g3.tolist())
        self.assertTrue(right(best(t), t))
        # the same, answering with the repaired patch only
        t = task([(masked(g1, 1, 1, 2, 3).tolist(), g1[1:3, 1:4].tolist()),
                  (masked(g2, 4, 5, 3, 2).tolist(), g2[4:7, 5:7].tolist())],
                 masked(g3, 5, 0, 2, 2).tolist(), g3[5:7, 0:2].tolist())
        self.assertTrue(right(best(t), t))

    def test_template_scaled_by_object_size(self):   # family O
        # each square gets a copy of itself in 4 at its right, one square-width away
        def make(squares, h, w):
            g = np.zeros((h, w), dtype=int)
            for r, c, k in squares:
                g[r:r + k, c:c + k] = 1
            o = g.copy()
            for r, c, k in squares:
                o[r:r + k, c + 2 * k:c + 3 * k] = 4
            return g.tolist(), o.tolist()
        t = task([make([(0, 0, 1), (3, 0, 2)], 6, 7), make([(1, 1, 3)], 5, 10)], *make([(0, 0, 2), (4, 1, 1)], 6, 8))
        e = best(t)
        self.assertTrue(right(e, t))

    def test_line_up_with_nearest(self):             # family O
        # the 3-cell moves along its column to the row of the 5-bar's centre
        g1 = [[0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 5]]
        o1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5], [0, 3, 0, 5], [0, 0, 0, 5]]
        g2 = [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0]]
        o2 = [[5, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        test = [[0, 0, 0, 3], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0]]
        out = [[0, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 3], [5, 0, 0, 0], [5, 0, 0, 0]]
        t = task([(g1, o1), (g2, o2)], test, out)
        self.assertTrue(right(best(t), t))


class TestPartialMatches(unittest.TestCase):
    def test_exception_condition(self):              # family X
        # rays go down from every object except the colour-5 ones (a near miss without the condition)
        def make(cols, h, w):
            g = np.zeros((h, w), dtype=int)
            for c, v in cols:
                g[0, c] = v
            o = g.copy()
            for c, v in cols:
                if v != 5:
                    o[1:, c] = v
            return g.tolist(), o.tolist()
        p1 = make([(0, 2), (2, 5), (4, 3)], 4, 5)
        p2 = make([(1, 5), (3, 1), (5, 5)], 3, 6)
        test = make([(0, 5), (2, 4), (3, 6)], 5, 4)
        import time
        found = exception_patterns([(A(a), A(b)) for a, b in (p1, p2)], 0, time.time() + 60)
        self.assertTrue(found)
        self.assertTrue(any(np.array_equal(f.predict(A(test[0])), A(test[1])) for f in found))

    def test_exception_on_majority_rule(self):       # family X on a family-A near miss
        # colour-1 objects become 2, except the 3-cell ones, which stay
        g1 = [[1, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 1]]
        o1 = [[2, 0, 1, 1, 1], [0, 0, 0, 0, 0], [2, 2, 0, 0, 2]]
        g2 = [[1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 1]]
        o2 = [[1, 0, 0, 2], [1, 0, 0, 0], [1, 0, 2, 2]]
        test = [[1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 1, 1]]
        out = [[1, 1, 1, 0], [0, 0, 0, 0], [2, 0, 2, 2]]
        import time
        found = exception_patterns([(A(g1), A(o1)), (A(g2), A(o2))], 0, time.time() + 60)
        self.assertTrue(any(f.name.startswith('A:') and np.array_equal(f.predict(A(test)), A(out)) for f in found))

    def test_chain_from_partial_step(self):          # family L, partial step first
        # mirror the shape left-right about the grid centre (a partial step: it reaches only
        # the mirrored cells), then outline everything in 3
        def make(cells, h, w):
            g = np.zeros((h, w), dtype=int)
            for r, c in cells:
                g[r, c] = 1
            m = np.where(g == 0, np.fliplr(g), g)
            o = m.copy()
            for r in range(h):
                for c in range(w):
                    if m[r, c] == 0 and any(0 <= r + dr < h and 0 <= c + dc < w and m[r + dr, c + dc] == 1
                                            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))):
                        o[r, c] = 3
            return g, o
        import time
        pairs = [make([(1, 1), (2, 1)], 4, 6), make([(2, 0), (2, 1), (3, 2)], 6, 7)]
        chains = multi_step_patterns(pairs, 0, time.time() + 60)
        test_in, test_out = make([(1, 2), (3, 1)], 5, 8)
        self.assertTrue(any(np.array_equal(c.predict(test_in), test_out) for c in chains))


if __name__ == '__main__':
    unittest.main()
