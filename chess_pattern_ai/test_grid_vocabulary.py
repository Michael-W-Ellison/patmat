#!/usr/bin/env python3
"""Unit tests for grid_vocabulary (run: python3 -m unittest test_grid_vocabulary)"""

import unittest

from grid_vocabulary import (Grid, adjacent_cells, between, bracketed, components,
                             counts, describe, enclosed, lines, longest_line,
                             symmetries, touches)


class TestComponents(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([[1, 1, 0, 2],
                          [0, 1, 0, 2],
                          [0, 0, 1, 0]])

    def test_equal_value_regions_orthogonal(self):
        ones = sorted((r.size for r in components(self.grid, values=[1])), reverse=True)
        self.assertEqual(ones, [3, 1])          # (2,2) touches only diagonally

    def test_diagonal_connectivity(self):
        ones = components(self.grid, values=[1], diagonal=True)
        self.assertEqual([r.size for r in ones], [4])

    def test_multicolor_objects(self):
        objects = components(Grid([[1, 2, 0], [0, 3, 0], [0, 0, 4]]), multicolor=True)
        self.assertEqual(sorted(r.size for r in objects), [1, 3])

    def test_bbox(self):
        region = components(Grid([[0, 5, 5], [0, 5, 0]]), values=[5])[0]
        self.assertEqual(region.bbox, (0, 1, 1, 2))

    def test_liberties(self):
        group = components(self.grid, values=[2])[0]
        self.assertEqual(adjacent_cells(self.grid, group.cells, value=0), {(0, 2), (1, 2), (2, 3)})

    def test_touches(self):
        group = components(self.grid, values=[2])[0]
        self.assertTrue(touches(self.grid, group.cells, 'right'))
        self.assertFalse(touches(self.grid, group.cells, 'left'))


class TestLines(unittest.TestCase):
    def test_runs_in_all_directions(self):
        g = Grid([[1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
        self.assertEqual(longest_line(g, 1), 4)            # main diagonal
        horizontal = [ln for ln in lines(g, 1) if ln.axis == ('W', 'E')]
        self.assertEqual([ln.length for ln in horizontal], [3])
        self.assertEqual(horizontal[0].open_ends, 1)

    def test_absent_value(self):
        self.assertEqual(longest_line(Grid([[0, 0]]), 7), 0)

    def test_between(self):
        g = Grid([[0] * 5 for _ in range(5)])
        self.assertEqual(between(g, (0, 0), (0, 4)), [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(between(g, (0, 0), (3, 3)), [(1, 1), (2, 2)])
        self.assertIsNone(between(g, (0, 0), (1, 2)))

    def test_bracketed_like_othello(self):
        g = Grid([[0, 0, 0, 0, 0],
                  [0, 2, 2, 1, 0],
                  [0, 2, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
        self.assertEqual(sorted(bracketed(g, (1, 0), own=1, other=2)), [(1, 1), (1, 2)])
        self.assertEqual(sorted(bracketed(g, (0, 1), own=1, other=2)), [(1, 1), (2, 1)])
        self.assertEqual(bracketed(g, (4, 4), own=1, other=2), [])


class TestEnclosed(unittest.TestCase):
    def test_enclosed_region(self):
        g = Grid([[0, 0, 0, 0, 0],
                  [0, 3, 3, 3, 0],
                  [0, 3, 0, 3, 0],
                  [0, 3, 3, 3, 0],
                  [0, 0, 0, 0, 0]])
        regions = enclosed(g, walls=[3])
        self.assertEqual([sorted(r.cells) for r in regions], [[(2, 2)]])

    def test_leak_through_diagonal(self):
        g = Grid([[0, 3, 0],
                  [3, 0, 3],
                  [0, 3, 0]])
        self.assertEqual(len(enclosed(g, walls=[3])), 1)
        self.assertEqual(len(enclosed(g, walls=[3], diagonal=True)), 0)


class TestSymmetry(unittest.TestCase):
    def test_grid_symmetries(self):
        s = symmetries(Grid([[1, 2, 1], [3, 4, 3], [1, 2, 1]]))
        self.assertTrue(s['flip_lr'] and s['flip_ud'] and s['rot180'])
        self.assertFalse(s['transpose'])

    def test_region_symmetry(self):
        g = Grid([[0, 5, 0, 0],
                  [5, 5, 5, 0],
                  [0, 5, 0, 7]])
        plus = components(g, values=[5])[0]
        self.assertTrue(all(symmetries(g, plus).values()))


class TestHexAndGraph(unittest.TestCase):
    def test_hex_neighbours_follow_row_parity(self):
        g = Grid([[0] * 3 for _ in range(3)], topology='hex')
        self.assertEqual(sorted(g.neighbors((1, 1))), [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)])
        self.assertEqual(sorted(g.neighbors((2, 1))), [(1, 0), (1, 1), (2, 0), (2, 2)])

    def test_hex_chain_component(self):
        # (0,0) -> (1,0) -> (2,0): connected through hex SE/SW steps
        g = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]], topology='hex')
        self.assertEqual([r.size for r in components(g, values=[1])], [3])

    def test_graph_lines(self):
        cells = [[1, 1, 1], [0, 2, 0]]
        adjacency = {(0, 0): [(0, 1)], (0, 1): [(0, 0), (0, 2), (1, 1)], (0, 2): [(0, 1)], (1, 1): [(0, 1)]}
        valid = [[True, True, True], [False, True, False]]
        g = Grid(cells, topology='graph', valid=valid, adjacency=adjacency,
                 line_sets=[[(0, 0), (0, 1), (0, 2)]])
        self.assertEqual([ln.length for ln in lines(g, 1, min_length=3)], [3])
        self.assertEqual(between(g, (0, 0), (0, 2)), [(0, 1)])


class TestSummary(unittest.TestCase):
    def test_describe(self):
        g = Grid([[1, 1, 0], [0, 0, 0], [0, 0, 2]])
        d = describe(g)
        self.assertEqual(d['values'][1], {'cells': 2, 'regions': 1, 'largest_region': 2, 'longest_line': 2})
        self.assertEqual(d['objects'], 2)
        self.assertEqual(counts(g)[0], 6)


if __name__ == '__main__':
    unittest.main()
