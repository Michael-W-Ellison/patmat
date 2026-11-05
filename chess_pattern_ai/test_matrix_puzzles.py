#!/usr/bin/env python3
"""Tests for matrix_puzzles and the colour-keeping boolean patterns of arc_pattern_solver"""

import unittest

import numpy as np

from matrix_puzzles import MatrixPuzzle, from_grid, predictions, solve

L = np.array([[1, 0], [1, 1]])          # an L shape (no turn of it equals another)


def panel(shape, color, top_left=(0, 0), size=6):
    out = np.zeros((size, size), int)
    s = np.asarray(shape)
    out[top_left[0]:top_left[0] + s.shape[0], top_left[1]:top_left[1] + s.shape[1]] = s * color
    return out


def puzzle(rows, choices=()):
    rows = [list(r) for r in rows]
    truth = rows[2][2]
    rows[2][2] = None
    return MatrixPuzzle(rows, list(choices)), truth


class TestAttributes(unittest.TestCase):
    def test_rotation_colour_and_scale_at_once(self):
        # along each row the L turns by 90 degrees, the colour goes 2 -> 5 -> 7, and the scale 1x -> 2x -> 3x
        bases = [L, np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 0], [1, 0], [1, 1]])]   # one per row
        rows = [[panel(np.kron(np.rot90(base, j), np.ones((j + 1, j + 1), int)), (2, 5, 7)[j], size=9)
                 for j in range(3)] for base in bases]
        p, truth = puzzle(rows)
        answer = solve(p)
        self.assertEqual(answer['family'], 'attributes')
        self.assertTrue(np.array_equal(answer['panel'], truth))

    def test_a_scale_step_read_from_each_row(self):
        # rows grow by 1x per panel from different starting sizes: 1, 2, 3 / 2, 3, 4 ... (the step is the row's)
        rows = [[panel(np.kron(L, np.ones((s, s), int)), 3, size=10) for s in (r + 1, r + 2, r + 3)]
                for r in range(3)]
        p, truth = puzzle(rows)
        self.assertTrue(np.array_equal(solve(p)['panel'], truth))

    def test_movement(self):
        rows = [[panel(L, 4 + r, (1, j)) for j in range(3)] for r in range(3)]
        p, truth = puzzle(rows)
        self.assertTrue(np.array_equal(solve(p)['panel'], truth))

    def test_column_rule_and_missing_first_panel(self):
        # down each column the L turns; the missing panel is the top-left one
        rows = [[panel(np.rot90(L, r), c) for c in (1, 2, 3)] for r in range(3)]
        truth = rows[0][0]
        rows[0][0] = None
        answer = solve(MatrixPuzzle(rows))
        self.assertTrue(np.array_equal(answer['panel'], truth))


class TestDistribution(unittest.TestCase):
    def test_the_panel_the_last_row_lacks(self):
        a, b, c = panel(L, 1), panel(L, 2), panel(L.T, 3)
        p, truth = puzzle([[a, b, c], [b, c, a], [c, a, b]])
        answer = solve(p)
        self.assertEqual(answer['family'], 'distribution')
        self.assertTrue(np.array_equal(answer['panel'], truth))


class TestCombination(unittest.TestCase):
    def test_xor_in_each_rows_colour_then_a_turn(self):
        rng = np.random.default_rng(0)
        rows = []
        for color in (2, 6, 8):
            a = (rng.random((5, 5)) < 0.5) * color
            b = (rng.random((5, 5)) < 0.5) * color
            c = np.rot90(np.where((a > 0) ^ (b > 0), color, 0))
            rows.append([a, b, c])
        p, truth = puzzle(rows)
        answer = solve(p)
        self.assertEqual(answer['family'], 'combination')
        self.assertTrue(np.array_equal(answer['panel'], truth))


class TestChoicesAndReading(unittest.TestCase):
    def test_answer_among_choices_and_from_one_grid(self):
        rows = [[panel(np.rot90(L, j), 5) for j in range(3)] for _ in range(3)]
        truth = rows[2][2]
        choices = [np.fliplr(truth), truth, panel(L, 5)]
        grid = np.full((3 * 6 + 2, 3 * 6 + 2), 9)
        for r in range(3):
            for c in range(3):
                grid[r * 7:r * 7 + 6, c * 7:c * 7 + 6] = rows[r][c] if (r, c) != (2, 2) else 8
        p = from_grid(grid, separator=9, missing_value=8, choices=choices)
        self.assertEqual(p.missing, (2, 2))
        self.assertEqual(solve(p)['choice'], 1)

    def test_predictions_without_a_rule_are_not_forced(self):
        rng = np.random.default_rng(1)
        rows = [[(rng.random((5, 5)) < 0.5) * int(rng.integers(1, 9)) for _ in range(3)] for _ in range(3)]
        p, truth = puzzle(rows)
        self.assertEqual(predictions(p, use_slow=False), [])


class TestArcOwnColours(unittest.TestCase):
    def test_xor_keeps_the_colours_of_the_parts(self):
        from arc_pattern_solver import find_patterns
        rng = np.random.default_rng(2)
        task = {'train': [], 'test': []}
        for i, color in enumerate((2, 4, 7)):
            a = (rng.random((4, 4)) < 0.5) * color
            b = (rng.random((4, 4)) < 0.5) * color
            pair = {'input': np.hstack([a, np.full((4, 1), 5), b]).tolist(),
                    'output': np.where((a > 0) ^ (b > 0), color, 0).tolist()}
            task['train' if i < 2 else 'test'].append(pair)
        consistent, _ = find_patterns(task)
        truth = np.array(task['test'][0]['output'])
        self.assertTrue(any(np.array_equal(np.array(e['preds'][0]), truth) for e in consistent))


if __name__ == '__main__':
    unittest.main()
