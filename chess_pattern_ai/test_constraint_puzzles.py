#!/usr/bin/env python3
"""Unit tests for constraint_puzzles (run: python3 -m unittest test_constraint_puzzles)"""

import unittest

from constraint_puzzles import EMPTY, mosaic, nonogram, orthogonal, placement, totals
from minesweeper.minesweeper_game import HIDDEN, MINE


class TestPuzzles(unittest.TestCase):
    def test_clues_match_the_solution(self):
        p = orthogonal(1)
        for (r, c), v in p.clues.items():
            self.assertEqual(v, sum((r + dr, c + dc) in p.filled for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))))
        p = mosaic(1)
        for (r, c), v in p.clues.items():
            self.assertEqual(v, sum((r + dr, c + dc) in p.filled for dr in (-1, 0, 1) for dc in (-1, 0, 1)))
        p = totals(1)
        for i in range(1, 7):
            self.assertEqual(p.clues[(i, 0)], sum((i, c) in p.filled for c in range(1, 7)))
            self.assertEqual(p.clues[(0, i)], sum((r, i) in p.filled for r in range(1, 7)))

    def test_nonogram_runs(self):
        p = nonogram(2, n=6)
        m = 3
        for i in range(6):
            row = [(i + m, c + m) in p.filled for c in range(6)]
            runs, n = [], 0
            for v in row + [False]:
                if v:
                    n += 1
                elif n:
                    runs.append(n)
                    n = 0
            shown = [p.clues[(i + m, k)] for k in range(m) if (i + m, k) in p.clues]
            self.assertEqual(shown, runs or [0])

    def test_placement_one_per_row_column_box(self):
        p = placement(3)
        self.assertEqual(len(p.filled), 9)
        self.assertEqual({r for r, _ in p.filled}, set(range(9)))
        self.assertEqual({c for _, c in p.filled}, set(range(9)))
        self.assertEqual({(r // 3, c // 3) for r, c in p.filled}, {(i, j) for i in range(3) for j in range(3)})

    def test_play_protocol(self):
        p = totals(4)
        view = p.observation()
        self.assertEqual(view[0][0], EMPTY)
        self.assertTrue(all(view[r][c] == HIDDEN for r, c in p.playable))
        with self.assertRaises(ValueError):
            p.final_view()
        empty = sorted(p.playable - p.filled)
        for pos in empty:
            p.reveal(pos)
        self.assertTrue(p.won())
        self.assertTrue(all(p.final_view()[r][c] == MINE for r, c in p.filled))


if __name__ == '__main__':
    unittest.main()
