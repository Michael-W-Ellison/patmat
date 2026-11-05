#!/usr/bin/env python3
"""Tests for sequence_puzzles and the transcribed children's worksheets (children_worksheets)"""

import unittest

import numpy as np

from sequence_puzzles import SequencePuzzle, predictions, solve


def P(x, shape=(2, 2)):
    return np.full(shape, x)


def seq(text):
    return [P(ord(c) - 64) for c in text]


def letters(answer):
    return ''.join(chr(int(p[0, 0]) + 64) for p in answer['panels'])


class TestRepeat(unittest.TestCase):
    def test_shortest_period_and_several_blanks(self):
        self.assertEqual(letters(solve(SequencePuzzle(seq('MMNNM'), 2))), 'MN')
        self.assertEqual(letters(solve(SequencePuzzle(seq('HHKHH'), 2))), 'KH')
        self.assertEqual(letters(solve(SequencePuzzle(seq('RLRL'), 5))), 'RLRLR')

    def test_a_first_panel_apart(self):
        # an engine (E) and cars alternating in two colours (B, C)
        answer = solve(SequencePuzzle(seq('EBCB'), 1))
        self.assertEqual(answer['family'], 'repeat')
        self.assertEqual(letters(answer), 'C')

    def test_a_cycle_not_yet_seen_twice(self):
        answer = solve(SequencePuzzle(seq('RRLL'), 5), use_slow=False)
        self.assertEqual(answer['family'], 'repeat_all')
        self.assertEqual(letters(answer), 'RRLLR')


class TestRuns(unittest.TestCase):
    def test_runs_that_grow(self):
        answer = solve(SequencePuzzle(seq('ABAABAAA'), 1))
        self.assertEqual(answer['family'], 'runs')
        self.assertEqual(letters(answer), 'B')
        self.assertEqual(letters(solve(SequencePuzzle(seq('ABAABAAAB'), 4))), 'AAAA')

    def test_equal_runs_are_left_to_repeat(self):
        self.assertEqual([p['family'] for p in predictions(SequencePuzzle(seq('AABBAAB'), 1))][0], 'repeat')


class TestGrowth(unittest.TestCase):
    def test_segments_added_at_one_end(self):
        def caterpillar(k):
            g = np.zeros((2, 2 + 2 * k), int)
            g[:, :2] = 5
            for i in range(k):
                g[:, 2 + 2 * i] = 3
            return g
        answer = solve(SequencePuzzle([caterpillar(1), caterpillar(3), caterpillar(5)], 2))
        self.assertEqual(answer['family'], 'growth')
        self.assertTrue(np.array_equal(answer['panels'][0], caterpillar(7)))
        self.assertTrue(np.array_equal(answer['panels'][1], caterpillar(9)))

    def test_two_parts_moving_apart(self):
        # an L of cells: each step adds one on top and one at the right
        def ell(n):
            g = np.zeros((n, n), int)
            g[:, 0] = 1
            g[n - 1, :] = 1
            return g
        answer = solve(SequencePuzzle([ell(1), ell(2), ell(3)]))
        self.assertTrue(np.array_equal(answer['panels'][0], ell(4)))


class TestChoices(unittest.TestCase):
    def test_the_answer_among_the_choices(self):
        answer = solve(SequencePuzzle(seq('ABAB'), 1, [P(2), P(1)]))
        self.assertEqual(answer['choice'], 1)


class TestWorksheets(unittest.TestCase):
    def test_the_sheets_seen_while_writing_the_reader(self):
        from children_worksheets import ITEMS, run
        summary = run(ITEMS, use_slow=False)
        wrong = [(i.sheet, i.number) for i, o, _ in summary['results'] if o != 'right']
        self.assertEqual(wrong, [])

    def test_the_held_out_sheets(self):
        # every held-out item is answered; only the number sequences (arithmetic) are not read
        from children_worksheets import HELD_OUT, run
        summary = run(HELD_OUT, use_slow=False)
        wrong = {i.kind for i, o, _ in summary['results'] if o != 'right'}
        self.assertEqual(wrong, {'number'})
        self.assertEqual(summary['unanswered'], 0)


if __name__ == '__main__':
    unittest.main()
