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


class TestLetters(unittest.TestCase):
    def test_missing_letters_filled_in_from_the_alphabet(self):
        from symbol_learning import SymbolMemory
        m = SymbolMemory()
        shapes = {}
        for i, ch in enumerate('ABCDEF'):
            g = np.zeros((7, 5), int)
            g[i % 7, :] = 1
            g[:, i % 5] = 1
            g[6 - i % 7, (i + 2) % 5] = 1
            shapes[ch] = np.kron(g, np.ones((4, 4), int))
            m.learn(shapes[ch], ch)
        answer = solve(SequencePuzzle([shapes['A'], None, shapes['C'], None, shapes['E']], 1), symbols=m)
        self.assertEqual(answer['family'], 'letters')
        self.assertEqual(answer['values'], ['B', 'D', 'F'])


class TestReadingInContext(unittest.TestCase):
    def test_a_doubtful_symbol_read_as_its_neighbours_require(self):
        from sequence_puzzles import _in_context
        # panel 2 reads likelier as 7 than as 3, but only 3 makes 1 2 3 4 5 go by a fixed step
        options = [[(1, 0.05)], [(2, 0.06)], [(7, 0.10), (3, 0.12)], [(4, 0.05)], [(5, 0.07)]]
        places, cost = _in_context(options)
        self.assertEqual(places, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(cost, 0.35)

    def test_a_row_of_second_choices_is_not_answered(self):
        from sequence_puzzles import _in_context
        options = [[(9, 0.05), (1, 0.06)], [(8, 0.05), (2, 0.06)], [(6, 0.05), (3, 0.07)], [(5, 0.05), (4, 0.08)]]
        self.assertIsNone(_in_context(options))


class TestWorksheets(unittest.TestCase):
    def test_the_sheets_seen_while_writing_the_reader(self):
        from children_worksheets import ITEMS, run
        summary = run(ITEMS, use_slow=False)
        wrong = [(i.sheet, i.number) for i, o, _ in summary['results'] if o != 'right']
        self.assertEqual(wrong, [])

    def test_the_held_out_sheets(self):
        # with the digits learned from counting worksheets (digit_lessons), the number sequences are read too
        from children_worksheets import HELD_OUT, run
        summary = run(HELD_OUT, use_slow=False)
        self.assertEqual([(i.sheet, i.number) for i, o, _ in summary['results'] if o != 'right'], [])

    def test_the_missing_letters_sheets(self):
        from children_worksheets import LETTER_ITEMS, run
        summary = run(LETTER_ITEMS, use_slow=False)
        self.assertEqual(summary['wrong'], 0)
        self.assertGreaterEqual(summary['right'], 23)

    def test_the_shape_sheets(self):
        from children_worksheets import SHAPE_ITEMS, _SHAPE_ROWS, run, run_identify
        self.assertEqual(run_identify()['right'], 7)
        summary = run(SHAPE_ITEMS, use_slow=False)
        self.assertGreaterEqual(summary['right'], 23)
        # every row answered wrongly holds a shape no lesson taught
        for item, outcome, _ in summary['results']:
            if outcome != 'right':
                row = _SHAPE_ROWS[int(item.sheet[-1]) - 1][item.number - 1].split()
                self.assertTrue({'cross', 'pentagon'} & set(row), (item.sheet, item.number))

    def test_without_learned_symbols_the_number_sequences_are_not_read(self):
        from children_worksheets import HELD_OUT, run
        summary = run([i for i in HELD_OUT if i.kind == 'number'], symbols=None, use_slow=False)
        self.assertEqual(summary['right'], 0)


if __name__ == '__main__':
    unittest.main()
