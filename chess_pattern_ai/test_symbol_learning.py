#!/usr/bin/env python3
"""Tests for symbol_learning (learning symbols from examples) and its use on the number worksheet"""

import os
import tempfile
import unittest

import numpy as np

from symbol_learning import SymbolMemory, counting_scene, pack, unpack

# two type styles of the digits 0-9, as 5 x 7 bitmaps scaled up
STYLE_A = {'0': '.###.|#...#|#..##|#.#.#|##..#|#...#|.###.', '1': '..#..|.##..|..#..|..#..|..#..|..#..|.###.',
           '2': '.###.|#...#|....#|...#.|..#..|.#...|#####', '3': '#####|...#.|..#..|...#.|....#|#...#|.###.',
           '4': '...#.|..##.|.#.#.|#..#.|#####|...#.|...#.', '5': '#####|#....|####.|....#|....#|#...#|.###.'}
STYLE_B = {'2': '####.|....#|....#|.###.|#....|#....|#####', '4': '#...#|#...#|#...#|#####|....#|....#|....#',
           '5': '#####|#....|#....|####.|....#|....#|####.'}


def glyph(rows, scale=6):
    a = np.array([[ch == '#' for ch in r] for r in rows.split('|')], dtype=int)
    return np.kron(a, np.ones((scale, scale), int))


def memory_of(style, digits):
    m = SymbolMemory()
    for d in digits:
        m.learn(glyph(style[d]), int(d), 'test')
    return m


class TestRecognition(unittest.TestCase):
    def test_learned_symbols_are_recognised_at_another_size(self):
        m = memory_of(STYLE_A, '012345')
        for d in '012345':
            self.assertEqual(m.recognise(glyph(STYLE_A[d], scale=3)), int(d))

    def test_unlike_shapes_are_unknown(self):
        m = memory_of(STYLE_A, '012345')
        blot = np.zeros((40, 40), int)
        blot[5:35, 5:35] = 1
        self.assertIsNone(m.recognise(blot))

    def test_reading_a_number(self):
        m = memory_of(STYLE_A, '012345')
        panel = np.hstack([glyph(STYLE_A['2']), np.zeros((42, 6), int), glyph(STYLE_A['4'])])
        self.assertEqual(m.read_number(panel), 24)
        self.assertEqual(m.read_number(m.draw_number(503)), 503)

    def test_saved_and_loaded(self):
        path = os.path.join(tempfile.mkdtemp(), 'symbols.json')
        memory_of(STYLE_A, '12').save(path)
        self.assertEqual(SymbolMemory(path).recognise(glyph(STYLE_A['2'])), 2)
        mask = (np.random.default_rng(0).random((7, 13)) > 0.5).astype(int)
        self.assertTrue(np.array_equal(unpack(pack(mask)), mask))


class TestLessons(unittest.TestCase):
    def test_counting_and_place_value(self):
        m = SymbolMemory()
        self.assertEqual(m.learn_by_counting([glyph(STYLE_A['1'])], 1)['outcome'], 'learned')
        # 10: the 1 is known, so the other symbol is 0
        r = m.learn_by_counting([glyph(STYLE_A['1']), glyph(STYLE_A['0'])], 10)
        self.assertEqual(r, {'outcome': 'learned', 'learned': [0]})
        self.assertEqual(m.recognise(glyph(STYLE_A['0'])), 0)

    def test_choice_by_elimination(self):
        m = memory_of(STYLE_A, '0123')
        # "count the 4 objects: 3 or 4?" with the 4 in a style not yet known: the 3 is ruled out
        r = m.learn_by_choice([glyph(STYLE_A['3']), glyph(STYLE_B['4'])], 4)
        self.assertEqual(r['outcome'], 'learned')
        self.assertEqual(m.recognise(glyph(STYLE_B['4'])), 4)

    def test_choice_undecided_and_contradiction(self):
        m = memory_of(STYLE_A, '0123')
        self.assertEqual(m.learn_by_choice([glyph(STYLE_B['4']), glyph(STYLE_B['5'])], 4)['outcome'], 'pending')
        self.assertEqual(m.learn_by_choice([glyph(STYLE_A['1']), glyph(STYLE_A['2'])], 3)['outcome'],
                         'contradiction')
        self.assertEqual(len(m.examples), 4)

    def test_matching_and_pending_lessons_resolved_later(self):
        m = memory_of(STYLE_A, '0123')
        lessons = [
            # 4 or 5, both unknown in style B: undecided until the matching lesson below has taught the 5
            {'kind': 'choice', 'numerals': [glyph(STYLE_B['4']), glyph(STYLE_B['5'])], 'count': 4},
            {'kind': 'matching', 'numerals': [glyph(STYLE_A['1']), glyph(STYLE_B['5']), glyph(STYLE_A['2'])],
             'counts': [2, 5, 1]},
        ]
        outcomes = m.teach(lessons)
        self.assertEqual([o['outcome'] for o in outcomes], ['learned', 'learned'])
        self.assertEqual(m.recognise(glyph(STYLE_B['4'])), 4)
        self.assertEqual(m.recognise(glyph(STYLE_B['5'])), 5)


class TestLetterLessons(unittest.TestCase):
    CAPS_A = {'A': '.###.|#...#|#...#|#####|#...#|#...#|#...#', 'B': '####.|#...#|#...#|####.|#...#|#...#|####.',
              'C': '.###.|#...#|#....|#....|#....|#...#|.###.', 'D': '####.|#...#|#...#|#...#|#...#|#...#|####.',
              'E': '#####|#....|#....|####.|#....|#....|#####'}
    # a second style: heavier strokes drawn as 3 x 3 blocks of the same letters, on a 7 x 7 grid
    CAPS_B = {'B': '#####..|##..##.|##..##.|#####..|##..##.|##..##.|#####..',
              'C': '.#####.|##...##|##.....|##.....|##.....|##...##|.#####.'}

    def test_an_anchored_order_places_new_styles(self):
        m = SymbolMemory()
        m.learn_in_order([glyph(self.CAPS_A[c]) for c in 'ADE'], list('ADE'))
        # A, B, C, D in order from an unknown start: A and D fix the start, so B and C are learned
        row = [glyph(self.CAPS_A['A']), glyph(self.CAPS_B['B']), glyph(self.CAPS_B['C']), glyph(self.CAPS_A['D'])]
        r = m.learn_by_anchored_order(row, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', domain='upper')
        self.assertEqual(r, {'outcome': 'learned', 'learned': ['B', 'C']})
        self.assertEqual(m.recognise(glyph(self.CAPS_B['C']), 'upper'), 'C')

    def test_an_order_with_too_few_anchors_waits(self):
        m = SymbolMemory()
        m.learn(glyph(self.CAPS_A['A']), 'A')
        row = [glyph(self.CAPS_A['A']), glyph(self.CAPS_B['B']), glyph(self.CAPS_B['C'])]
        self.assertEqual(m.learn_by_anchored_order(row, 'ABCDEFGH', domain='upper')['outcome'], 'pending')

    def test_domains_keep_digits_and_letters_apart(self):
        m = memory_of(STYLE_A, '0')
        m.learn(glyph(self.CAPS_A['C']), 'C')
        self.assertEqual(m.recognise(glyph(STYLE_A['0']), 'digit'), 0)
        self.assertEqual(m.recognise(glyph(STYLE_A['0']), 'upper'), 'C')


class TestScenes(unittest.TestCase):
    def test_a_numeral_and_copies_of_a_picture(self):
        scene = np.zeros((60, 200), int)
        scene[5:47, 5:35] = glyph(STYLE_A['3'])
        ring = np.zeros((20, 20), int)
        ring[2:18, 2:18] = 1
        ring[5:15, 5:15] = 0
        ring[9:11, 9:11] = 1                    # a dot inside: a picture of two parts
        for k in range(3):
            scene[20:40, 60 + 40 * k:80 + 40 * k] = ring
        pictures, numerals = counting_scene(scene)
        self.assertEqual(len(pictures), 3)
        self.assertEqual(len(numerals), 1)


class TestNumberWorksheet(unittest.TestCase):
    def test_the_learned_digits_read_the_sheets_cells(self):
        import children_worksheets as cw
        from digit_lessons import LEARNED
        m = SymbolMemory(LEARNED)
        truth = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [26, 24, 22, 20, 18], [27, 24, 21, 18, 15],
                 [17, 19, 21, 23, 25], [36, 32, 28, 24, 20], [15, 16, 17, 18, 19]]
        cells = cw._number_cells()
        self.assertEqual([[m.read_number(cells[r][c]) for c in range(5)] for r in range(7)], truth)
        self.assertTrue(all(row[5] is None for row in cells))      # the answer boxes are empty

    def test_the_learned_letters_read_the_order_of_letters_sheets(self):
        import children_worksheets as cw
        from digit_lessons import LEARNED
        m = SymbolMemory(LEARNED)
        cells = cw._letter_cells()
        right = total = 0
        for k, rows in enumerate(cw._LETTER_ROWS):
            for r, row in enumerate(rows):
                domain = 'upper' if row.strip('_').isupper() else 'lower'
                for c, ch in enumerate(row):
                    if ch != '_':
                        total += 1
                        right += m.recognise(cells[k][r][c], domain) == ch
        self.assertEqual(total, 78)
        self.assertGreaterEqual(right, 73)


if __name__ == '__main__':
    unittest.main()
