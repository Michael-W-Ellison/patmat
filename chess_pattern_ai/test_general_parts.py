#!/usr/bin/env python3
"""Tests for the part values of the game-independent patterns (GeneralPatternLearner.part_value)"""

import os
import tempfile
import unittest

from general_move_patterns import GeneralPatternLearner

RESTRICTS = 'gain=zero|net=0|mob=restricts|threat=safe'
FREES = 'gain=zero|net=0|mob=frees|threat=safe'
RESTRICTS_BEST = 'gain=best|net=+|mob=restricts|threat=safe'


class TestPartValues(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def test_a_part_is_valued_over_every_pattern_that_has_it(self):
        other = GeneralPatternLearner(self.path, 'othello')
        for i in range(40):
            # games in which restricting moves were played are won, the others lost
            other.record_game([RESTRICTS if i % 2 else RESTRICTS_BEST], 'win')
            other.record_game([FREES], 'loss')
        self.assertGreater(other.part_value('mob=restricts'), 0.0)
        self.assertLess(other.part_value('mob=frees'), 0.0)
        # both restricting patterns count toward 'mob=restricts', so it has more evidence than either
        self.assertGreater(other.part_value('mob=restricts'), other.value(RESTRICTS))
        other.close()

        # another game imports the evidence at reduced weight; without importing, it has none
        chess_import = GeneralPatternLearner(self.path, 'chess')
        self.assertGreater(chess_import.part_value('mob=restricts'), 0.0)
        self.assertLess(chess_import.part_value('mob=restricts'),
                        GeneralPatternLearner(self.path, 'othello').part_value('mob=restricts'))
        self.assertEqual(GeneralPatternLearner(self.path, 'chess', import_cross_game=False)
                         .part_value('mob=restricts'), 0.0)


if __name__ == '__main__':
    unittest.main()
