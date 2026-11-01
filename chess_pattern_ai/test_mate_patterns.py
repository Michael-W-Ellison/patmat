#!/usr/bin/env python3
"""Unit tests for mate patterns (mate_patterns)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

if CHESS_AVAILABLE:
    from mate_patterns import MatePatternLearner, surroundings


def moves_of(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


SCHOLARS_MATE = ['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#']


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestMatePatterns(unittest.TestCase):

    def test_back_rank_mate(self):
        board = chess.Board('3R2k1/5ppp/8/8/8/8/8/6K1 b - - 0 1')
        self.assertTrue(board.is_checkmate())
        self.assertEqual(surroundings(board, chess.BLACK), 'f0 b3 a2 e3 +')
        # the square behind the king on the checking rank counts as attacked
        board = chess.Board('3R3k/6pp/8/8/8/8/8/6K1 b - - 0 1')
        self.assertEqual(surroundings(board, chess.BLACK), 'f0 b2 a1 e5 +')

    def test_board_unchanged(self):
        board = moves_of(SCHOLARS_MATE[:-1])
        fen, stack = board.fen(), list(board.move_stack)
        surroundings(board, chess.BLACK)
        self.assertEqual((board.fen(), list(board.move_stack)), (fen, stack))

    def test_learns_the_surroundings_before_mate(self):
        learner = MatePatternLearner(':memory:')
        before = moves_of(SCHOLARS_MATE[:-1])           # White to move, about to mate
        self.assertEqual(learner.value(before), 0.0)
        for _ in range(10):
            learner.record_game(list(moves_of(SCHOLARS_MATE).move_stack))
        # Black's king's surroundings are now known to precede mate: good for White, the side to move
        self.assertGreater(learner.value(before), 0.0)
        self.assertGreater(learner.rate(surroundings(before, chess.BLACK)), 0.5)
        learner.close()


if __name__ == '__main__':
    unittest.main()
