#!/usr/bin/env python3
"""
Unit tests for multi-move (sequence) patterns.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

if CHESS_AVAILABLE:
    from sequence_patterns import SequencePatternLearner, token, MATE_VALUE


def moves_of(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestSequencePatterns(unittest.TestCase):

    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(handle)
        self.learner = SequencePatternLearner(self.db_path)

    def tearDown(self):
        self.learner.close()
        os.unlink(self.db_path)

    def test_token_describes_a_fork(self):
        board = chess.Board('3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')
        self.assertEqual(token(board, chess.Move.from_uci('e5f7')), 'N-+ a2 s')
        self.assertEqual(board.fen(), '3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')     # unchanged

    def test_token_marks_a_capture_left_hanging(self):
        # the queen takes a pawn the c6 pawn defends, and attacks that pawn
        board = chess.Board('4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1')
        self.assertEqual(token(board, chess.Move.from_uci('d1d5')), 'QxP a1 h')

    def test_mating_sequence_is_learned_for_both_sides(self):
        game = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#'])
        self.learner.record_game(list(game.move_stack), chess.WHITE, 'win')

        # White's mating move, after Black's knight move: a large gain for White
        before_mate = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6'])
        mate = chess.Move.from_uci('h5f7')
        self.assertGreater(self.learner.value(before_mate, mate), MATE_VALUE / 10)

        # Black's Nf6, which allowed the mate within the horizon: a large loss for Black
        before_nf6 = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4'])
        self.assertLess(self.learner.value(before_nf6, chess.Move.from_uci('g8f6')), -MATE_VALUE / 10)

        # Multi-move patterns are recorded (Bc4, Nf6, Qxf7# as one three-ply pattern)
        three = [p for p in self.learner.stats if p.count(' | ') == 2 and p.endswith(token(before_mate, mate))]
        self.assertTrue(three)

    def test_statistics_persist(self):
        game = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#'])
        self.learner.record_game(list(game.move_stack), chess.WHITE, 'win')
        count = len(self.learner.stats)
        self.learner.close()
        self.learner = SequencePatternLearner(self.db_path)
        self.assertEqual(len(self.learner.stats), count)
        self.assertIn('ai', self.learner.baseline)


if __name__ == '__main__':
    unittest.main()
