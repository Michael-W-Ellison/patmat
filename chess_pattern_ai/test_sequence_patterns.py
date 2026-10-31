#!/usr/bin/env python3
"""
Unit tests for multi-move (sequence) patterns, learned piece categories and
the pattern-guided search.
"""

import os
import random
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
    from piece_knowledge import PieceKnowledge
    from sequence_patterns import SequencePatternLearner, MATE_VALUE
    from pattern_search import PatternSearch


def moves_of(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def watch_random_games(pieces, games=20, seed=0):
    """Let pieces observe the moves of random games"""
    rng = random.Random(seed)
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 200:
            move = rng.choice(list(board.legal_moves))
            pieces.observe(board, move)
            board.push(move)


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestPieceKnowledge(unittest.TestCase):

    def test_categories_are_learned_from_moves_watched(self):
        pieces = PieceKnowledge(':memory:')
        self.assertEqual(pieces.category(chess.KNIGHT), 'u')        # nothing watched yet
        watch_random_games(pieces)
        learned = {p: pieces.category(p) for p in chess.PIECE_TYPES}
        self.assertEqual(learned, {chess.PAWN: 'o:dSf', chess.KNIGHT: 'jS', chess.BISHOP: 'dL',
                                   chess.ROOK: 'oL', chess.QUEEN: 'doL', chess.KING: 'doS'})
        pieces.close()


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestSequencePatterns(unittest.TestCase):

    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(handle)
        self.learner = SequencePatternLearner(self.db_path)
        watch_random_games(self.learner.pieces)

    def tearDown(self):
        self.learner.close()
        os.unlink(self.db_path)

    def test_token_describes_a_fork(self):
        # the knight checks the king and attacks the rook (worth more than the knight)
        board = chess.Board('3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')
        self.assertEqual(self.learner.token(board, chess.Move.from_uci('e5f7')), 'jS -+ a2> s')
        self.assertEqual(board.fen(), '3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')     # unchanged

    def test_token_marks_a_piece_a_cheaper_piece_attacks(self):
        # the queen takes a pawn that the c6 pawn defends
        board = chess.Board('4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1')
        self.assertEqual(self.learner.token(board, chess.Move.from_uci('d1d5')), 'doL xo:dSf a1 c')

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
        last = self.learner.token(before_mate, mate)
        self.assertTrue([p for p in self.learner.stats if p.count(' | ') == 2 and p.endswith(last)])

    def test_statistics_persist(self):
        game = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#'])
        self.learner.record_game(list(game.move_stack), chess.WHITE, 'win')
        count = len(self.learner.stats)
        self.learner.close()
        self.learner = SequencePatternLearner(self.db_path)
        self.assertEqual(len(self.learner.stats), count)
        self.assertIn('ai', self.learner.baseline)
        self.assertEqual(self.learner.pieces.category(chess.KNIGHT), 'jS')


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestPatternSearch(unittest.TestCase):

    def setUp(self):
        self.learner = SequencePatternLearner(':memory:')
        watch_random_games(self.learner.pieces)

    def tearDown(self):
        self.learner.close()

    def test_finds_a_mate_in_one(self):
        board = chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')
        self.assertEqual(PatternSearch(self.learner, seconds=1.0).choose(board), chess.Move.from_uci('a1a8'))

    def test_wins_material_two_moves_deep(self):
        # the knight fork Nc7+ wins the rook on a8 a move later
        board = chess.Board('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1')
        search = PatternSearch(self.learner, seconds=2.0)
        self.assertEqual(search.choose(board), chess.Move.from_uci('b5c7'))
        self.assertGreaterEqual(search.depth_reached, 3)


if __name__ == '__main__':
    unittest.main()
