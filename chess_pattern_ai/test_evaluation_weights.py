#!/usr/bin/env python3
"""Tests for evaluation_weights (weighted position features, tuned by the cross-entropy method)"""

import os
import random
import tempfile
import unittest

import chess

import evaluation_weights as W
from pattern_search import PatternSearch
from sequence_patterns import SequencePatternLearner

# white: knight f3 defended, king g1 behind three pawns, doubled and isolated c-pawns; black: a passed pawn on a3,
# an undefended knight on a5
UNEVEN = '7k/8/8/n7/8/p4N2/2P2PPP/2P3K1 w - - 0 1'


class TestFeatures(unittest.TestCase):
    def test_symmetric_positions_are_even(self):
        self.assertEqual(W.features(chess.Board()), [0.0] * len(W.FEATURES))

    def test_features_of_an_uneven_position(self):
        f = dict(zip(W.FEATURES, W.features(chess.Board(UNEVEN))))
        self.assertEqual(f['king_shield'], 3)
        self.assertEqual(f['loose'], -1)            # black's knight is loose, white's is not
        self.assertEqual(f['pawn_weakness'], 2)     # white 3 (one doubled, two isolated), black 1
        self.assertEqual(f['centre'], 2)            # the knight on f3 attacks d4 and e5
        self.assertAlmostEqual(f['mobility'], (6 - 4) / 5)

    def test_the_other_side_to_move_sees_the_opposite(self):
        b = chess.Board(UNEVEN)
        white = W.features(b)
        b.turn = chess.BLACK
        self.assertEqual([round(-x, 9) for x in white], [round(x, 9) for x in W.features(b)])


def reference(board):
    """The features worked out square by square (the first, slower version), for comparison"""
    def side(color):
        other, own = not color, board.occupied_co[color]
        pieces = own & (board.knights | board.bishops | board.rooks | board.queens)
        mobility = sum(chess.popcount(board.attacks_mask(sq) & ~own) for sq in chess.scan_forward(pieces))
        loose = sum(1 for sq in chess.scan_forward(pieces) if not board.attackers_mask(color, sq))
        king = board.king(color)
        danger = sum(1 for sq in chess.scan_forward(chess.BB_KING_ATTACKS[king]) if board.attackers_mask(other, sq))
        shield = 0
        front = king + (8 if color == chess.WHITE else -8)
        if 0 <= front < 64:
            for df in (-1, 0, 1):
                if 0 <= chess.square_file(king) + df < 8 and board.piece_at(front + df) == chess.Piece(chess.PAWN, color):
                    shield += 1
        pawns, theirs = board.pawns & own, board.pawns & board.occupied_co[other]
        files = [chess.popcount(pawns & chess.BB_FILES[f]) for f in range(8)]
        weak = sum(n - 1 for n in files if n > 1) + sum(
            n for f, n in enumerate(files) if n and not (f > 0 and files[f - 1]) and not (f < 7 and files[f + 1]))
        passed = 0
        for sq in chess.scan_forward(pawns):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
            if not any(theirs & chess.BB_SQUARES[chess.square(ff, rr)] for ff in (f - 1, f, f + 1) if 0 <= ff < 8
                       for rr in ahead):
                passed += r if color == chess.WHITE else 7 - r
        centre = sum(1 for sq in W.CENTRE if board.attackers_mask(color, sq)) + chess.popcount(own & int(W.CENTRE))
        undeveloped = chess.popcount((board.knights | board.bishops) & own & int(W.HOME[color]))
        return [mobility / 5.0, danger, shield, loose, weak, passed / 3.0, centre, undeveloped]
    return [a - b for a, b in zip(side(board.turn), side(not board.turn))]


class TestFastFeatures(unittest.TestCase):
    def test_the_same_as_square_by_square(self):
        rng = random.Random(1)
        for _ in range(150):
            b = chess.Board()
            for _ in range(rng.randint(0, 120)):
                if b.is_game_over():
                    break
                b.push(rng.choice(list(b.legal_moves)))
            for again in range(2):          # the second time from the cache
                for x, y in zip(W.features(b), reference(b)):
                    self.assertAlmostEqual(x, y, msg=b.fen())


class TestTuner(unittest.TestCase):
    def test_the_mean_moves_toward_the_best_games(self):
        t = W.WeightTuner(os.path.join(tempfile.mkdtemp(), 'w.db'), seed=0)
        n = len(W.FEATURES)
        for i in range(W.CE_POPULATION):
            # the games with the largest first weight score best
            t.record([float(i)] + [0.0] * (n - 1), score=float(i))
        best = list(range(W.CE_POPULATION))[-W.CE_ELITE:]
        self.assertAlmostEqual(t.mean()[0], W.CE_SMOOTH * sum(best) / W.CE_ELITE)
        self.assertEqual(t.generations, 1)
        self.assertEqual(t.done, [])
        self.assertGreaterEqual(min(t.std), W.CE_MIN_STD)

    def test_the_distribution_is_kept(self):
        path = os.path.join(tempfile.mkdtemp(), 'w.db')
        t = W.WeightTuner(path, seed=0)
        w = t.draw()
        t.record(w, 0.3)
        t.close()
        again = W.WeightTuner(path)
        self.assertEqual(again.episodes, 1)
        self.assertEqual(again.done, [[0.3, w]])

    def test_episode_score(self):
        self.assertGreater(W.episode_score('loss', [-5.0]), W.episode_score('loss', [-50.0]))
        self.assertGreater(W.episode_score('win', [-500.0]), W.episode_score('draw', [500.0]))


class TestSearch(unittest.TestCase):
    def test_the_weights_enter_the_evaluation(self):
        s = PatternSearch(SequencePatternLearner(os.path.join(tempfile.mkdtemp(), 's.db')), selective=True)
        b = chess.Board(UNEVEN)
        plain = s.evaluate(b)
        s.weights = [0.0, 0.0, 1.0] + [0.0] * (len(W.FEATURES) - 3)
        self.assertAlmostEqual(s.evaluate(b), plain + 3)


if __name__ == '__main__':
    unittest.main()
