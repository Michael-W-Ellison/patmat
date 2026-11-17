#!/usr/bin/env python3
"""Tests for lookahead_patterns (learning which positions hide a tactic) and its use in pattern_search"""

import os
import tempfile
import unittest

import chess

import lookahead_patterns as L
from pattern_search import PatternSearch
from sequence_patterns import SequencePatternLearner

# White to move: Nc7+ forks the king on e8 and the rook on a6
FORK = '4k3/8/r7/1N6/8/8/8/4K3 w - - 0 1'


def search(depth):
    db = os.path.join(tempfile.mkdtemp(), 't.db')
    return PatternSearch(SequencePatternLearner(db), seconds=30, selective=True, max_depth=depth)


class TestSignature(unittest.TestCase):
    def test_tension_described(self):
        b = chess.Board(FORK)
        b.push_san('Nc7+')
        # black in check, its rook attacked and undefended (a rook's worth at stake); white's knight loose
        self.assertEqual(L.signature(b), 'c m1R t0 p0 l1 o1')
        # at the start each side's rooks are undefended (loose)
        self.assertEqual(L.signature(chess.Board()), '- m0 t0 p0 l2 o2')

    def test_threats_and_pins(self):
        # black's knight on c6 pinned by the bishop on b5 to the king on e8, and attacked by the pawn on d5
        b = chess.Board('4k3/8/2n5/1B1P4/8/8/8/4K3 b - - 0 1')
        self.assertEqual(L.signature(b), '- m1M t0 p1 l1 o1')


class TestLessons(unittest.TestCase):
    def test_a_fork_is_hidden_from_a_glance(self):
        s = search(4)
        b = chess.Board(FORK)
        self.assertLess(s.glance(b), L.SWING)                  # captures only: nothing to win
        self.assertGreaterEqual(s.look(b, 4, 30) - s.glance(b), L.SWING)    # the fork wins the rook
        learner = L.LookaheadLearner(os.path.join(tempfile.mkdtemp(), 'l.db'))
        self.assertIs(learner.lesson(b, s), True)
        self.assertEqual(learner.counts[L.signature(b)], [1, 1])

    def test_a_game_gives_lessons_that_are_kept(self):
        path = os.path.join(tempfile.mkdtemp(), 'l.db')
        learner = L.LookaheadLearner(path, seed=0)
        moves = [chess.Move.from_uci(m) for m in ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'f3g5', 'd7d5')]
        self.assertEqual(learner.record_game(moves, search(2)), len(moves))
        learner.close()
        again = L.LookaheadLearner(path)
        self.assertEqual(sum(n for n, _ in again.counts.values()), len(moves))


class TestReadingFurther(unittest.TestCase):
    def test_the_fork_is_seen_when_its_kind_of_position_is_read_further(self):
        after = chess.Board(FORK)
        after.push_san('Nc7+')                  # black to move, forked: at a line's end
        learner = L.LookaheadLearner(os.path.join(tempfile.mkdtemp(), 'l.db'))
        learner.counts = {L.signature(after): [100, 100], 'other': [1000, 0]}   # that kind often hides a swing

        def value(lookahead):
            s = search(1)
            s.lookahead = lookahead
            s.deadline, s.history = float('inf'), set()
            return s.negamax(after.copy(), 0, -float('inf'), float('inf'), [], 1), s.extensions
        plain, n_plain = value(None)
        reading, n_reading = value(learner)
        self.assertEqual(n_plain, 0)
        self.assertEqual(n_reading, 1)
        self.assertGreater(plain, 0)                           # a glance: black is a rook for a knight up
        self.assertLessEqual(reading, plain - 50)              # read further: the rook is lost
        # a kind of position that rarely hid anything is not read further
        learner.counts = {L.signature(after): [100, 0], 'other': [1000, 0]}
        learner._rates = None
        self.assertEqual(value(learner)[1], 0)

if __name__ == '__main__':
    unittest.main()
