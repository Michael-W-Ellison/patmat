#!/usr/bin/env python3
"""Tests for tactical_patterns and its use in general_move_patterns"""

import os
import tempfile
import unittest

import chess

from general_move_patterns import GeneralPatternLearner, analyze_candidates, split_patterns
from tactical_patterns import classify
from test_move_judges import Race


class TestClassify(unittest.TestCase):
    def test_classes(self):
        now = (0.25, 10.0)
        self.assertEqual(classify(now, None), 'tac=none|top=same')
        self.assertEqual(classify(now, (0.1, 5.0, 1, True)), 'tac=wins|top=down')
        self.assertEqual(classify(now, (0.5, 30.0, 3, False)), 'tac=multi|top=up')
        self.assertEqual(classify(now, (0.5, 10.0, 1, False)), 'tac=more|top=same')
        self.assertEqual(classify(now, (0.25, 10.0, 2, False)), 'tac=same|top=same')
        self.assertEqual(classify(now, (0.0, 0.0, 0, False)), 'tac=fewer|top=down')


class TestAnalysis(unittest.TestCase):
    def test_a_win_the_answer_cannot_stop(self):
        # at 3, adding 1 leaves 4: whatever the opponent adds, the agent then reaches 7; adding 2 lets the
        # opponent win at once
        features = [a['features'] for a in analyze_candidates(Race(), (3, 0), 0, [1, 2], tactics=True)]
        self.assertTrue(features[0].endswith('|tac=wins|top=up'))
        self.assertTrue(features[1].endswith('|tac=none|top=same'))
        plain = [a['features'] for a in analyze_candidates(Race(), (3, 0), 0, [1, 2])]
        self.assertEqual([f.rsplit('|', 2)[0] for f in features], plain)

    def test_a_knight_fork_leaves_a_bigger_gain_after_the_answer(self):
        from chess_adapter import ChessAdapter
        board = chess.Board('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1')
        moves = list(board.legal_moves)
        analysis = analyze_candidates(ChessAdapter(), board, chess.WHITE, moves, max_replies=float('inf'),
                                      tactics=True)
        up = [board.san(m) for m, a in zip(moves, analysis) if a['features'].endswith('top=up')]
        # Nc7+ forks the king and the rook: after the king moves, Nxa8 wins the rook
        self.assertEqual(up, ['Nc7+'])


class TestLearner(unittest.TestCase):
    def test_the_two_halves_are_learned_and_their_values_added(self):
        self.assertEqual(split_patterns('a|b|c|d'), ['a|b|c|d'])
        self.assertEqual(split_patterns('a|b|c|d|tac=x|top=y'), ['a|b|c|d', 'tac=x|top=y'])
        learner = GeneralPatternLearner(os.path.join(tempfile.mkdtemp(), 'u.db'), 'race')
        for _ in range(30):
            learner.record_game(['a|b|c|d|tac=wins|top=up'], 'win')
            learner.record_game(['a|b|c|d|tac=fewer|top=down'], 'loss')
        self.assertGreater(learner.value('tac=wins|top=up'), 0.0)
        self.assertLess(learner.value('tac=fewer|top=down'), 0.0)
        self.assertEqual(learner.value('a|b|c|d|tac=wins|top=up'),
                         learner.value('a|b|c|d') + learner.value('tac=wins|top=up'))
        self.assertGreater(learner.part_value('tac=wins'), learner.part_value('tac=fewer'))
        learner.close()

    def test_analyze_appends_the_tactical_parts_when_on(self):
        learner = GeneralPatternLearner(os.path.join(tempfile.mkdtemp(), 'u.db'), 'race')
        self.assertEqual(len(learner.analyze(Race(), (3, 0), 0, [1, 2])[0]['features'].split('|')), 4)
        learner.TACTICS = True
        self.assertEqual(len(learner.analyze(Race(), (3, 0), 0, [1, 2])[0]['features'].split('|')), 6)
        learner.close()


if __name__ == '__main__':
    unittest.main()
