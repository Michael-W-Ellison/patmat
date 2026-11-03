#!/usr/bin/env python3
"""Tests for refutation_patterns and its use in general_move_patterns and pattern_search"""

import os
import sqlite3
import tempfile
import unittest

import chess

from general_move_patterns import WIN_VALUE, GeneralPatternLearner, analyze_candidates
from move_judges import SearchJudge
from refutation_patterns import RefutationLearner, reply_features
from test_move_judges import Race


class TestReplyFeatures(unittest.TestCase):
    def test_terminal_and_quiet_replies(self):
        # at 5, adding 2 reaches 7 and ends the game; adding 1 does not
        self.assertEqual(reply_features(Race(), (5, 1), 1, [1, 2]), ['gain=zero|mob=neutral', 'gain=zero|mob=terminal'])


class TestRefutationLearner(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), 'u.db')

    def test_the_pattern_that_refutes_is_rated_above_the_others(self):
        learner = RefutationLearner(sqlite3.connect(self.db), 'one')
        for _ in range(20):
            learner.record(['r', 'q', 'q', 'q'], 0)
        self.assertGreater(learner.rate('r'), 1.5)
        self.assertLess(learner.rate('q'), 1.0)
        self.assertEqual(learner.rate('unseen'), 1.0)
        self.assertEqual(learner.ranked(['q', 'r', 'q']), [1])

    def test_other_games_evidence_is_imported(self):
        conn = sqlite3.connect(self.db)
        one = RefutationLearner(conn, 'one')
        for _ in range(20):
            one.record(['r', 'q', 'q', 'q'], 0)
        self.assertGreater(RefutationLearner(conn, 'two').rate('r'), 1.0)
        self.assertEqual(RefutationLearner(conn, 'two', import_cross_game=False).rate('r'), 1.0)


class TestExtension(unittest.TestCase):
    def test_a_loss_beyond_one_ply_is_found_through_a_rated_reply(self):
        # at 2: adding 1 leaves 3, the opponent adds 1 to 4 and the agent then loses; adding 2 leaves 4 (wins)
        db = os.path.join(tempfile.mkdtemp(), 'u.db')
        learner = RefutationLearner(sqlite3.connect(db), 'race')
        for _ in range(20):
            learner.record(['gain=zero|mob=neutral', 'gain=zero|mob=terminal'], 0)
        plain = analyze_candidates(Race(), (2, 0), 0, [1, 2])
        extended = analyze_candidates(Race(), (2, 0), 0, [1, 2], refutations=learner)
        self.assertEqual(plain[0]['net'], 0.0)
        self.assertLessEqual(extended[0]['net'], -WIN_VALUE / 2)
        self.assertGreater(extended[1]['net'], -WIN_VALUE / 2)
        self.assertEqual([a['features'] for a in plain], [a['features'] for a in extended])


class TestLearning(unittest.TestCase):
    def test_a_blunder_records_the_judges_refutation(self):
        db = os.path.join(tempfile.mkdtemp(), 'u.db')
        learner = GeneralPatternLearner(db, 'race')
        learner.judge = SearchJudge(Race(), depth=4)
        learner.LEARN_REFUTATIONS = True
        learner.record_judged([('f|f|f|f', 1.0)])       # the game's median loss is now 1
        # at 3, adding 2 leaves 5: the opponent adds 2 and wins
        learner.note_move(Race(), (3, 0), 0, 2, 'f|f|f|f')
        self.assertGreater(learner.last_loss, 2.0)
        self.assertGreater(learner.refutations.rate('gain=zero|mob=terminal'), 1.0)
        self.assertLess(learner.refutations.rate('gain=zero|mob=neutral'), 1.0)
        learner.close()


class FakeRefutations:
    """Rates one reply above the others"""

    def __init__(self, uci):
        self.uci = uci
        self.features_seen = None

    def has_evidence(self):
        return True

    def ranked(self, features, limit, exclude=None):
        return [self.index]


class TestChessSearch(unittest.TestCase):
    def test_root_replies_include_the_rated_reply(self):
        from chess_adapter import ChessAdapter
        from pattern_search import PatternSearch
        from sequence_patterns import SequencePatternLearner
        db = os.path.join(tempfile.mkdtemp(), 'c.db')
        sequences = SequencePatternLearner(db)
        search = PatternSearch(sequences, selective=True)
        board = chess.Board()
        board.push_uci('e2e4')
        moves = search.ordered(board, [])
        plain = [e[0] for e in search.candidates(board, [], 1)]
        target = next(i for i, e in enumerate(moves) if e[0] not in plain)
        fake = FakeRefutations(None)
        fake.index = target
        search.refutations, search.adapter = fake, ChessAdapter()
        self.assertIn(moves[target][0], [e[0] for e in search.candidates(board, [], 1)])
        # only below root moves (ply 1)
        self.assertNotIn(moves[target][0], [e[0] for e in search.candidates(board, [], 2)])
        sequences.close()


if __name__ == '__main__':
    unittest.main()
