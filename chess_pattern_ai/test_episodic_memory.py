#!/usr/bin/env python3
"""Tests for episodic_memory (episodes of punished moves, found in hindsight; consolidated into refutation patterns)"""

import os
import sqlite3
import tempfile
import unittest

import chess

import episodic_memory as E
from chess_adapter import ChessAdapter
from pattern_search import PatternSearch
from refutation_patterns import RefutationLearner
from sequence_patterns import SequencePatternLearner

# 2. Bc4?? hangs the bishop to the pawn on d5
GAME = ['e2e4', 'd7d5', 'f1c4', 'd5c4', 'g1f3', 'g8f6', 'b1c3', 'b8c6']


def search(depth=3, seconds=30):
    return PatternSearch(SequencePatternLearner(os.path.join(tempfile.mkdtemp(), 's.db')), seconds=seconds,
                         selective=True, max_depth=depth)


def memory():
    conn = sqlite3.connect(':memory:')
    semantic = RefutationLearner(conn, 'chess')
    return E.EpisodicMemory(os.path.join(tempfile.mkdtemp(), 'e.db'), semantic=semantic, adapter=ChessAdapter())


class TestRecallKey(unittest.TestCase):
    def test_the_same_jump_from_either_side(self):
        white = chess.Board('4k3/8/8/1N6/8/8/8/4K3 w - - 0 1')
        black = chess.Board('4k3/8/8/8/1n6/8/8/4K3 b - - 0 1')
        self.assertEqual(E.key(white, chess.Move.from_uci('b5c7')), E.key(black, chess.Move.from_uci('b4c2')))
        self.assertNotEqual(E.key(white, chess.Move.from_uci('b5c7')), E.key(white, chess.Move.from_uci('b5d6')))


class TestEpisodes(unittest.TestCase):
    def test_a_hung_piece_is_remembered_with_its_punishment(self):
        m = memory()
        found = m.record_game([chess.Move.from_uci(u) for u in GAME], chess.WHITE, search())
        self.assertEqual([(e['move'], e['reply'], e['blunderer']) for e in found], [('f1c4', 'd5c4', 'agent')])
        self.assertEqual(found[0]['lost'], 33)
        self.assertEqual(len(m), 1)
        # recalled: the same capture, and the same capture by the other side (d4xc5 for white)
        after = chess.Board()
        for u in GAME[:3]:
            after.push_uci(u)
        self.assertTrue(m.recall(after, chess.Move.from_uci('d5c4')))
        self.assertTrue(m.recall(chess.Board('4k3/8/8/2b5/3P4/8/8/4K3 w - - 0 1'), chess.Move.from_uci('d4c5')))
        self.assertFalse(m.recall(after, chess.Move.from_uci('g8f6')))
        # consolidated: the semantic memory has evidence of what kind of reply punished
        self.assertTrue(m.semantic.has_evidence())

    def test_episodes_are_kept_between_runs(self):
        path = os.path.join(tempfile.mkdtemp(), 'e.db')
        E.EpisodicMemory(path).record_game([chess.Move.from_uci(u) for u in GAME], chess.WHITE, search())
        again = E.EpisodicMemory(path)
        self.assertEqual(len(again), 1)
        self.assertEqual(again.games, 1)

    def test_a_quiet_game_has_no_episodes(self):
        m = memory()
        quiet = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']
        self.assertEqual(m.record_game([chess.Move.from_uci(u) for u in quiet], chess.WHITE, search()), [])


class TestRecallInTheSearch(unittest.TestCase):
    def test_a_remembered_punishment_is_read_further(self):
        m = memory()
        m.record_game([chess.Move.from_uci(u) for u in GAME], chess.WHITE, search())
        board = chess.Board()
        for u in GAME[:2]:
            board.push_uci(u)
        s = search(2)
        s.episodes = m
        move = s.choose(board)
        self.assertGreater(s.recalls, 0)
        self.assertNotEqual(move, chess.Move.from_uci('f1c4'))
        # the deeper look used for episodes does not recall
        s.recalls = 0
        s.look(board, 2, 30)
        self.assertEqual(s.recalls, 0)


if __name__ == '__main__':
    unittest.main()
