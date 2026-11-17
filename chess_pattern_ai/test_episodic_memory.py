#!/usr/bin/env python3
"""Tests for episodic_memory (episodes of punished moves, found in hindsight; recalled by situation;
consolidated into refutation patterns)"""

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

# 2. Bc4?? hangs the bishop to the pawn on d5: an episode, but its punishment takes at once (no situation)
GAME = ['e2e4', 'd7d5', 'f1c4', 'd5c4', 'g1f3', 'g8f6', 'b1c3', 'b8c6']
# white's Nc7+ forks the king on e8 and the rook on a8
FORK_WHITE = 'r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1'
# black's ...Nc2+ forks the king on e1 and the rook on a1: the same situation elsewhere, for the other side
FORK_BLACK = '4k3/8/8/8/1n6/8/8/R3K3 b - - 0 1'


def search(depth=3, seconds=30):
    return PatternSearch(SequencePatternLearner(os.path.join(tempfile.mkdtemp(), 's.db')), seconds=seconds,
                         selective=True, max_depth=depth)


def memory():
    conn = sqlite3.connect(':memory:')
    semantic = RefutationLearner(conn, 'chess')
    return E.EpisodicMemory(os.path.join(tempfile.mkdtemp(), 'e.db'), semantic=semantic, adapter=ChessAdapter())


class TestSituation(unittest.TestCase):
    def test_a_fork_described_wherever_it_is(self):
        fork = E.situation(chess.Board(FORK_WHITE), chess.Move.from_uci('b5c7'))
        self.assertEqual(fork, (chess.KNIGHT, 0, True, (chess.ROOK,), 1, 0))
        self.assertEqual(E.situation(chess.Board(FORK_BLACK), chess.Move.from_uci('b4c2')), fork)
        # a move that leaves nothing a single answer cannot parry is no situation
        self.assertIsNone(E.situation(chess.Board(FORK_WHITE), chess.Move.from_uci('e1d1')))
        self.assertIsNone(E.situation(chess.Board(FORK_WHITE), chess.Move.from_uci('b5a7')))   # attacks the rook only

    def test_a_discovered_check(self):
        # the bishop leaves the rook's line to the king, giving check, and attacks the undefended queen
        board = chess.Board('3k4/8/8/8/3B3q/8/8/3RK3 w - - 0 1')
        self.assertEqual(E.situation(board, chess.Move.from_uci('d4f6')),
                         (chess.BISHOP, 0, True, (chess.QUEEN,), 1, 0))


class TestEpisodes(unittest.TestCase):
    def test_a_hung_piece_is_an_episode(self):
        m = memory()
        found = m.record_game([chess.Move.from_uci(u) for u in GAME], chess.WHITE, search())
        self.assertEqual([(e['move'], e['reply'], e['blunderer']) for e in found], [('f1c4', 'd5c4', 'agent')])
        self.assertEqual(found[0]['lost'], 33)
        self.assertEqual(m.episodes, 1)
        self.assertEqual(len(m), 0)                     # a capture at once: nothing to recall
        self.assertTrue(m.semantic.has_evidence())      # consolidated all the same

    def test_episodes_are_kept_between_runs(self):
        path = os.path.join(tempfile.mkdtemp(), 'e.db')
        E.EpisodicMemory(path).record_game([chess.Move.from_uci(u) for u in GAME], chess.WHITE, search())
        again = E.EpisodicMemory(path)
        self.assertEqual(again.episodes, 1)
        self.assertEqual(again.games, 1)

    def test_a_quiet_game_has_no_episodes(self):
        m = memory()
        quiet = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']
        self.assertEqual(m.record_game([chess.Move.from_uci(u) for u in quiet], chess.WHITE, search()), [])


class TestRecall(unittest.TestCase):
    def test_a_remembered_fork_is_recalled_in_another_position(self):
        m = memory()
        m._remember(chess.Board(FORK_WHITE), chess.Move.from_uci('b5c7'))
        self.assertEqual(len(m), 1)
        other = chess.Board(FORK_BLACK)
        self.assertTrue(m.recall(other, chess.Move.from_uci('b4c2')))
        self.assertFalse(m.recall(other, chess.Move.from_uci('b4d5')))         # the same piece, no fork
        self.assertFalse(m.recall(other, chess.Move.from_uci('e8d7')))

    def test_a_remembered_fork_is_read_in_the_search(self):
        m = memory()
        m._remember(chess.Board(FORK_WHITE), chess.Move.from_uci('b5c7'))
        board = chess.Board('4k3/8/8/8/1n6/8/7P/R3K3 w - - 0 1')              # after h3, ...Nc2+ forks
        s = search(2)
        s.episodes = m
        s.choose(board)
        self.assertGreater(s.recalls, 0)
        # the deeper look used for episodes does not recall
        s.recalls = 0
        s.look(board, 2, 30)
        self.assertEqual(s.recalls, 0)


    def test_memory_at_the_root_only(self):
        m = memory()
        m._remember(chess.Board(FORK_WHITE), chess.Move.from_uci('b5c7'))
        board = chess.Board('4k3/8/8/8/1n6/8/7P/R3K3 w - - 0 1')
        s = search(1)
        s.episodes = m
        s.memory_mode = 'root'
        s.history, s.depth_reached = set(), 1
        # suppose the search's best moves were h3 (which allows ...Nc2+, the remembered fork), Kd1 and Ra2
        h3, kd1, ra2 = (chess.Move.from_uci(u) for u in ('h2h3', 'e1d1', 'a1a2'))
        root = [(h3, 0.0, None), (kd1, 0.0, None), (ra2, 0.0, None)]
        chosen = s.check(board, root, [], None, h3)
        self.assertEqual(s.root_flagged, 1)             # one of them met the remembered situation: searched again
        self.assertEqual(s.recalls, 1)
        self.assertNotEqual(chosen, h3)                 # read further, h3 loses the rook
        # no remembered situation among the best moves: the search's own choice, unchecked
        s.root_flagged = 0
        self.assertEqual(s.check(board, [(kd1, 0.0, None), (ra2, 0.0, None)], [], None, kd1), kd1)
        self.assertEqual(s.root_flagged, 0)

if __name__ == '__main__':
    unittest.main()
