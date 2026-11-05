#!/usr/bin/env python3
"""Tests for move_history and its use in move_judges, general_move_patterns and pattern_search"""

import random
import unittest
from dataclasses import dataclass

import chess

from general_move_patterns import _sample_replies
from move_history import MoveHistory, move_key
from move_judges import SearchJudge


@dataclass
class Unhashable:
    square: tuple
    __hash__ = None


class TestMoveHistory(unittest.TestCase):
    def test_killers_history_and_decay(self):
        h = MoveHistory()
        h.record(2, 'w', 'a', 3)
        h.record(2, 'w', 'b', 1)
        h.record(2, 'w', 'c', 1)
        self.assertEqual(h.killers[2], ['c', 'b'])         # the last two cutoffs, newest first
        self.assertEqual(h.order(['x', 'a', 'b', 'c'], 2, 'w'), ['b', 'c', 'a', 'x'])
        self.assertEqual(h.order(['x', 'a', 'b', 'c'], 3, 'w'), ['a', 'b', 'c', 'x'])
        self.assertEqual(h.score('b', 'a'), 0.0)            # per side
        h.new_search()
        self.assertEqual(h.killers, {})
        self.assertEqual(h.score('w', 'a'), 4.5)

    def test_unhashable_moves_are_keyed_by_their_contents(self):
        h = MoveHistory()
        h.record(1, 0, Unhashable((1, 2)), 2)
        self.assertEqual(move_key(Unhashable((1, 2))), repr(Unhashable((1, 2))))
        self.assertTrue(h.is_killer(1, Unhashable((1, 2))))
        self.assertEqual(h.score(0, Unhashable((1, 2))), 4.0)


class TestJudgeOrdering(unittest.TestCase):
    def test_same_losses_with_fewer_nodes(self):
        from othello.othello_adapter import OthelloAdapter
        from othello.othello_game import OthelloGame
        adapter = OthelloAdapter()
        rng = random.Random(3)
        plain, ordered = SearchJudge(adapter, 4, max_nodes=10 ** 9), SearchJudge(adapter, 4, max_nodes=10 ** 9)
        ordered.move_history = MoveHistory()
        nodes = [0, 0]
        game = OthelloGame()
        for _ in range(12):
            color = adapter.to_move(game)
            moves = adapter.legal_moves(game, color)
            if not moves:
                break
            move = rng.choice(moves)
            self.assertAlmostEqual(plain.move_loss(game, color, move), ordered.move_loss(game, color, move))
            nodes[0] += plain.nodes
            nodes[1] += ordered.nodes
            game = adapter.apply(game, move)
        self.assertLess(nodes[1], nodes[0])


class TestReplySampling(unittest.TestCase):
    def test_killer_and_scored_replies_are_always_sampled(self):
        replies = list(range(100))
        h = MoveHistory()
        h.record(1, 'b', 57, 1)                             # a killer reply
        h.record(3, 'b', 81, 1)                             # scored, not a killer at ply 1
        for seed in range(20):
            sample = _sample_replies(replies, 'b', 30, random.Random(seed), h)
            self.assertEqual(len(sample), 30)
            self.assertEqual(len(set(sample)), 30)
            self.assertEqual(sample[:2], [57, 81])
        self.assertEqual(len(_sample_replies(replies, 'b', 30, random.Random(0), None)), 30)


class TestChessSearch(unittest.TestCase):
    def test_killers_are_searched_and_recorded(self):
        import os
        import tempfile
        from pattern_search import PatternSearch
        from sequence_patterns import SequencePatternLearner
        sequences = SequencePatternLearner(os.path.join(tempfile.mkdtemp(), 'c.db'))
        search = PatternSearch(sequences, seconds=5, selective=True, max_depth=3)
        search.move_history = MoveHistory()
        board = chess.Board()
        board.push_uci('e2e4')
        plain = [e[0] for e in search.candidates(board, [], 1)]
        killer = next(m for m in board.legal_moves if m not in plain)
        search.move_history.record(1, board.turn, killer, 2)
        self.assertIn(killer, [e[0] for e in search.candidates(board, [], 1)])
        self.assertNotIn(killer, [e[0] for e in search.candidates(board, [], 2)])
        self.assertIsNotNone(search.choose(chess.Board()))
        self.assertTrue(search.move_history.scores)
        sequences.close()


if __name__ == '__main__':
    unittest.main()
