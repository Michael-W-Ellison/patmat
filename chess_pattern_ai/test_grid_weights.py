#!/usr/bin/env python3
"""Tests for grid_weights (learned weights of game-independent position features in the board games)"""

import os
import tempfile
import unittest

import grid_weights as G
from go.go_adapter import GoAdapter
from go.go_board import Color
from go.go_game import GoGame


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.adapter = GoAdapter()
        self.game = GoGame(board_size=9)

    def test_the_empty_board_is_even(self):
        self.assertEqual(G.features(self.adapter, self.game, Color.BLACK), [0] * len(G.FEATURES))

    def test_a_stone_in_the_centre(self):
        moves = self.adapter.legal_moves(self.game, Color.BLACK)
        centre = next(m for m in moves if m.position == (4, 4))
        after = self.adapter.apply(self.game, centre)
        f = dict(zip(G.FEATURES, G.features(self.adapter, after, Color.BLACK)))
        self.assertEqual(f['pieces'], 1)
        self.assertEqual(f['largest_group'], 1)
        self.assertEqual(f['groups'], 1)
        self.assertEqual(f['longest_line'], 1)
        self.assertEqual(f['edge'], 0)
        self.assertEqual(f['liberties'], 4)
        self.assertEqual(f['centre'], 1)
        # the other side sees the opposite
        self.assertEqual(G.features(self.adapter, after, Color.WHITE),
                         [-x for x in G.features(self.adapter, after, Color.BLACK)])


class TestWeights(unittest.TestCase):
    def test_learning_and_the_value_of_a_move(self):
        path = os.path.join(tempfile.mkdtemp(), 't.db')
        adapter = GoAdapter()
        w = G.GridWeights(path, 'go', adapter, seed=0)
        self.assertEqual(w.tuner.mean(), [0.0] * len(G.FEATURES))
        w.current = [1.0] + [0.0] * (len(G.FEATURES) - 1)        # one point per stone more than the opponent
        game = GoGame(board_size=9)
        move = adapter.legal_moves(game, Color.BLACK)[0]
        self.assertEqual(w.value(game, move, Color.BLACK), 1.0)
        from evaluation_weights import CE_POPULATION
        for i in range(CE_POPULATION):
            w.end_game('win' if i % 2 else 'loss', 10.0 * i)
        self.assertEqual(w.tuner.generations, 1)
        w.close()
        again = G.GridWeights(path, 'go', adapter, tune=False)
        self.assertEqual(again.current, again.tuner.mean())        # not tuning: the learned mean
        self.assertNotEqual(again.current, [0.0] * len(G.FEATURES))

    def test_game_score(self):
        self.assertGreater(G.game_score('loss', 50), G.game_score('loss', -50))
        self.assertGreater(G.game_score('draw', -1e9), G.game_score('loss', 1e9))
        self.assertGreater(G.game_score('win', -1e9), G.game_score('draw', 1e9))


if __name__ == '__main__':
    unittest.main()
