#!/usr/bin/env python3
"""Unit tests for Minesweeper and the single-player / hidden-information adapter interface
(run: python3 -m unittest test_minesweeper)"""

import os
import random
import tempfile
import unittest

import general_move_patterns as gmp
from minesweeper.minesweeper_adapter import NATURE, PLAYER, MinesweeperAdapter
from minesweeper.minesweeper_game import HIDDEN, MinesweeperGame
from minesweeper.constraint_learner import ConstraintLearner
from minesweeper.minesweeper_headless_trainer import cell_features, play_game, train


class TestRules(unittest.TestCase):
    def test_first_move_is_safe_and_opens_a_region(self):
        for seed in range(20):
            g = MinesweeperGame(9, 9, 10, seed)
            g.reveal((4, 4))
            self.assertFalse(g.lost())
            self.assertEqual(g.count((4, 4)), 0)
            self.assertGreaterEqual(len(g.revealed), 9)

    def test_numbers_and_flood_fill(self):
        g = MinesweeperGame(3, 4, 1, seed=1)
        g.reveal((0, 0))
        (mine,) = g.mines
        self.assertNotIn(mine, {(0, 0), *g.neighbours((0, 0))})     # the first cell's neighbourhood is clear
        for r in range(3):
            for c in range(4):
                if (r, c) in g.revealed:
                    self.assertEqual(g.count((r, c)), sum(1 for n in g.neighbours((r, c)) if n == mine))

    def test_win_and_loss(self):
        g = MinesweeperGame(3, 4, 1, seed=1)
        g.reveal((0, 0))
        (mine,) = g.mines
        lost = g.copy()
        lost.reveal(mine)
        self.assertTrue(lost.lost() and lost.is_over() and not lost.won())
        for pos in g.legal_moves():
            if pos != mine and not g.is_over():
                g.reveal(pos)
        self.assertTrue(g.won())
        self.assertEqual(g.legal_moves(), [])

    def test_seed_repeats_a_game(self):
        a, b = MinesweeperGame(seed=7), MinesweeperGame(seed=7)
        a.reveal((0, 0))
        b.reveal((0, 0))
        self.assertEqual(a.mines, b.mines)


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MinesweeperAdapter()

    def test_single_player_interface(self):
        ad = self.adapter
        self.assertEqual(ad.players, 1)
        g = ad.new_game(seed=3)
        self.assertEqual(ad.to_move(g), PLAYER)
        self.assertEqual(ad.opponent(PLAYER), NATURE)
        after = ad.apply(g, (4, 4))
        self.assertEqual(len(g.revealed), 0)                  # apply leaves the given state as it was
        self.assertGreater(ad.advantage(after, PLAYER), 0)
        g = after
        mine = next(iter(g.mines))
        lost = ad.apply(g, mine)
        self.assertEqual(ad.winner(lost), NATURE)
        self.assertEqual(gmp.evaluate(ad, lost, PLAYER), -gmp.WIN_VALUE)

    def test_observation_hides_mines(self):
        ad = self.adapter
        g = ad.apply(ad.new_game(seed=5), (0, 0))
        obs = ad.observe(g, PLAYER)
        for r, c in g.mines:
            self.assertEqual(obs[r][c], HIDDEN)
        grid = ad.to_grid(g)
        self.assertEqual(grid.shape, (9, 9))

    def test_lookahead_is_refused(self):
        ad = self.adapter
        g = ad.apply(ad.new_game(seed=5), (0, 0))
        with self.assertRaises(ValueError):
            gmp.analyze_candidates(ad, g, PLAYER, ad.legal_moves(g, PLAYER)[:3])
        with self.assertRaises(ValueError):
            gmp.winning_moves(ad, g, PLAYER, ad.legal_moves(g, PLAYER))


class TestTrainer(unittest.TestCase):
    def test_features_use_only_the_observation(self):
        obs = [[HIDDEN, 1, 0], [HIDDEN, 2, 0], [HIDDEN, HIDDEN, HIDDEN]]
        self.assertEqual(cell_features(obs, (0, 0)), 'rev=1-2|min=1|max=2|pos=corner')
        self.assertEqual(cell_features(obs, (2, 2)), 'rev=1-2|min=0|max=2|pos=corner')

    def test_training_runs_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, 'ms.db')
            rates = train(20, 'learner', db, seed=1, block=10, verbose=False)
            self.assertEqual(len(rates), 2)
            import sqlite3
            conn = sqlite3.connect(db)
            n, games = conn.execute(
                "SELECT COUNT(*), (SELECT games_recorded FROM win_lift_baseline) FROM minesweeper_patterns").fetchone()
            conn.close()
            self.assertGreater(n, 0)
            self.assertEqual(games, 20)
        self.assertEqual(len(train(10, 'random', seed=1, block=10, verbose=False)), 1)


class TestConstraintLearner(unittest.TestCase):
    def test_discovers_the_counting_constraint(self):
        ad = MinesweeperAdapter(6, 6, 4)
        learner = ConstraintLearner(random.Random(0), accept_after=50)
        for i in range(10):
            play_game(ad, i, learner.choose, learner.end_game)
        # a revealed cell is never a mine, so counting its 3x3 block (itself included) is the
        # same rule; nothing in Minesweeper tells the two apart
        self.assertEqual(set(learner.accepted()), {('count', '8 neighbours', 'mine'), ('count', '3x3 block', 'mine')})

    def test_deduction(self):
        learner = ConstraintLearner(random.Random(0), accept_after=0)
        learner.hypotheses = {('count', '8 neighbours', 'mine'): {'confirmed': 1, 'violated': 0}}
        H = HIDDEN
        # a 1 touching a single hidden cell makes it a mine; the 1 next to it is then satisfied,
        # so its other hidden neighbour is safe
        obs = [[0, 1, H],
               [0, 1, H],
               [0, 1, 1]]
        safe, mines = learner.deduce(obs)
        self.assertEqual(mines, {(1, 2)})
        self.assertEqual(safe, {(0, 2)})
        # nested equations: {a, b} and {b, c} each hold 1 mine, {a, b, c} holds 1, so a and c are safe
        obs = [[H, H, H],
               [1, 1, 1],
               [0, 0, 0]]
        safe, mines = learner.deduce(obs)
        self.assertEqual((safe, mines), ({(0, 0), (0, 2)}, {(0, 1)}))
        # 1-2-1: {a, b, c} hold 2 and {a, b}, {b, c} hold 1 each, so a and c are mines and b is safe
        obs = [[H, H, H],
               [1, 2, 1],
               [0, 0, 0]]
        safe, mines = learner.deduce(obs)
        self.assertEqual((safe, mines), ({(0, 1)}, {(0, 0), (0, 2)}))

    def test_beats_random_on_the_standard_board(self):
        ad = MinesweeperAdapter()
        learner = ConstraintLearner(random.Random(0))
        wins = sum(play_game(ad, 1000 + i, learner.choose, learner.end_game) for i in range(100))
        self.assertGreater(wins, 50)
        self.assertGreater(learner.moves_by['deduced'], learner.moves_by['guessed'])


if __name__ == '__main__':
    unittest.main()
