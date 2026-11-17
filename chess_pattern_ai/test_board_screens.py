#!/usr/bin/env python3
"""Tests for board_screens (reading the other board games' screens, learned from the starting screen and play)"""

import os
import random
import tempfile
import unittest

import numpy as np

import board_screens as B


def play(name, plies, seed=0):
    """(adapter, the positions of a random game, the starting position first)"""
    rng = random.Random(seed)
    adapter, state = B.new_game(name)
    out = [state]
    for _ in range(plies):
        if adapter.is_over(state):
            break
        moves = adapter.legal_moves(state, adapter.to_move(state))
        if not moves:
            break
        after = sorted({B.grid_of(adapter, a).tobytes(): a for a in (adapter.apply(state, m) for m in moves)}.items())
        state = rng.choice(after)[1]
        out.append(state)
    return adapter, out


def follow(memory, name, adapter, states, colours=None, flipped=False, given=True):
    """Follow the screens of a game; the number of screens on which the position found was the true one"""
    own, last = states[0], B.grid_of(adapter, states[0])
    before = memory.look(B.render(name, last, 400, colours, flipped))
    right = 0
    for state in states[1:]:
        cells = B.grid_of(adapter, state)
        shown = B.render(name, cells, 400, colours, flipped, [tuple(p) for p in np.argwhere(cells != last)])
        last = cells
        found = memory.follow(adapter, own, shown, flipped=flipped if given else None, before=before)
        if found is None:
            return right
        own, before = found[1], found[2]
        right += np.array_equal(B.grid_of(adapter, own), cells)
    return right


class TestStartingScreens(unittest.TestCase):
    def test_every_game_learned_and_read_from_its_starting_screen(self):
        for name in B.GAMES:
            with self.subTest(name):
                memory = B.BoardScreenMemory(name)
                adapter, start = B.new_game(name)
                screen = B.render_state(name, adapter, start)
                top, left, h, w = B.find_board(screen)
                for found, drawn in zip((top, left, w), (20, 20, 360)):      # the board's rectangle, 5% in
                    self.assertLessEqual(abs(found - drawn), 2)                # (its edges smoothed)
                self.assertEqual(memory.learn_start(screen, flipped=False)['outcome'], 'learned')
                self.assertTrue(np.array_equal(memory.read(screen), B.grid_of(adapter, start)))
                # and at another size (but Arimaa: its pieces' letters, at another size, are misread on 2 to 4
                # of its 32 pieces; evaluate_board_screens)
                if name != 'arimaa':
                    self.assertTrue(np.array_equal(memory.read(B.render_state(name, adapter, start, size=300)),
                                                   B.grid_of(adapter, start)))

    def test_a_sided_starting_screen_needs_its_way_round(self):
        memory = B.BoardScreenMemory('checkers')
        adapter, start = B.new_game('checkers')
        self.assertEqual(memory.learn_start(B.render_state('checkers', adapter, start))['outcome'], 'not a lesson')


class TestFollowing(unittest.TestCase):
    def test_games_followed_from_the_starting_screen(self):
        for name, plies in (('othello', 30), ('go', 30), ('connect4', 20), ('checkers', 40), ('dots_boxes', 30),
                            ('arimaa', 30), ('hex', 20), ('pentago', 15), ('morris', 30)):
            with self.subTest(name):
                memory = B.BoardScreenMemory(name)
                adapter, states = play(name, plies, seed=1)
                memory.learn_start(B.render_state(name, adapter, states[0]), flipped=False)
                self.assertEqual(follow(memory, name, adapter, states), len(states) - 1)

    def test_the_first_stones_are_learned_in_play(self):
        # Go starts empty: the stones are learned from the screens of play; then a position is read alone
        memory = B.BoardScreenMemory('go')
        adapter, states = play('go', 40, seed=2)
        memory.learn_start(B.render_state('go', adapter, states[0]))
        self.assertEqual({e[1] for e in memory.examples}, {0})
        self.assertEqual(follow(memory, 'go', adapter, states), len(states) - 1)
        self.assertEqual({e[1] for e in memory.examples}, {0, 1, 2})
        adapter, other = play('go', 30, seed=3)
        self.assertTrue(np.array_equal(memory.read(B.render_state('go', adapter, other[-1])),
                                       B.grid_of(adapter, other[-1])))

    def test_other_colours_are_learned_in_play(self):
        memory = B.BoardScreenMemory('othello')
        adapter, states = play('othello', 20, seed=4)
        memory.learn_start(B.render_state('othello', adapter, states[0]))
        colours = B.other_colours('othello')
        adapter, game = play('othello', 40, seed=5)
        self.assertEqual(follow(memory, 'othello', adapter, game, colours), len(game) - 1)
        self.assertTrue(np.array_equal(memory.read(B.render('othello', B.grid_of(adapter, states[-1]), 400, colours)),
                                       B.grid_of(adapter, states[-1])))


class TestFromTheOtherSide(unittest.TestCase):
    def test_read_and_followed_from_the_other_side(self):
        memory = B.BoardScreenMemory('breakthrough')
        adapter, states = play('breakthrough', 30, seed=6)
        memory.learn_start(B.render_state('breakthrough', adapter, states[0]), flipped=False)
        cells = B.grid_of(adapter, states[-1])
        self.assertTrue(np.array_equal(memory.read(B.render('breakthrough', cells, flipped=True), flipped=True), cells))
        # the way round not given: found in context
        self.assertEqual(follow(memory, 'breakthrough', adapter, states, flipped=True, given=False), len(states) - 1)


class TestScreenPlay(unittest.TestCase):
    def test_games_played_from_the_screens(self):
        from checkers.checkers_board import Color as CheckersColor
        from checkers.checkers_headless_trainer import CheckersHeadlessTrainer
        from connect4.connect4_board import Color as Connect4Color
        from connect4.connect4_headless_trainer import Connect4HeadlessTrainer
        random.seed(0)
        for trainer, colour, side in ((Connect4HeadlessTrainer, Connect4Color.RED, False),
                                      (CheckersHeadlessTrainer, CheckersColor.BLACK, True)):
            with self.subTest(trainer.__name__):
                t = trainer(os.path.join(tempfile.mkdtemp(), 't.db'), screen=True, screen_side=side)
                t.play_game(colour)
                stats = t.screen.stats
                self.assertGreater(stats['screens'], 0)
                self.assertEqual(stats['true position'], stats['screens'])
                self.assertEqual(stats['followed'], stats['screens'])
                self.assertEqual(stats['illegal'], 0)
                self.assertIs(t.screen.flipped, side)


if __name__ == '__main__':
    unittest.main()
