#!/usr/bin/env python3
"""Tests for screen_reading (reading chess screens, learned from the starting screen and from play)"""

import random
import unittest

import chess

import screen_reading as S


def random_position(seed, plies=30):
    rng = random.Random(seed)
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return b


class TestReading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = S.ScreenMemory()
        cls.memory.learn_start(S.render(chess.Board(), 400))

    def test_the_starting_screen_teaches_every_piece(self):
        self.assertEqual({p for _, p, _ in self.memory.examples}, set(S.PIECES))

    def test_other_positions_sizes_and_colours(self):
        for seed, kw in enumerate([{}, {'size': 280}, {'colors': {'square light': '#dee3e6', 'square dark': '#8ca2ad'}}]):
            b = random_position(seed)
            kw = dict(kw)
            size = kw.pop('size', 400)
            self.assertEqual(self.memory.read(S.render(b, size, lastmove=b.peek(), **kw)), S.placement(b))

    def test_board_fen(self):
        b = random_position(5)
        self.assertEqual(S.board_fen(S.placement(b)), b.board_fen())


class TestFollowing(unittest.TestCase):
    def test_the_move_made_is_found_from_the_screen(self):
        memory = S.ScreenMemory()
        memory.learn_start(S.render(chess.Board(), 400))
        b = random_position(3, 20)
        known = b.copy()
        move = random.Random(1).choice(list(b.legal_moves))
        b.push(move)
        self.assertEqual(memory.follow(known, S.render(b, 400, lastmove=move)), [move])

    def test_a_new_style_is_learned_in_play(self):
        # a board of colours and size not seen: each screen of a game followed in context, the squares misread
        # learned; at the end the screen reads right
        memory = S.ScreenMemory()
        memory.learn_start(S.render(chess.Board(), 400))
        before = len(memory.examples)
        colours = {'square light': '#ffffff', 'square dark': '#999999'}
        b, known = chess.Board(), chess.Board()
        rng = random.Random(2)
        for _ in range(12):
            move = rng.choice(list(b.legal_moves))
            b.push(move)
            found = memory.follow(known, S.render(b, 300, colors=colours, lastmove=move))
            self.assertEqual(found, [move])
            known.push(found[0])
        self.assertGreaterEqual(len(memory.examples), before)
        self.assertEqual(memory.read(S.render(b, 300, colors=colours)), S.placement(b))


class TestScreenPlay(unittest.TestCase):
    def test_a_game_played_from_the_screen(self):
        import os
        import tempfile
        from headless_trainer import HeadlessTrainer
        random.seed(0)
        t = HeadlessTrainer(os.path.join(tempfile.mkdtemp(), 's.db'), screen=True)
        try:
            t.play_game(chess.WHITE)
            self.assertGreater(t.screen_stats['turns'], 0)
            self.assertEqual(t.screen_stats['true board'], t.screen_stats['turns'])
            self.assertEqual(t.screen_stats['illegal'], 0)
        finally:
            t.close()


if __name__ == '__main__':
    unittest.main()
