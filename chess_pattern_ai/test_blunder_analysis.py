#!/usr/bin/env python3
"""Unit tests for the blunder classifier (blunder_analysis)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

if CHESS_AVAILABLE:
    from blunder_analysis import first_lost, loss_ply, motif


def m(uci):
    return chess.Move.from_uci(uci)


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestMotif(unittest.TestCase):

    def check(self, fen, move, expected):
        board = chess.Board(fen)
        self.assertEqual(motif(board, m(move)), expected)
        self.assertEqual(board.fen(), fen)      # unchanged

    def test_mate(self):
        self.check('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'mate in 1')

    def test_takes_undefended_piece(self):
        self.check('4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1', 'd1d5', 'takes undefended piece')

    def test_takes_with_cheaper_piece(self):
        self.check('4k3/5p2/4q3/3P4/8/8/8/4K3 w - - 0 1', 'd5e6', 'takes with cheaper piece')

    def test_exchange(self):
        self.check('4k3/8/2p5/3n4/8/2N5/8/4K3 w - - 0 1', 'c3d5', 'exchange')

    def test_attacks_a_piece(self):
        self.check('4k3/8/8/3n4/8/8/8/3NK3 w - - 0 1', 'd1c3', 'attacks a piece')     # undefended knight
        self.check('4k3/8/2p5/3n4/8/8/8/3NK3 w - - 0 1', 'd1c3', 'quiet move')        # defended, same worth

    def test_fork(self):
        self.check('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1', 'b5c7', 'fork')

    def test_pin(self):
        # the bishop pins the knight to the queen behind it
        self.check('4k3/8/3q4/8/1n6/8/8/4KB2 w - - 0 1', 'f1a6', 'quiet move')
        self.check('4k3/7q/8/8/4n3/8/8/1B2K3 w - - 0 1', 'b1c2', 'pin')

    def test_skewer(self):
        # the rook checks the king, with the undefended rook behind it on the same file
        self.check('4r3/8/8/8/4k3/8/8/K6R w - - 0 1', 'h1e1', 'skewer')

    def test_discovered_attack(self):
        # the knight moves off the file, and the rook now attacks the queen
        self.check('3qk3/8/8/8/8/8/3N4/3RK3 w - - 0 1', 'd2f1', 'discovered attack')

    def test_check(self):
        self.check('4k3/8/8/8/8/8/8/R3K3 w - - 0 1', 'a1a8', 'check')


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestLine(unittest.TestCase):

    def test_loss_ply(self):
        self.assertEqual(loss_ply([0, 0, -32, -32, -32]), 2)
        self.assertEqual(loss_ply([0, -32, 0, 0]), None)        # won back: no lasting loss
        self.assertEqual(loss_ply([0, 0]), None)

    def test_first_lost(self):
        # Black's knight went to d5, where White's rook takes it
        board = chess.Board('4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1')
        self.assertEqual(first_lost(board, [m('d1d5')], chess.BLACK, chess.D5), 'the piece moved')
        self.assertEqual(first_lost(board, [m('d1d5')], chess.BLACK, chess.E8), 'another piece')


if __name__ == '__main__':
    unittest.main()
