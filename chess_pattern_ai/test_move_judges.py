#!/usr/bin/env python3
"""Tests for move_judges and the judged evidence of general_move_patterns"""

import os
import tempfile
import unittest

import chess

from general_move_patterns import GeneralPatternLearner
from move_judges import SearchJudge
from position_patterns import PositionPatternLearner


class Race:
    """Toy game: players alternately add 1 or 2 to a total; whoever reaches TARGET wins. State: (total, mover)"""
    TARGET = 7

    def legal_moves(self, state, color):
        return [1, 2] if not self.is_over(state) and state[1] == color else []

    def apply(self, state, move):
        return (state[0] + move, 1 - state[1])

    def advantage(self, state, color):
        return 0.0

    def is_over(self, state):
        return state[0] >= self.TARGET

    def winner(self, state):
        return (1 - state[1]) if self.is_over(state) else None

    def to_move(self, state):
        return state[1]

    def opponent(self, color):
        return 1 - color


class TestSearchJudge(unittest.TestCase):
    def test_losing_move_has_a_loss_and_the_winning_move_none(self):
        judge = SearchJudge(Race(), depth=4)
        # at 3, adding 1 leaves 4, from which every move lets the mover reach 7; adding 2 leaves 5: the opponent
        # adds 2 and wins
        self.assertEqual(judge.move_loss((3, 0), 0, 1), 0.0)
        self.assertGreater(judge.move_loss((3, 0), 0, 2), 0.0)

    def test_no_rating_where_every_move_loses(self):
        judge = SearchJudge(Race(), depth=4)
        # at 4 every move leaves the opponent a win
        self.assertIsNone(judge.move_loss((4, 0), 0, 1))

    def test_node_limit_falls_back_to_a_shallower_search(self):
        judge = SearchJudge(Race(), depth=6, max_nodes=1)
        # only the first ply completes; neither move ends the game there, so no loss is seen
        self.assertEqual(judge.move_loss((0, 0), 0, 2), 0.0)


@unittest.skipUnless(os.path.exists('/usr/games/stockfish'), 'needs Stockfish')
class TestStockfishJudge(unittest.TestCase):
    def test_no_rating_when_being_mated(self):
        from move_judges import StockfishJudge
        judge = StockfishJudge(depth=8)
        try:
            # Black's only move is Kb8, then Rh8 mates
            board = chess.Board('k7/8/1K6/8/8/8/8/7R b - - 0 1')
            self.assertEqual(judge.value(board, chess.BLACK), -1000)
            self.assertIsNone(judge.move_loss(board, chess.BLACK, next(iter(board.legal_moves))))
        finally:
            judge.close()


class FixedJudge:
    """Judge with given losses per move"""

    def __init__(self, losses):
        self.losses = losses

    def move_loss(self, state, color, move):
        return self.losses[move]


class TestJudgedEvidence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, 'u.db')

    def test_good_moves_rise_and_bad_moves_fall(self):
        learner = GeneralPatternLearner(self.db, 'race')
        learner.judge = FixedJudge({'good': 0.0, 'bad': 10.0})
        for _ in range(20):
            learner.note_move(None, None, 0, 'good', 'a|b|mob=x|d')
            learner.note_move(None, None, 0, 'bad', 'a|b|mob=y|d')
            learner.record_game(['a|b|mob=x|d', 'a|b|mob=y|d'], 'draw')
        self.assertGreater(learner.value('a|b|mob=x|d'), 2.0)
        self.assertLess(learner.value('a|b|mob=y|d'), -2.0)
        self.assertGreater(learner.part_value('mob=x'), learner.part_value('mob=y'))
        self.assertEqual(learner.pending, [])
        learner.close()

    def test_judged_evidence_of_another_game_is_imported(self):
        a = GeneralPatternLearner(self.db, 'one')
        a.judge = FixedJudge({'good': 0.0, 'bad': 10.0})
        for _ in range(20):
            a.note_move(None, None, 0, 'good', 'g|x|x|x')
            a.note_move(None, None, 0, 'bad', 'b|x|x|x')
            a.record_game(['g|x|x|x', 'b|x|x|x'], 'draw')
        a.close()
        b = GeneralPatternLearner(self.db, 'two')
        self.assertGreater(b.value('g|x|x|x'), b.value('b|x|x|x'))
        b.close()
        c = GeneralPatternLearner(self.db, 'two', import_cross_game=False)
        self.assertEqual(c.value('g|x|x|x'), 0.0)
        c.close()

    def test_judge_depth_makes_a_search_judge(self):
        GeneralPatternLearner.JUDGE_DEPTH = 4
        try:
            learner = GeneralPatternLearner(self.db, 'race')
            learner.note_move(Race(), (3, 0), 0, 2, 'f|f|f|f')
            self.assertIsInstance(learner.judge, SearchJudge)
            self.assertGreater(learner.last_loss, 0.0)
            learner.close()
        finally:
            GeneralPatternLearner.JUDGE_DEPTH = 0


class ConstantJudge:
    """Chess judge that rates every position +3 pawns for White"""

    def evaluate(self, board):
        return 300


class TestJudgedPositions(unittest.TestCase):
    def test_positions_learn_the_judges_evaluation_beyond_material(self):
        db = os.path.join(tempfile.mkdtemp(), 'c.db')
        learner = PositionPatternLearner(db)
        moves = [chess.Move.from_uci(u) for u in ('e2e4', 'e7e5', 'g1f3', 'b8c6')]
        for _ in range(30):
            learner.record_game(moves, chess.WHITE, 'draw', judge=ConstantJudge())
        # positions with White to move are worth +30 for the mover, with Black to move -30: the bias, fitted to
        # both, stays near 0, and the material control is 0 (equal material); what differs is learned by patterns
        board = chess.Board()
        for m in moves[:2]:
            board.push(m)
        black = board.copy()
        black.push(moves[2])
        self.assertGreater(learner.value(board), learner.value(black))
        learner.close()


if __name__ == '__main__':
    unittest.main()
