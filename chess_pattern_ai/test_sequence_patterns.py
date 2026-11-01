#!/usr/bin/env python3
"""
Unit tests for multi-move (sequence) patterns, learned piece categories and
the pattern-guided search.
"""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

if CHESS_AVAILABLE:
    from piece_knowledge import PieceKnowledge
    from sequence_patterns import SequencePatternLearner, MATE_VALUE
    from pattern_search import PatternSearch, forcing, threatened
    from position_patterns import PositionPatternLearner


def moves_of(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def watch_random_games(pieces, games=20, seed=0):
    """Let pieces observe the moves of random games"""
    rng = random.Random(seed)
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 200:
            move = rng.choice(list(board.legal_moves))
            pieces.observe(board, move)
            board.push(move)


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestPieceKnowledge(unittest.TestCase):

    def test_categories_are_learned_from_moves_watched(self):
        pieces = PieceKnowledge(':memory:')
        self.assertEqual(pieces.category(chess.KNIGHT), 'u')        # nothing watched yet
        watch_random_games(pieces)
        learned = {p: pieces.category(p) for p in chess.PIECE_TYPES}
        self.assertEqual(learned, {chess.PAWN: 'o:dSf', chess.KNIGHT: 'jS', chess.BISHOP: 'dL',
                                   chess.ROOK: 'oL', chess.QUEEN: 'doL', chess.KING: 'doS'})
        pieces.close()


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestSequencePatterns(unittest.TestCase):

    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(handle)
        self.learner = SequencePatternLearner(self.db_path)
        watch_random_games(self.learner.pieces)

    def tearDown(self):
        self.learner.close()
        os.unlink(self.db_path)

    def test_token_describes_a_fork(self):
        # the knight checks the king and attacks the rook (worth more than the knight)
        board = chess.Board('3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')
        self.assertEqual(self.learner.token(board, chess.Move.from_uci('e5f7')), 'jS -+ a2> s')
        self.assertEqual(board.fen(), '3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')     # unchanged

    def test_token_marks_a_piece_a_cheaper_piece_attacks(self):
        # the queen takes a pawn that the c6 pawn defends
        board = chess.Board('4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1')
        self.assertEqual(self.learner.token(board, chess.Move.from_uci('d1d5')), 'doL xo:dSf a1 c')

    def test_mating_sequence_is_learned_for_both_sides(self):
        game = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#'])
        self.learner.record_game(list(game.move_stack), chess.WHITE, 'win')

        # White's mating move, after Black's knight move: a large gain for White
        before_mate = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6'])
        mate = chess.Move.from_uci('h5f7')
        self.assertGreater(self.learner.value(before_mate, mate), MATE_VALUE / 10)

        # Black's Nf6, which allowed the mate within the horizon: a large loss for Black
        before_nf6 = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4'])
        self.assertLess(self.learner.value(before_nf6, chess.Move.from_uci('g8f6')), -MATE_VALUE / 10)

        # Multi-move patterns are recorded (Bc4, Nf6, Qxf7# as one three-ply pattern)
        last = self.learner.token(before_mate, mate)
        self.assertTrue([p for p in self.learner.stats if p.count(' | ') == 2 and p.endswith(last)])

    def test_statistics_persist(self):
        game = moves_of(['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#'])
        self.learner.record_game(list(game.move_stack), chess.WHITE, 'win')
        count = len(self.learner.stats)
        self.learner.close()
        self.learner = SequencePatternLearner(self.db_path)
        self.assertEqual(len(self.learner.stats), count)
        self.assertIn('ai', self.learner.baseline)
        self.assertEqual(self.learner.pieces.category(chess.KNIGHT), 'jS')


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestPositionPatterns(unittest.TestCase):

    def setUp(self):
        self.pieces = PieceKnowledge(':memory:')
        watch_random_games(self.pieces)
        self.positions = PositionPatternLearner(':memory:', self.pieces)

    def tearDown(self):
        self.positions.close()
        self.pieces.close()

    def test_features_describe_each_piece_from_the_side_to_move(self):
        # Black to move: White's queen on g4 is attacked by the cheaper c8 bishop
        board = moves_of(['e4', 'd5', 'Qg4'])
        feats = self.positions.features(board)
        self.assertEqual(feats['t doL c a2'], 1)        # their queen, attacked by a cheaper piece, attacks d7 and g7
        self.assertNotIn('m o:dSf s a0', feats)         # pieces neither attacked nor attacking show no pattern
        self.assertEqual(feats['=m o:dSf'], 1)          # what material covers: the kinds of piece each side has,
        self.assertEqual(feats['=m o:dSf 2+'], 1)       # (two or more of them)
        self.assertEqual(feats['=material'], 0.0)       # and the material balance

    def test_value_leaves_out_what_material_covers(self):
        board = moves_of(['e4', 'd5', 'Qg4'])
        self.positions.weights['=t doL'] = [-25.0, 1.0]       # "they have a queen": a control
        self.positions.weights['=material'] = [5.0, 1.0]
        self.assertEqual(self.positions.value(board), 0.0)
        self.positions.cache.clear()
        self.positions.weights['t doL c a2'] = [25.0, 1.0]    # their queen attacked by a cheaper piece: a pattern
        self.assertEqual(self.positions.value(board), 25.0)

    def test_learns_from_the_opponents_moves(self):
        # The agent (White) leaves its queen where the bishop takes it: the opponent's capture is learned
        before_capture = moves_of(['e4', 'd5', 'Qg4'])
        queen_safe = moves_of(['e4', 'd5', 'Qf3'])              # never played
        self.assertEqual(self.positions.value(before_capture), 0.0)
        for _ in range(30):
            game = moves_of(['e4', 'd5', 'Qg4', 'Bxg4'])
            self.positions.record_game(list(game.move_stack), chess.WHITE, 'loss')
        # Black, to move, is expected to gain more where a cheaper piece attacks White's queen
        self.assertGreater(self.positions.value(before_capture), self.positions.value(queen_safe) + 10.0)
        self.assertGreater(self.positions.weights['t doL c a2'][0], 0.0)

    def test_search_adds_the_learned_value_to_material(self):
        sequences = SequencePatternLearner(':memory:')
        board = moves_of(['e4', 'd5', 'Qg4'])
        self.positions.weights['t doL c a2'] = [25.0, 1.0]
        search = PatternSearch(sequences, positions=self.positions)
        self.assertEqual(search.evaluate(board), 25.0)
        self.assertEqual(PatternSearch(sequences).evaluate(board), 0.0)
        sequences.close()


@unittest.skipUnless(CHESS_AVAILABLE, 'python-chess not installed')
class TestPatternSearch(unittest.TestCase):

    def setUp(self):
        self.learner = SequencePatternLearner(':memory:')
        watch_random_games(self.learner.pieces)

    def tearDown(self):
        self.learner.close()

    def test_finds_a_mate_in_one(self):
        board = chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')
        self.assertEqual(PatternSearch(self.learner, seconds=1.0).choose(board), chess.Move.from_uci('a1a8'))

    def test_choice_matches_a_full_window_search(self):
        # Moves the root search cuts off must never be chosen over the best move (they come back with a value
        # exactly at the cut-off, which adding a fractional pattern bonus could round above the best)
        import pattern_search

        class FractionalPatterns:
            def context(self, board, plies=2):
                return []

            def token(self, board, move):
                return ''

            def value(self, board, move, context=None, token=None):
                return ((move.from_square * 7 + move.to_square * 13) % 10) * 0.37

        fens = ['r1q1kbnr/p1p1pppp/b1n5/1p1N4/3P4/5P2/PPPPB1PP/1RBQK1NR w Kkq - 0 8',
                'rnb1kb2/p3n2r/1p1pqpp1/2p1p2p/2PPPB2/P7/1P1K1PPP/RN2QBNR w - - 0 16',
                '1r3b1r/2P1p1pp/np6/pQ1b1pN1/3k3P/NP1P2P1/P3Pn2/2R1KB1R b K - 2 20']
        saved = pattern_search.MAX_DEPTH
        pattern_search.MAX_DEPTH = 2
        try:
            for fen in fens:
                board = chess.Board(fen)
                search = PatternSearch(FractionalPatterns(), seconds=1000.0)
                chosen = search.choose(board)
                totals = {}
                for move, pattern, tok in search.ordered(board, []):
                    board.push(move)
                    totals[move] = -search.negamax(board, 1, -float('inf'), float('inf'), [], 1) \
                        + pattern_search.PATTERN_WEIGHT * pattern
                    board.pop()
                self.assertAlmostEqual(totals[chosen], max(totals.values()), places=9, msg=fen)
        finally:
            pattern_search.MAX_DEPTH = saved

    def test_forcing_moves(self):
        fork = chess.Board('3k3r/8/8/4N3/8/8/8/4K3 w - - 0 1')
        self.assertTrue(forcing(chess.Move.from_uci('e5f7'), 'jS -+ a2> s', fork))           # check
        self.assertFalse(forcing(chess.Move.from_uci('e5c4'), 'jS - a0 s', fork))            # quiet
        greedy = chess.Board('4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1')
        self.assertFalse(forcing(chess.Move.from_uci('d1d5'), 'doL xo:dSf a1 c', greedy))    # gives up the queen
        self.assertTrue(forcing(chess.Move.from_uci('d1d5'), 'doL xo:dSf a1 s', greedy))     # (if it were safe)
        self.assertFalse(forcing(chess.Move.from_uci('e5f7'), 'jS - a2> h', fork))           # threat, piece lost

    def test_selective_search_finds_the_fork_and_the_mate(self):
        board = chess.Board('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1')
        search = PatternSearch(self.learner, seconds=30.0, selective=True, max_depth=3)
        self.assertEqual(search.choose(board), chess.Move.from_uci('b5c7'))
        board = chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')
        self.assertEqual(search.choose(board), chess.Move.from_uci('a1a8'))

    def test_threats(self):
        # the knight on e6 forks the queen and the rook, which defend each other
        board = chess.Board('3q1r2/7k/4N3/8/8/8/8/4K3 b - - 0 1')
        self.assertEqual(threatened(board, chess.BLACK), [(chess.D8, 58.0), (chess.F8, 18.0)])
        self.assertEqual(threatened(board, chess.WHITE), [])
        # where the search stops, the side to move can answer only one threat: the second costs its stake
        values = []
        for threats in (False, True):
            search = PatternSearch(self.learner, threats=threats)
            search.deadline = float('inf')
            values.append(search.quiesce(board, -float('inf'), float('inf'), [], 0, 0))
        self.assertEqual(values[0] - values[1], 18.0)
        # the selective search considers an escape to safety for each threatened piece
        search = PatternSearch(self.learner, selective=True, threats=True)
        moves = [m for m, _, _ in search.candidates(board, [])]
        def safe(m):
            board.push(m)
            try:
                return all(sq != m.to_square for sq, _ in threatened(board, chess.BLACK))
            finally:
                board.pop()
        for square in (chess.D8, chess.F8):
            self.assertTrue(any(m.from_square == square and safe(m) for m in moves))

    def test_wins_material_two_moves_deep(self):
        # the knight fork Nc7+ wins the rook on a8 a move later
        board = chess.Board('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1')
        search = PatternSearch(self.learner, seconds=2.0)
        self.assertEqual(search.choose(board), chess.Move.from_uci('b5c7'))
        self.assertGreaterEqual(search.depth_reached, 3)


if __name__ == '__main__':
    unittest.main()
