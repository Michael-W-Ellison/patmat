#!/usr/bin/env python3
"""
Unit tests for core pattern recognition AI components.
"""

import unittest
import tempfile
import os
import sys
import sqlite3

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from learnable_move_prioritizer import LearnableMovePrioritizer

# Import checkers components
from checkers.checkers_board import CheckersBoard, Piece, Color, PieceType
from checkers.checkers_game import CheckersGame
from checkers.checkers_scorer import CheckersScorer


class TestLearnableMovePrioritizer(unittest.TestCase):
    """Test the core learning system"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.prioritizer = LearnableMovePrioritizer(self.db_path)

    def tearDown(self):
        """Clean up temporary database"""
        self.prioritizer.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_database_creation(self):
        """Test that database is created with correct schema"""
        self.assertTrue(os.path.exists(self.db_path))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
        result = cursor.fetchone()
        self.assertIsNotNone(result, "learned_move_patterns table should exist")

        # Check table structure
        cursor.execute("PRAGMA table_info(learned_move_patterns)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        expected_columns = ['piece_type', 'move_category', 'distance_from_start',
                           'game_phase', 'times_seen', 'games_won', 'games_lost']

        for col in expected_columns:
            self.assertIn(col, column_names, f"Column {col} should exist in table")

        conn.close()

    def test_update_move_statistics_win(self):
        """Test updating move statistics for a win"""
        self.prioritizer._update_move_statistics(
            piece_type='pawn',
            move_category='capture',
            distance=3,
            phase='opening',
            result='win',
            final_score=100
        )

        patterns = self.prioritizer.get_top_patterns(limit=1)
        self.assertEqual(len(patterns), 1, "Should have 1 pattern")

        piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = patterns[0]
        self.assertEqual(piece_type, 'pawn')
        self.assertEqual(category, 'capture')
        self.assertEqual(distance, 3)
        self.assertEqual(phase, 'opening')
        self.assertEqual(times_seen, 1)
        self.assertEqual(win_rate, 1.0, "Win rate should be 100%")

    def test_win_rate_calculation(self):
        """Test that win rates are calculated correctly"""
        # Record 3 wins, 1 loss
        for _ in range(3):
            self.prioritizer._update_move_statistics(
                'knight', 'fork', 5, 'middlegame', 'win', 200
            )
        self.prioritizer._update_move_statistics(
            'knight', 'fork', 5, 'middlegame', 'loss', -150
        )

        patterns = self.prioritizer.get_top_patterns(limit=1)
        _, _, _, _, times_seen, win_rate, _, _ = patterns[0]

        self.assertEqual(times_seen, 4, "Should have seen pattern 4 times")
        self.assertAlmostEqual(win_rate, 0.75, places=2, msg="Win rate should be 75%")

    def test_priority_score_increases_with_wins(self):
        """Test that priority score increases with winning patterns"""
        # Record a losing pattern
        self.prioritizer._update_move_statistics(
            'piece1', 'bad_move', 1, 'opening', 'loss', -100
        )

        # Record a winning pattern
        for _ in range(5):
            self.prioritizer._update_move_statistics(
                'piece2', 'good_move', 1, 'opening', 'win', 100
            )

        patterns = self.prioritizer.get_top_patterns(limit=2)

        # The winning pattern should have higher priority
        if len(patterns) >= 2:
            win_pattern = [p for p in patterns if p[1] == 'good_move'][0]
            loss_pattern = [p for p in patterns if p[1] == 'bad_move'][0]

            win_priority = win_pattern[7]  # priority is index 7
            loss_priority = loss_pattern[7]

            self.assertGreater(win_priority, loss_priority,
                             "Winning pattern should have higher priority")

    def test_get_move_priority(self):
        """Test retrieving priority for a specific pattern"""
        self.prioritizer._update_move_statistics(
            'test_piece', 'test_move', 5, 'midgame', 'win', 50
        )

        priority = self.prioritizer.get_move_priority(
            'test_piece', 'test_move', 5, 'midgame'
        )

        self.assertIsNotNone(priority, "Should return a priority value")
        self.assertGreater(priority, 0, "Priority should be positive for winning pattern")


class TestCheckersBoard(unittest.TestCase):
    """Test Checkers board implementation"""

    def test_initial_setup(self):
        """Test that board is set up correctly at start"""
        board = CheckersBoard()

        # Count pieces
        white_pieces = board.get_pieces(Color.WHITE)
        black_pieces = board.get_pieces(Color.BLACK)

        self.assertEqual(len(white_pieces), 12, "Should have 12 white pieces")
        self.assertEqual(len(black_pieces), 12, "Should have 12 black pieces")

        # Check all are men (not kings)
        for piece in white_pieces + black_pieces:
            self.assertEqual(piece.type, PieceType.MAN, "All pieces should start as men")

    def test_get_piece(self):
        """Test retrieving pieces from board"""
        board = CheckersBoard()

        # Should have piece at (2, 1)
        piece = board.get_piece((2, 1))
        self.assertIsNotNone(piece, "Should have a piece at (2, 1)")
        self.assertEqual(piece.color, Color.WHITE, "Piece at (2, 1) should be white")

        # Should not have piece at (4, 4)
        empty = board.get_piece((4, 4))
        self.assertIsNone(empty, "Should not have piece at (4, 4)")

    def test_add_remove_piece(self):
        """Test adding and removing pieces"""
        board = CheckersBoard()
        board.clear()

        # Add a piece
        piece = Piece(PieceType.MAN, Color.WHITE, (3, 3))
        board.add_piece(piece)

        self.assertEqual(len(board.get_pieces(Color.WHITE)), 1, "Should have 1 white piece")
        self.assertEqual(board.get_piece((3, 3)), piece, "Piece should be at (3, 3)")

        # Remove the piece
        board.remove_piece((3, 3))
        self.assertEqual(len(board.get_pieces(Color.WHITE)), 0, "Should have 0 white pieces")
        self.assertIsNone(board.get_piece((3, 3)), "Position should be empty")

    def test_piece_hashable(self):
        """Test that pieces can be used in sets/dicts"""
        piece1 = Piece(PieceType.MAN, Color.WHITE, (2, 1))
        piece2 = Piece(PieceType.MAN, Color.WHITE, (2, 1))
        piece3 = Piece(PieceType.MAN, Color.BLACK, (2, 1))

        # Should be able to add to set
        piece_set = {piece1, piece2, piece3}

        # piece1 and piece2 are equal, so set should have 2 elements
        self.assertLessEqual(len(piece_set), 2, "Equal pieces should hash to same value")


class TestCheckersGame(unittest.TestCase):
    """Test Checkers game logic"""

    def test_initial_legal_moves(self):
        """Test that white has legal moves at start"""
        game = CheckersGame()
        legal_moves = game.get_legal_moves()

        self.assertGreater(len(legal_moves), 0, "White should have legal moves at start")
        self.assertLess(len(legal_moves), 10, "Should not have too many moves")

    def test_simple_move(self):
        """Test making a simple move"""
        game = CheckersGame()
        legal_moves = game.get_legal_moves()

        if legal_moves:
            # Make first legal move
            new_game = game.make_move(legal_moves[0])

            # Should switch players
            self.assertEqual(new_game.current_player, Color.BLACK,
                           "Should switch to black after white moves")

            # Should have legal moves for black
            black_moves = new_game.get_legal_moves()
            self.assertGreater(len(black_moves), 0, "Black should have legal moves")

    def test_capture_scenario(self):
        """Test that captures are detected correctly"""
        board = CheckersBoard()
        board.clear()

        # Set up: White at (2, 1), Black at (3, 2), empty at (4, 3)
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))
        board.add_piece(Piece(PieceType.MAN, Color.BLACK, (3, 2)))

        game = CheckersGame(board=board, current_player=Color.WHITE)
        legal_moves = game.get_legal_moves()

        # Should have a capture move
        capture_moves = [m for m in legal_moves if len(m) > 1]
        self.assertGreater(len(capture_moves), 0, "Should have at least one capture move")

    def test_game_over_no_pieces(self):
        """Test game over when no pieces left"""
        board = CheckersBoard()
        board.clear()

        # Only white pieces
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))

        game = CheckersGame(board=board, current_player=Color.BLACK)

        # Black has no pieces, game should be over
        self.assertTrue(game.is_game_over(), "Game should be over when player has no pieces")

    def test_king_promotion(self):
        """Test that men are promoted to kings on back row"""
        board = CheckersBoard()
        board.clear()

        # White man one move from promotion
        piece = Piece(PieceType.MAN, Color.WHITE, (6, 1))
        board.add_piece(piece)

        game = CheckersGame(board=board, current_player=Color.WHITE)
        legal_moves = game.get_legal_moves()

        # Find move to row 7
        promotion_moves = [m for m in legal_moves if m[-1][0] == 7]

        if promotion_moves:
            new_game = game.make_move(promotion_moves[0])

            # Check if piece was promoted
            pieces_at_7 = [p for p in new_game.board.get_pieces(Color.WHITE) if p.position[0] == 7]

            if pieces_at_7:
                self.assertEqual(pieces_at_7[0].type, PieceType.KING,
                               "Piece should be promoted to king on back row")


class TestCheckersDifferentialScoring(unittest.TestCase):
    """Test differential scoring system"""

    def test_equal_position_scores_zero(self):
        """Test that equal positions score 0"""
        board = CheckersBoard()  # Initial position
        scorer = CheckersScorer()

        score = scorer.score(board, Color.WHITE)
        self.assertEqual(score, 0, "Equal position should score 0")

    def test_material_advantage(self):
        """Test material advantage scoring"""
        board = CheckersBoard()
        board.clear()
        scorer = CheckersScorer()

        # White has 1 man, black has nothing
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))

        score = scorer.score(board, Color.WHITE)
        self.assertEqual(score, 100, "1 man advantage should score +100")

    def test_king_value(self):
        """Test that kings are worth more than men"""
        board = CheckersBoard()
        board.clear()
        scorer = CheckersScorer()

        # White has 1 king
        board.add_piece(Piece(PieceType.KING, Color.WHITE, (4, 4)))

        score = scorer.score(board, Color.WHITE)
        self.assertEqual(score, 300, "1 king should score +300")

    def test_differential_nature(self):
        """Test that scoring is differential (my_advantage - opponent_advantage)"""
        board = CheckersBoard()
        board.clear()
        scorer = CheckersScorer()

        # White: 2 men (200), Black: 1 man (100)
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 3)))
        board.add_piece(Piece(PieceType.MAN, Color.BLACK, (5, 2)))

        score_white = scorer.score(board, Color.WHITE)
        score_black = scorer.score(board, Color.BLACK)

        self.assertEqual(score_white, 100, "White should score +100")
        self.assertEqual(score_black, -100, "Black should score -100")
        self.assertEqual(score_white, -score_black, "Scores should be negatives of each other")


def run_unit_tests():
    """Run the unit test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLearnableMovePrioritizer))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckersBoard))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckersGame))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckersDifferentialScoring))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("="*70)
    print("UNIT TESTS FOR PATTERN RECOGNITION AI")
    print("="*70)
    print()

    result = run_unit_tests()

    print()
    print("="*70)
    if result.wasSuccessful():
        print("✓ ALL UNIT TESTS PASSED")
    else:
        print(f"✗ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
    print("="*70)

    exit(0 if result.wasSuccessful() else 1)
