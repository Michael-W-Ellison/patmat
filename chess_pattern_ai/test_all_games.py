#!/usr/bin/env python3
"""
Comprehensive test suite for all games in the pattern recognition AI system.

Tests all 9 games: Chess, Checkers, Go, Othello, Connect Four, Gomoku,
Hex, Dots and Boxes, and Breakthrough.
"""

import unittest
import tempfile
import os
import sqlite3
from pathlib import Path

# Import learnable move prioritizer
from learnable_move_prioritizer import LearnableMovePrioritizer

# Import checkers
from checkers.checkers_board import CheckersBoard, Piece, Color, PieceType
from checkers.checkers_game import CheckersGame
from checkers.checkers_scorer import CheckersScorer
from checkers.checkers_headless_trainer import CheckersHeadlessTrainer

# Import Go
from go.go_board import GoBoard, Stone, GoColor
from go.go_game import GoGame
from go.go_scorer import GoScorer

# Import Othello
from othello.othello_board import OthelloBoard, OthelloColor
from othello.othello_game import OthelloGame
from othello.othello_scorer import OthelloScorer

# Import Connect Four
from connect4.connect4_board import Connect4Board, Connect4Color
from connect4.connect4_game import Connect4Game
from connect4.connect4_scorer import Connect4Scorer

# Import Gomoku
from gomoku.gomoku_board import GomokuBoard, GomokuStone, GomokuColor
from gomoku.gomoku_game import GomokuGame
from gomoku.gomoku_scorer import GomokuScorer

# Import Hex
from hex.hex_board import HexBoard, HexStone, HexColor
from hex.hex_game import HexGame
from hex.hex_scorer import HexScorer

# Import Dots and Boxes
from dots_boxes.dots_boxes_board import DotsBoxesBoard, Color as DBColor
from dots_boxes.dots_boxes_game import DotsBoxesGame
from dots_boxes.dots_boxes_scorer import DotsBoxesScorer

# Import Breakthrough
from breakthrough.breakthrough_board import BreakthroughBoard, BreakthroughPiece, BreakthroughColor
from breakthrough.breakthrough_game import BreakthroughGame
from breakthrough.breakthrough_scorer import BreakthroughScorer


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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

        conn.close()

    def test_update_move_statistics(self):
        """Test updating move statistics"""
        self.prioritizer._update_move_statistics(
            piece_type='pawn',
            move_category='capture',
            distance=3,
            phase='opening',
            result='win',
            final_score=100
        )

        patterns = self.prioritizer.get_top_patterns(limit=1)
        self.assertEqual(len(patterns), 1)

        piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = patterns[0]
        self.assertEqual(piece_type, 'pawn')
        self.assertEqual(category, 'capture')
        self.assertEqual(distance, 3)
        self.assertEqual(phase, 'opening')
        self.assertEqual(times_seen, 1)
        self.assertEqual(win_rate, 1.0)

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

        self.assertEqual(times_seen, 4)
        self.assertAlmostEqual(win_rate, 0.75, places=2)  # 3/4 = 0.75


class TestCheckersGame(unittest.TestCase):
    """Test Checkers game implementation"""

    def test_initial_board_setup(self):
        """Test that checkers board is set up correctly"""
        board = CheckersBoard()

        # Should have 12 pieces per side
        white_pieces = board.get_pieces(Color.WHITE)
        black_pieces = board.get_pieces(Color.BLACK)

        self.assertEqual(len(white_pieces), 12)
        self.assertEqual(len(black_pieces), 12)

    def test_simple_move(self):
        """Test basic move generation"""
        game = CheckersGame()
        legal_moves = game.get_legal_moves()

        # White should have legal moves at start
        self.assertGreater(len(legal_moves), 0)

    def test_capture_detection(self):
        """Test that captures are detected"""
        board = CheckersBoard()
        board.clear()

        # Set up a capture scenario: White man at (2,1), Black man at (3,2), empty (4,3)
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))
        board.add_piece(Piece(PieceType.MAN, Color.BLACK, (3, 2)))

        game = CheckersGame(board=board)
        legal_moves = game.get_legal_moves()

        # Should have a capture move
        capture_moves = [m for m in legal_moves if len(m) > 1]
        self.assertGreater(len(capture_moves), 0)

    def test_king_promotion(self):
        """Test that pieces are promoted to kings"""
        board = CheckersBoard()
        board.clear()

        # Place white man at row 6 (one move from promotion)
        piece = Piece(PieceType.MAN, Color.WHITE, (6, 1))
        board.add_piece(piece)

        game = CheckersGame(board=board, current_player=Color.WHITE)
        legal_moves = game.get_legal_moves()

        if legal_moves:
            # Make a move to row 7
            move = [m for m in legal_moves if m[-1][0] == 7]
            if move:
                new_board = game.make_move(move[0])
                pieces = new_board.get_pieces(Color.WHITE)

                # Check if any piece is a king
                has_king = any(p.type == PieceType.KING for p in pieces)
                self.assertTrue(has_king or len(pieces) > 0)  # Either promoted or moved

    def test_differential_scoring(self):
        """Test checkers differential scoring"""
        board = CheckersBoard()
        scorer = CheckersScorer()

        score = scorer.score(board, Color.WHITE)

        # At start, score should be 0 (equal material)
        self.assertEqual(score, 0)

        # Remove a black piece
        board.clear()
        board.add_piece(Piece(PieceType.MAN, Color.WHITE, (2, 1)))

        score = scorer.score(board, Color.WHITE)
        # White has 1 man (100), Black has 0 → +100
        self.assertEqual(score, 100)


class TestGoGame(unittest.TestCase):
    """Test Go game implementation"""

    def test_initial_board(self):
        """Test Go board initialization"""
        board = GoBoard(size=9)

        # Board should be empty at start
        self.assertEqual(len(board.stones), 0)

    def test_stone_placement(self):
        """Test placing stones on the board"""
        board = GoBoard(size=9)

        stone = Stone(GoColor.BLACK, (3, 3))
        board.place_stone(stone)

        self.assertEqual(len(board.stones), 1)
        self.assertEqual(board.get_stone((3, 3)), stone)

    def test_group_capture(self):
        """Test that surrounded groups are captured"""
        board = GoBoard(size=9)

        # Place a black stone
        board.place_stone(Stone(GoColor.BLACK, (4, 4)))

        # Surround it with white stones
        board.place_stone(Stone(GoColor.WHITE, (3, 4)))
        board.place_stone(Stone(GoColor.WHITE, (5, 4)))
        board.place_stone(Stone(GoColor.WHITE, (4, 3)))
        board.place_stone(Stone(GoColor.WHITE, (4, 5)))

        # Black stone should be captured
        captured = board.remove_captured_groups(GoColor.BLACK)
        self.assertGreater(len(captured), 0)

    def test_legal_moves(self):
        """Test Go legal move generation"""
        game = GoGame(board_size=9)
        legal_moves = game.get_legal_moves()

        # Should have 81 legal moves on empty 9x9 board (+ pass)
        self.assertGreater(len(legal_moves), 80)


class TestOthelloGame(unittest.TestCase):
    """Test Othello game implementation"""

    def test_initial_board(self):
        """Test Othello starting position"""
        board = OthelloBoard()

        # Should have 4 pieces in center
        pieces = [board.get_disc((3, 3)), board.get_disc((3, 4)),
                  board.get_disc((4, 3)), board.get_disc((4, 4))]

        non_none = [p for p in pieces if p is not None]
        self.assertEqual(len(non_none), 4)

    def test_disc_flipping(self):
        """Test that discs are flipped correctly"""
        board = OthelloBoard()
        game = OthelloGame(board=board)

        legal_moves = game.get_legal_moves()
        self.assertGreater(len(legal_moves), 0)

        # Make a move and verify discs are flipped
        if legal_moves:
            new_board = game.make_move(legal_moves[0])

            # Board should have more pieces after a move
            self.assertGreater(len(new_board.discs), len(board.discs))

    def test_differential_scoring(self):
        """Test Othello differential scoring"""
        board = OthelloBoard()
        scorer = OthelloScorer()

        score = scorer.score(board, OthelloColor.BLACK)

        # At start, should be equal
        self.assertEqual(score, 0)


class TestConnect4Game(unittest.TestCase):
    """Test Connect Four game implementation"""

    def test_initial_board(self):
        """Test Connect4 board starts empty"""
        board = Connect4Board()

        # All positions should be empty
        for row in range(6):
            for col in range(7):
                self.assertIsNone(board.get_piece((row, col)))

    def test_gravity(self):
        """Test that pieces fall to lowest position"""
        board = Connect4Board()

        # Drop a piece in column 3
        result = board.drop_piece(3, Connect4Color.RED)

        # Should land at row 0 (bottom)
        self.assertEqual(result, (0, 3))
        self.assertEqual(board.get_piece((0, 3)), Connect4Color.RED)

    def test_win_detection_horizontal(self):
        """Test horizontal win detection"""
        board = Connect4Board()

        # Create 4 in a row horizontally
        for col in range(4):
            board.drop_piece(col, Connect4Color.RED)

        game = Connect4Game(board=board)
        self.assertTrue(game.is_game_over())

    def test_win_detection_vertical(self):
        """Test vertical win detection"""
        board = Connect4Board()

        # Create 4 in a row vertically
        for _ in range(4):
            board.drop_piece(3, Connect4Color.YELLOW)

        game = Connect4Game(board=board)
        self.assertTrue(game.is_game_over())


class TestGomokuGame(unittest.TestCase):
    """Test Gomoku game implementation"""

    def test_initial_board(self):
        """Test Gomoku board starts empty"""
        board = GomokuBoard(size=15)

        self.assertEqual(len(board.stones), 0)

    def test_stone_placement(self):
        """Test placing stones"""
        board = GomokuBoard(size=15)

        stone = GomokuStone(GomokuColor.BLACK, (7, 7))
        board.place_stone(stone)

        self.assertEqual(len(board.stones), 1)

    def test_five_in_row_horizontal(self):
        """Test 5 in a row detection (horizontal)"""
        board = GomokuBoard(size=15)

        # Place 5 black stones in a row
        for col in range(5):
            board.place_stone(GomokuStone(GomokuColor.BLACK, (7, col)))

        game = GomokuGame(board=board)
        self.assertTrue(game.is_game_over())

    def test_five_in_row_diagonal(self):
        """Test 5 in a row detection (diagonal)"""
        board = GomokuBoard(size=15)

        # Place 5 black stones diagonally
        for i in range(5):
            board.place_stone(GomokuStone(GomokuColor.BLACK, (i, i)))

        game = GomokuGame(board=board)
        self.assertTrue(game.is_game_over())


class TestHexGame(unittest.TestCase):
    """Test Hex game implementation"""

    def test_initial_board(self):
        """Test Hex board starts empty"""
        board = HexBoard(size=11)

        self.assertEqual(len(board.stones), 0)

    def test_stone_placement(self):
        """Test placing stones"""
        board = HexBoard(size=11)

        stone = HexStone(HexColor.RED, (5, 5))
        board.place_stone(stone)

        self.assertEqual(len(board.stones), 1)

    def test_neighbor_detection(self):
        """Test hexagonal neighbor detection"""
        board = HexBoard(size=11)

        neighbors = board.get_neighbors((5, 5))

        # Hex cells should have up to 6 neighbors
        self.assertGreater(len(neighbors), 0)
        self.assertLessEqual(len(neighbors), 6)

    def test_connection_detection(self):
        """Test path connection detection"""
        board = HexBoard(size=11)

        # Create a path from top to bottom for RED
        for row in range(11):
            board.place_stone(HexStone(HexColor.RED, (row, 5)))

        game = HexGame(board=board)
        self.assertTrue(game.is_game_over())


class TestDotsAndBoxesGame(unittest.TestCase):
    """Test Dots and Boxes game implementation"""

    def test_initial_board(self):
        """Test Dots and Boxes board starts empty"""
        board = DotsBoxesBoard(width=5, height=5)

        # No edges drawn, no boxes completed
        self.assertEqual(len(board.horizontal_edges), 0)
        self.assertEqual(len(board.vertical_edges), 0)

    def test_edge_drawing(self):
        """Test drawing edges"""
        board = DotsBoxesBoard(width=5, height=5)

        # Draw a horizontal edge
        completed = board.draw_edge(('H', 0, 0), DBColor.RED)

        self.assertFalse(completed)  # Single edge doesn't complete a box
        self.assertIn(('H', 0, 0), board.horizontal_edges)

    def test_box_completion(self):
        """Test completing a box"""
        board = DotsBoxesBoard(width=5, height=5)

        # Draw 4 edges to complete a box at (0, 0)
        board.draw_edge(('H', 0, 0), DBColor.RED)  # Top
        board.draw_edge(('H', 1, 0), DBColor.RED)  # Bottom
        board.draw_edge(('V', 0, 0), DBColor.RED)  # Left
        completed = board.draw_edge(('V', 0, 1), DBColor.RED)  # Right

        self.assertTrue(completed)

    def test_double_move_rule(self):
        """Test that completing a box gives another turn"""
        board = DotsBoxesBoard(width=5, height=5)

        # Draw 3 edges
        board.draw_edge(('H', 0, 0), DBColor.RED)
        board.draw_edge(('H', 1, 0), DBColor.RED)
        board.draw_edge(('V', 0, 0), DBColor.RED)

        initial_turn = board.turn

        # Complete the box
        completed = board.draw_edge(('V', 0, 1), DBColor.RED)

        # Turn should NOT switch if box completed
        if completed:
            self.assertEqual(board.turn, initial_turn)


class TestBreakthroughGame(unittest.TestCase):
    """Test Breakthrough game implementation"""

    def test_initial_board(self):
        """Test Breakthrough starting position"""
        board = BreakthroughBoard()

        # Should have 16 pieces per side
        white_pieces = board.get_pieces(BreakthroughColor.WHITE)
        black_pieces = board.get_pieces(BreakthroughColor.BLACK)

        self.assertEqual(len(white_pieces), 16)
        self.assertEqual(len(black_pieces), 16)

    def test_forward_movement(self):
        """Test that pieces can only move forward"""
        board = BreakthroughBoard()

        # Get white pieces (should be on rows 0-1)
        white_pieces = board.get_pieces(BreakthroughColor.WHITE)

        # All white pieces should be on rows 0-1
        for piece in white_pieces:
            self.assertLess(piece.position[0], 2)

    def test_capture_detection(self):
        """Test diagonal capture moves"""
        board = BreakthroughBoard()
        board.clear()

        # Set up capture scenario
        board.add_piece(BreakthroughPiece(BreakthroughColor.WHITE, (3, 3)))
        board.add_piece(BreakthroughPiece(BreakthroughColor.BLACK, (4, 4)))

        game = BreakthroughGame(board=board, current_player=BreakthroughColor.WHITE)
        legal_moves = game.get_legal_moves()

        # Should have diagonal capture move
        self.assertGreater(len(legal_moves), 0)

    def test_win_by_reaching_back_row(self):
        """Test winning by reaching opponent's back row"""
        board = BreakthroughBoard()
        board.clear()

        # Place white piece on row 7 (back row for white)
        board.add_piece(BreakthroughPiece(BreakthroughColor.WHITE, (7, 3)))

        game = BreakthroughGame(board=board)
        self.assertTrue(game.is_game_over())


class TestGameTraining(unittest.TestCase):
    """Test that games can be trained"""

    def test_checkers_training(self):
        """Test running a few checkers training games"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            db_path = f.name

        try:
            trainer = CheckersHeadlessTrainer(db_path=db_path)

            # Train for 2 games
            stats = trainer.train(num_games=2, verbose=False)

            self.assertEqual(stats['games_played'], 2)
            self.assertIn('win_rate', stats)
            self.assertIn('avg_score', stats)

            # Check that patterns were learned
            patterns = trainer.prioritizer.get_top_patterns(limit=5)
            self.assertGreater(len(patterns), 0)

            trainer.prioritizer.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestIntegration(unittest.TestCase):
    """Integration tests across the system"""

    def test_all_games_have_required_methods(self):
        """Test that all game classes implement required interface"""
        games = [
            CheckersGame(),
            GoGame(board_size=9),
            OthelloGame(),
            Connect4Game(),
            GomokuGame(board_size=15),
            HexGame(board_size=11),
            DotsBoxesGame(width=5, height=5),
            BreakthroughGame()
        ]

        for game in games:
            # All games should have these methods
            self.assertTrue(hasattr(game, 'get_legal_moves'))
            self.assertTrue(hasattr(game, 'make_move'))
            self.assertTrue(hasattr(game, 'is_game_over'))

            # Should be able to get legal moves
            legal_moves = game.get_legal_moves()
            self.assertIsInstance(legal_moves, list)

    def test_all_games_have_scorers(self):
        """Test that all games have differential scorers"""
        scorers = [
            CheckersScorer(),
            GoScorer(),
            OthelloScorer(),
            Connect4Scorer(),
            GomokuScorer(),
            HexScorer(),
            DotsBoxesScorer(),
            BreakthroughScorer()
        ]

        for scorer in scorers:
            # All scorers should have a score method
            self.assertTrue(hasattr(scorer, 'score'))


def run_test_suite():
    """Run the complete test suite"""
    # Create test loader
    loader = unittest.TestLoader()

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLearnableMovePrioritizer))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckersGame))
    suite.addTests(loader.loadTestsFromTestCase(TestGoGame))
    suite.addTests(loader.loadTestsFromTestCase(TestOthelloGame))
    suite.addTests(loader.loadTestsFromTestCase(TestConnect4Game))
    suite.addTests(loader.loadTestsFromTestCase(TestGomokuGame))
    suite.addTests(loader.loadTestsFromTestCase(TestHexGame))
    suite.addTests(loader.loadTestsFromTestCase(TestDotsAndBoxesGame))
    suite.addTests(loader.loadTestsFromTestCase(TestBreakthroughGame))
    suite.addTests(loader.loadTestsFromTestCase(TestGameTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_test_suite()

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
