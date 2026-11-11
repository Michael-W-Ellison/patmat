#!/usr/bin/env python3
"""
Comprehensive test suite for all games in the pattern recognition AI system.

Tests the learnable move prioritizer and 8 games: Checkers, Go, Othello,
Connect Four, Gomoku, Hex, Dots and Boxes, and Breakthrough.
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
from go.go_board import GoBoard, Stone, Color as GoColor, Move as GoMove
from go.go_game import GoGame
from go.go_scorer import GoScorer

# Import Othello
from othello.othello_board import OthelloBoard, Color as OthelloColor
from othello.othello_game import OthelloGame
from othello.othello_scorer import OthelloScorer

# Import Connect Four
from connect4.connect4_board import Connect4Board, Color as Connect4Color
from connect4.connect4_game import Connect4Game
from connect4.connect4_scorer import Connect4Scorer

# Import Gomoku
from gomoku.gomoku_board import (GomokuBoard, Stone as GomokuStone, Color as GomokuColor,
                                 Move as GomokuMove)
from gomoku.gomoku_game import GomokuGame
from gomoku.gomoku_scorer import GomokuScorer

# Import Hex
from hex.hex_board import HexBoard, Stone as HexStone, Color as HexColor, Move as HexMove
from hex.hex_game import HexGame
from hex.hex_scorer import HexScorer

# Import Dots and Boxes
from dots_boxes.dots_boxes_board import DotsBoxesBoard, Color as DBColor, Edge
from dots_boxes.dots_boxes_game import DotsBoxesGame
from dots_boxes.dots_boxes_scorer import DotsBoxesScorer

# Import Breakthrough
from breakthrough.breakthrough_board import BreakthroughBoard, Piece as BreakthroughPiece, Color as BreakthroughColor
from breakthrough.breakthrough_game import BreakthroughGame
from breakthrough.breakthrough_scorer import BreakthroughScorer


def _empty_board(board_class):
    """A checkers/breakthrough board with every piece removed (8x8 grid)"""
    board = board_class()
    board.board = [[None for _ in range(8)] for _ in range(8)]
    board.pieces = set()
    return board


def _add_piece(board, piece):
    """Put a piece on a checkers/breakthrough board"""
    row, col = piece.position
    board.board[row][col] = piece
    board.pieces.add(piece)


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
            result='win',
            final_score=100
        )
        self.prioritizer.conn.commit()

        # One game's evidence is below get_top_patterns' confidence cut-off,
        # so read the row directly
        rows = self.prioritizer.cursor.execute('''
            SELECT piece_type, move_category, distance_from_start,
                   times_seen, games_won, win_rate, avg_score
            FROM learned_move_patterns
        ''').fetchall()
        self.assertEqual(len(rows), 1)

        piece_type, category, distance, times_seen, games_won, win_rate, avg_score = rows[0]
        self.assertEqual(piece_type, 'pawn')
        self.assertEqual(category, 'capture')
        self.assertEqual(distance, 3)
        self.assertEqual(times_seen, 1)
        self.assertEqual(games_won, 1)
        self.assertEqual(win_rate, 1.0)
        self.assertEqual(avg_score, 100)

    def test_win_rate_calculation(self):
        """Test that win rates are calculated correctly"""
        pattern = [{'piece_type': 'knight', 'category': 'fork', 'distance': 5}]

        # Record 3 won games and 1 lost game using the pattern
        for _ in range(3):
            self.prioritizer.record_game_patterns(pattern, 'win', 200)
        self.prioritizer.record_game_patterns(pattern, 'loss', -150)

        patterns = self.prioritizer.get_top_patterns(limit=1)
        self.assertEqual(len(patterns), 1)
        piece_type, category, distance, times_seen, win_rate, _, _ = patterns[0]

        self.assertEqual((piece_type, category, distance), ('knight', 'fork', 5))
        self.assertEqual(times_seen, 4)
        self.assertAlmostEqual(win_rate, 0.75, places=2)  # 3/4 = 0.75


class TestCheckersGame(unittest.TestCase):
    """Test Checkers game implementation"""

    def test_initial_board_setup(self):
        """Test that checkers board is set up correctly"""
        board = CheckersBoard()

        # Should have 12 pieces per side
        red_pieces = board.get_pieces(Color.RED)
        black_pieces = board.get_pieces(Color.BLACK)

        self.assertEqual(len(red_pieces), 12)
        self.assertEqual(len(black_pieces), 12)

    def test_simple_move(self):
        """Test basic move generation"""
        game = CheckersGame()
        legal_moves = game.get_legal_moves(game.board.turn)

        # Red (first to move) should have legal moves at start
        self.assertGreater(len(legal_moves), 0)

    def test_capture_detection(self):
        """Test that captures are detected"""
        board = _empty_board(CheckersBoard)

        # Red man at (5,2) moves toward row 0; Black man at (4,3); (3,4) empty
        _add_piece(board, Piece(PieceType.MAN, Color.RED, (5, 2)))
        black = Piece(PieceType.MAN, Color.BLACK, (4, 3))
        _add_piece(board, black)

        game = CheckersGame(board=board)
        legal_moves = game.get_legal_moves(Color.RED)

        # Should have a capture move (and captures are mandatory)
        capture_moves = [m for m in legal_moves if m.is_jump]
        self.assertGreater(len(capture_moves), 0)
        self.assertEqual(len(capture_moves), len(legal_moves))
        self.assertEqual(capture_moves[0].to_pos, (3, 4))
        self.assertIn(black, capture_moves[0].captured_pieces)

        game.make_move(capture_moves[0])
        self.assertIsNone(board.get_piece_at((4, 3)))
        self.assertEqual(len(board.get_pieces(Color.BLACK)), 0)

    def test_king_promotion(self):
        """Test that pieces are promoted to kings"""
        board = _empty_board(CheckersBoard)

        # Red man at row 1 (one move from promotion on row 0)
        _add_piece(board, Piece(PieceType.MAN, Color.RED, (1, 2)))
        # A distant black piece so the game is not over
        _add_piece(board, Piece(PieceType.MAN, Color.BLACK, (7, 0)))

        game = CheckersGame(board=board)
        legal_moves = game.get_legal_moves(Color.RED)

        promoting = [m for m in legal_moves if m.to_pos[0] == 0]
        self.assertGreater(len(promoting), 0)

        self.assertTrue(game.make_move(promoting[0]))
        piece = board.get_piece_at(promoting[0].to_pos)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.KING)

    def test_differential_scoring(self):
        """Test checkers differential scoring"""
        board = CheckersBoard()
        scorer = CheckersScorer()

        # At start, material is equal and nobody has won
        advantage = (scorer.calculate_material(board, Color.RED) -
                     scorer.calculate_material(board, Color.BLACK))
        self.assertEqual(advantage, 0)
        self.assertEqual(scorer.calculate_final_score(board, Color.RED, 0), (0, 'draw'))

        # Only one red man left
        board = _empty_board(CheckersBoard)
        _add_piece(board, Piece(PieceType.MAN, Color.RED, (5, 2)))

        # Red has 1 man (100), Black has 0 -> +100 material advantage
        advantage = (scorer.calculate_material(board, Color.RED) -
                     scorer.calculate_material(board, Color.BLACK))
        self.assertEqual(advantage, 100)

        # Red won: +100 material + 1000 win bonus (no time bonus after 100 rounds)
        self.assertEqual(scorer.calculate_final_score(board, Color.RED, 100), (1100, 'win'))
        self.assertEqual(scorer.calculate_final_score(board, Color.BLACK, 100), (-1100, 'loss'))


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

        self.assertTrue(board.make_move(GoMove(GoColor.BLACK, (3, 3))))

        self.assertEqual(len(board.stones), 1)
        self.assertEqual(board.get_stone_at((3, 3)), Stone(GoColor.BLACK, (3, 3)))

    def test_group_capture(self):
        """Test that surrounded groups are captured"""
        board = GoBoard(size=9)

        # Place a black stone
        board.make_move(GoMove(GoColor.BLACK, (4, 4)))

        # Surround it with white stones
        board.make_move(GoMove(GoColor.WHITE, (3, 4)))
        board.make_move(GoMove(GoColor.WHITE, (5, 4)))
        board.make_move(GoMove(GoColor.WHITE, (4, 3)))
        last = GoMove(GoColor.WHITE, (4, 5))
        board.make_move(last)

        # Black stone should be captured by the move that removed its last liberty
        self.assertIn(Stone(GoColor.BLACK, (4, 4)), last.captured)
        self.assertIsNone(board.get_stone_at((4, 4)))
        self.assertEqual(board.captured_by_white, 1)

    def test_legal_moves(self):
        """Test Go legal move generation"""
        game = GoGame(board_size=9)
        legal_moves = game.get_legal_moves(GoColor.BLACK)

        # Should have 81 legal moves on empty 9x9 board (+ pass)
        self.assertEqual(len(legal_moves), 82)


class TestOthelloGame(unittest.TestCase):
    """Test Othello game implementation"""

    def test_initial_board(self):
        """Test Othello starting position"""
        board = OthelloBoard()

        # Should have 4 pieces in center
        pieces = [board.get_disc_at((3, 3)), board.get_disc_at((3, 4)),
                  board.get_disc_at((4, 3)), board.get_disc_at((4, 4))]

        non_none = [p for p in pieces if p is not None]
        self.assertEqual(len(non_none), 4)
        self.assertEqual(board.count_discs(OthelloColor.BLACK), 2)
        self.assertEqual(board.count_discs(OthelloColor.WHITE), 2)

    def test_disc_flipping(self):
        """Test that discs are flipped correctly"""
        board = OthelloBoard()
        game = OthelloGame(board=board)

        legal_moves = game.get_legal_moves(OthelloColor.BLACK)
        self.assertGreater(len(legal_moves), 0)

        # Make a move and verify discs are flipped
        move = legal_moves[0]
        self.assertFalse(move.is_pass)
        flipped_positions = [d.position for d in move.flipped_discs]
        self.assertGreater(len(flipped_positions), 0)

        self.assertTrue(game.make_move(move))

        # Board should have one more disc; every opening move flips exactly one
        self.assertEqual(len(board.discs), 5)
        self.assertEqual(board.count_discs(OthelloColor.BLACK), 4)
        self.assertEqual(board.count_discs(OthelloColor.WHITE), 1)
        for pos in flipped_positions:
            self.assertEqual(board.get_disc_at(pos).color, OthelloColor.BLACK)

    def test_differential_scoring(self):
        """Test Othello differential scoring"""
        board = OthelloBoard()
        scorer = OthelloScorer()

        # At start, should be equal
        advantage = (scorer.calculate_material(board, OthelloColor.BLACK) -
                     scorer.calculate_material(board, OthelloColor.WHITE))
        self.assertEqual(advantage, 0)
        self.assertEqual(scorer.calculate_final_score(board, OthelloColor.BLACK, 0), (0, 'draw'))


class TestConnect4Game(unittest.TestCase):
    """Test Connect Four game implementation"""

    @staticmethod
    def _drop(game, column, color):
        """Drop a piece of color into column; returns the move made"""
        move = next(m for m in game.get_legal_moves(color) if m.column == column)
        game.make_move(move)
        return move

    def test_initial_board(self):
        """Test Connect4 board starts empty"""
        board = Connect4Board()

        # All positions should be empty
        for row in range(6):
            for col in range(7):
                self.assertIsNone(board.get_piece_at((row, col)))

    def test_gravity(self):
        """Test that pieces fall to lowest position"""
        game = Connect4Game()
        board = game.board

        # Drop a piece in column 3: lands on the bottom row (row 5)
        self._drop(game, 3, Connect4Color.RED)
        self.assertEqual(board.get_piece_at((5, 3)).color, Connect4Color.RED)

        # The next piece in that column stacks on top of it
        self._drop(game, 3, Connect4Color.YELLOW)
        self.assertEqual(board.get_piece_at((4, 3)).color, Connect4Color.YELLOW)
        for row in range(4):
            self.assertIsNone(board.get_piece_at((row, 3)))

    def test_win_detection_horizontal(self):
        """Test horizontal win detection"""
        game = Connect4Game()

        # Create 4 in a row horizontally
        for col in range(4):
            self.assertFalse(game.is_game_over())
            self._drop(game, col, Connect4Color.RED)

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), Connect4Color.RED)

    def test_win_detection_vertical(self):
        """Test vertical win detection"""
        game = Connect4Game()

        # Create 4 in a row vertically
        for _ in range(4):
            self.assertFalse(game.is_game_over())
            self._drop(game, 3, Connect4Color.YELLOW)

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), Connect4Color.YELLOW)


class TestGomokuGame(unittest.TestCase):
    """Test Gomoku game implementation"""

    @staticmethod
    def _place(game, pos, color):
        stone = GomokuStone(color, pos)
        return game.make_move(GomokuMove(position=pos, color=color, stone=stone))

    def test_initial_board(self):
        """Test Gomoku board starts empty"""
        board = GomokuBoard(size=15)

        self.assertEqual(len(board.stones), 0)

    def test_stone_placement(self):
        """Test placing stones"""
        game = GomokuGame(board_size=15)

        self.assertTrue(self._place(game, (7, 7), GomokuColor.BLACK))

        self.assertEqual(len(game.board.stones), 1)
        self.assertEqual(game.board.get_stone_at((7, 7)), GomokuStone(GomokuColor.BLACK, (7, 7)))
        self.assertFalse(game.board.can_place_stone((7, 7)))

    def test_five_in_row_horizontal(self):
        """Test 5 in a row detection (horizontal)"""
        game = GomokuGame(board_size=15)

        # Place 5 black stones in a row
        for col in range(5):
            self.assertFalse(game.is_game_over())
            self._place(game, (7, col), GomokuColor.BLACK)

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), GomokuColor.BLACK)

    def test_five_in_row_diagonal(self):
        """Test 5 in a row detection (diagonal)"""
        game = GomokuGame(board_size=15)

        # Place 5 black stones diagonally
        for i in range(5):
            self.assertFalse(game.is_game_over())
            self._place(game, (i, i), GomokuColor.BLACK)

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), GomokuColor.BLACK)


class TestHexGame(unittest.TestCase):
    """Test Hex game implementation"""

    def test_initial_board(self):
        """Test Hex board starts empty"""
        board = HexBoard(size=11)

        self.assertEqual(len(board.stones), 0)

    def test_stone_placement(self):
        """Test placing stones"""
        board = HexBoard(size=11)

        self.assertTrue(board.make_move(HexMove(HexColor.RED, (5, 5))))

        self.assertEqual(len(board.stones), 1)
        self.assertEqual(board.get_stone_at((5, 5)), HexStone(HexColor.RED, (5, 5)))

    def test_neighbor_detection(self):
        """Test hexagonal neighbor detection"""
        board = HexBoard(size=11)

        neighbors = board.get_neighbors((5, 5))

        # Hex cells should have up to 6 neighbors
        self.assertGreater(len(neighbors), 0)
        self.assertLessEqual(len(neighbors), 6)

    def test_connection_detection(self):
        """Test path connection detection"""
        game = HexGame(board_size=11)

        # Create a path from top to bottom for RED
        for row in range(11):
            self.assertFalse(game.is_game_over())
            game.make_move(HexMove(HexColor.RED, (row, 5)))

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), HexColor.RED)


class TestDotsAndBoxesGame(unittest.TestCase):
    """Test Dots and Boxes game implementation"""

    # The four edges of box (0, 0)
    TOP = Edge((0, 0), (0, 1))
    BOTTOM = Edge((1, 0), (1, 1))
    LEFT = Edge((0, 0), (1, 0))
    RIGHT = Edge((0, 1), (1, 1))

    def test_initial_board(self):
        """Test Dots and Boxes board starts empty"""
        board = DotsBoxesBoard()

        # No edges drawn, no boxes completed
        self.assertEqual(len(board.edges), 0)
        self.assertTrue(all(owner is None for owner in board.boxes.values()))
        # 5x5 dots: 2 * 5 * 4 = 40 edges available
        self.assertEqual(len(board.get_legal_edges()), 40)

    def test_edge_drawing(self):
        """Test drawing edges"""
        board = DotsBoxesBoard()

        # Draw a horizontal edge
        completed = board.draw_edge(self.TOP, DBColor.RED)

        self.assertIsNone(completed)  # Single edge doesn't complete a box
        self.assertIn(self.TOP, board.edges)
        self.assertFalse(board.can_draw_edge(self.TOP))

    def test_box_completion(self):
        """Test completing a box"""
        board = DotsBoxesBoard()

        # Draw 4 edges to complete a box at (0, 0)
        board.draw_edge(self.TOP, DBColor.RED)
        board.draw_edge(self.BOTTOM, DBColor.RED)
        board.draw_edge(self.LEFT, DBColor.RED)
        completed = board.draw_edge(self.RIGHT, DBColor.RED)

        self.assertEqual(completed, (0, 0))
        self.assertEqual(board.boxes[(0, 0)], DBColor.RED)

    def test_double_move_rule(self):
        """Test that completing a box gives another turn"""
        game = DotsBoxesGame()

        # Draw 3 edges; none completes a box, so the turn passes each time
        for edge in (self.TOP, self.BOTTOM, self.LEFT):
            before = game.current_player
            success, completed = game.make_move(edge)
            self.assertTrue(success)
            self.assertIsNone(completed)
            self.assertNotEqual(game.current_player, before)

        initial_turn = game.current_player

        # Complete the box
        success, completed = game.make_move(self.RIGHT)
        self.assertTrue(success)
        self.assertEqual(completed, (0, 0))

        # Turn should NOT switch if box completed
        self.assertEqual(game.current_player, initial_turn)
        self.assertTrue(game.get_another_turn())
        self.assertEqual(game.board.boxes[(0, 0)], initial_turn)


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

        # All white pieces should be on rows 0-1
        for piece in board.get_pieces(BreakthroughColor.WHITE):
            self.assertLess(piece.position[0], 2)

        # White moves toward row 7, Black toward row 0, one row at a time
        game = BreakthroughGame(board=board)
        for move in game.get_legal_moves(BreakthroughColor.WHITE):
            self.assertEqual(move.to_pos[0], move.from_pos[0] + 1)
        for move in game.get_legal_moves(BreakthroughColor.BLACK):
            self.assertEqual(move.to_pos[0], move.from_pos[0] - 1)

    def test_capture_detection(self):
        """Test diagonal capture moves"""
        board = _empty_board(BreakthroughBoard)

        # Set up capture scenario
        _add_piece(board, BreakthroughPiece(BreakthroughColor.WHITE, (3, 3)))
        black = BreakthroughPiece(BreakthroughColor.BLACK, (4, 4))
        _add_piece(board, black)

        game = BreakthroughGame(board=board)
        legal_moves = game.get_legal_moves(BreakthroughColor.WHITE)

        # Should have diagonal capture move
        captures = [m for m in legal_moves if m.is_capture]
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0].to_pos, (4, 4))
        self.assertEqual(captures[0].captured_piece, black)

        game.make_move(captures[0])
        self.assertEqual(board.get_piece_at((4, 4)).color, BreakthroughColor.WHITE)
        self.assertEqual(len(board.get_pieces(BreakthroughColor.BLACK)), 0)

    def test_win_by_reaching_back_row(self):
        """Test winning by reaching opponent's back row"""
        board = _empty_board(BreakthroughBoard)

        # White piece one step from row 7 (back row for white), Black far away
        white = BreakthroughPiece(BreakthroughColor.WHITE, (6, 3))
        _add_piece(board, white)
        _add_piece(board, BreakthroughPiece(BreakthroughColor.BLACK, (3, 0)))

        game = BreakthroughGame(board=board)
        self.assertFalse(game.is_game_over())

        move = next(m for m in game.get_legal_moves(BreakthroughColor.WHITE)
                    if m.to_pos == (7, 3))
        game.make_move(move)

        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winner(), BreakthroughColor.WHITE)


class TestGameTraining(unittest.TestCase):
    """Test that games can be trained"""

    def test_checkers_training(self):
        """Test running a few checkers training games"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, 'checkers_training.db')
            trainer = CheckersHeadlessTrainer(db_path=db_path)

            try:
                # Train for 2 games
                trainer.train(num_games=2, verbose=False)

                self.assertEqual(trainer.games_played, 2)
                self.assertEqual(trainer.wins + trainer.losses + trainer.draws, 2)
                self.assertEqual(len(trainer.score_history), 2)

                # Check that patterns were learned (stored; two games are
                # below get_top_patterns' confidence cut-off)
                count = trainer.prioritizer.cursor.execute(
                    "SELECT COUNT(*) FROM learned_move_patterns").fetchone()[0]
                self.assertGreater(count, 0)
            finally:
                trainer.prioritizer.close()
                trainer.general.close()


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
            DotsBoxesGame(),
            BreakthroughGame()
        ]

        for game in games:
            with self.subTest(game=type(game).__name__):
                # All games should have these methods
                self.assertTrue(hasattr(game, 'get_legal_moves'))
                self.assertTrue(hasattr(game, 'make_move'))
                self.assertTrue(hasattr(game, 'is_game_over'))

                # Should be able to get legal moves for the player to move
                legal_moves = game.get_legal_moves(game.board.turn)
                self.assertIsInstance(legal_moves, list)
                self.assertGreater(len(legal_moves), 0)
                self.assertFalse(game.is_game_over())

    def test_all_games_have_scorers(self):
        """Test that all games have differential scorers"""
        scorers = [
            (CheckersScorer(), CheckersBoard(), Color.RED),
            (GoScorer(), GoBoard(size=9), GoColor.BLACK),
            (OthelloScorer(), OthelloBoard(), OthelloColor.BLACK),
            (Connect4Scorer(), Connect4Board(), Connect4Color.RED),
            (GomokuScorer(), GomokuBoard(size=15), GomokuColor.BLACK),
            (HexScorer(), HexBoard(size=11), HexColor.RED),
            (DotsBoxesScorer(), DotsBoxesBoard(), DBColor.RED),
            (BreakthroughScorer(), BreakthroughBoard(), BreakthroughColor.WHITE),
        ]

        for scorer, board, color in scorers:
            with self.subTest(scorer=type(scorer).__name__):
                # All scorers should score a game as (score, result)
                self.assertTrue(hasattr(scorer, 'calculate_final_score'))
                score, result = scorer.calculate_final_score(board, color, 0)
                self.assertIsInstance(score, (int, float))
                # (HexScorer reports an unfinished game, which cannot be a draw in Hex, as 'in_progress')
                self.assertIn(result, ('win', 'loss', 'draw', 'in_progress'))


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
