#!/usr/bin/env python3
"""
Test Suite for Pattern Recognition Chess AI Evaluators
Uses TDD approach to verify all evaluators work correctly
"""

import unittest
import sqlite3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from material_evaluator import MaterialEvaluator
from safety_evaluator import SafetyEvaluator
from opening_evaluator import OpeningEvaluator
from game_phase_detector import GamePhaseDetector
from temporal_evaluator import TemporalEvaluator
from weak_square_detector import WeakSquareDetector
from position_abstractor import PositionAbstractor


class TestMaterialEvaluator(unittest.TestCase):
    """Test material evaluation using discovered piece values"""

    def setUp(self):
        self.evaluator = MaterialEvaluator("rule_discovery.db")
        self.evaluator._load_piece_values()

    def test_piece_values_loaded(self):
        """Test that piece values are loaded from database"""
        self.assertIsNotNone(self.evaluator.piece_values)
        self.assertGreater(len(self.evaluator.piece_values), 0)

        # Check expected pieces exist
        for piece in ['P', 'N', 'B', 'R', 'Q']:
            self.assertIn(piece, self.evaluator.piece_values)

    def test_starting_position_equal(self):
        """Test that starting position has equal material"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        score = self.evaluator.evaluate_material(fen)
        self.assertEqual(score, 0.0, "Starting position should have equal material")

    def test_white_ahead_material(self):
        """Test position where white is ahead in material"""
        # Black missing knight (white ahead)
        fen = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        score = self.evaluator.evaluate_material(fen)
        self.assertGreater(score, 0, "White should be ahead in material")

    def test_black_ahead_material(self):
        """Test position where black is ahead in material"""
        # White missing rook (black ahead)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"
        score = self.evaluator.evaluate_material(fen)
        self.assertLess(score, 0, "Black should be ahead in material")

    def test_endgame_material(self):
        """Test endgame position with minimal material"""
        # King and pawns only
        fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        score = self.evaluator.evaluate_material(fen)
        self.assertGreater(score, 0, "White should be ahead with extra pawn")

    def tearDown(self):
        self.evaluator.close()


class TestSafetyEvaluator(unittest.TestCase):
    """Test king safety evaluation"""

    def setUp(self):
        self.evaluator = SafetyEvaluator("rule_discovery.db")
        self.evaluator._load_safety_weights()

    def test_safety_weights_loaded(self):
        """Test that safety patterns are loaded"""
        # May be empty if not discovered yet, but should not error
        self.assertIsNotNone(self.evaluator.safety_weights)

    def test_starting_position_symmetrical(self):
        """Test that starting position has symmetrical safety"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        score = self.evaluator.evaluate_safety(fen)
        # Should be close to 0 (symmetrical position)
        self.assertAlmostEqual(score, 0.0, delta=5.0)

    def test_kings_found(self):
        """Test that king positions are correctly identified"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board_part = fen.split()[0]
        white_king, black_king = self.evaluator._find_kings(board_part)

        self.assertIsNotNone(white_king)
        self.assertIsNotNone(black_king)
        self.assertEqual(white_king, 4, "White king should be on e1 (square 4)")
        self.assertEqual(black_king, 60, "Black king should be on e8 (square 60)")

    def tearDown(self):
        self.evaluator.close()


class TestOpeningEvaluator(unittest.TestCase):
    """Test opening evaluation"""

    def setUp(self):
        self.evaluator = OpeningEvaluator("rule_discovery.db")
        self.evaluator._load_opening_weights()

    def test_opening_weights_loaded(self):
        """Test that opening patterns are loaded"""
        self.assertIsNotNone(self.evaluator.opening_weights)

    def test_starting_position_evaluated(self):
        """Test that starting position can be evaluated"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        score = self.evaluator.evaluate_opening(fen)
        # Should return a numeric score (may be 0)
        self.assertIsInstance(score, float)

    def test_opening_phase_only(self):
        """Test that evaluation only applies in opening phase"""
        # Move 20 - past opening
        fen = "r1bq1rk1/pp1pppbp/2n2np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 20"
        score = self.evaluator.evaluate_opening(fen)
        self.assertEqual(score, 0.0, "Should not evaluate past move 15")

    def test_center_control_detection(self):
        """Test center control pattern detection"""
        # Position with white controlling center
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        board_part = fen.split()[0]
        has_control = self.evaluator._has_center_control(board_part)
        self.assertIsInstance(has_control, bool)

    def tearDown(self):
        self.evaluator.close()


class TestGamePhaseDetector(unittest.TestCase):
    """Test game phase detection"""

    def setUp(self):
        self.detector = GamePhaseDetector("rule_discovery.db")

    def test_starting_position_is_opening(self):
        """Test that starting position is detected as opening"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        phase = self.detector.detect_game_phase(fen, 1)
        self.assertEqual(phase, 'opening')

    def test_endgame_detection_few_pieces(self):
        """Test endgame detection with few pieces"""
        fen = "6k1/5ppp/8/3p4/3P4/8/5PPP/6K1 w - - 0 40"
        phase = self.detector.detect_game_phase(fen, 40)
        self.assertEqual(phase, 'endgame')

    def test_middlegame_detection(self):
        """Test middlegame detection"""
        # Move 20 with moderate pieces - should be middlegame
        fen = "r1bqr1k1/pp3pbp/2np1np1/4p3/2B1P3/2NP1N2/PPPQ1PPP/R1B1R1K1 w - - 0 20"
        phase = self.detector.detect_game_phase(fen, 20)
        self.assertEqual(phase, 'middlegame')

    def test_phase_weights_returned(self):
        """Test that phase weights are returned correctly"""
        weights = self.detector.get_phase_weights('opening')

        self.assertIsInstance(weights, dict)
        self.assertIn('material', weights)
        self.assertIn('mobility', weights)
        self.assertIn('safety', weights)

        # Opening should emphasize safety
        self.assertGreater(weights['safety'], weights['material'])

    def test_endgame_weights_emphasize_material(self):
        """Test that endgame emphasizes material"""
        weights = self.detector.get_phase_weights('endgame')

        # Endgame should emphasize material and pawn structure
        self.assertGreater(weights['material'], 1.0)
        self.assertGreater(weights['pawn_structure'], 1.0)

    def tearDown(self):
        self.detector.close()


class TestTemporalEvaluator(unittest.TestCase):
    """Test temporal (phase-based) evaluation"""

    def setUp(self):
        self.evaluator = TemporalEvaluator("rule_discovery.db")

    def test_phase_detection(self):
        """Test that phase detection works"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        phase = self.evaluator._detect_simple_phase(fen, 1)
        self.assertEqual(phase, 'opening')

    def test_score_combination(self):
        """Test that scores are combined with phase weights"""
        scores = {
            'material': 100.0,
            'mobility': 50.0,
            'safety': 30.0,
            'pawn_structure': 20.0,
            'positional': 40.0,
            'tactical': 60.0
        }

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        total = self.evaluator.evaluate_with_phase_adaptation(scores, fen, 1)

        self.assertIsInstance(total, float)
        self.assertNotEqual(total, sum(scores.values()),
                          "Weighted sum should differ from simple sum")

    def test_different_phases_different_scores(self):
        """Test that different phases produce different evaluations"""
        scores = {
            'material': 100.0,
            'mobility': 50.0,
            'safety': 30.0,
            'pawn_structure': 20.0,
            'positional': 40.0,
            'tactical': 60.0
        }

        opening_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        endgame_fen = "6k1/5ppp/8/3p4/3P4/8/5PPP/6K1 w - - 0 40"

        opening_score = self.evaluator.evaluate_with_phase_adaptation(scores, opening_fen, 1)
        endgame_score = self.evaluator.evaluate_with_phase_adaptation(scores, endgame_fen, 40)

        self.assertNotEqual(opening_score, endgame_score,
                          "Opening and endgame should weight components differently")

    def tearDown(self):
        self.evaluator.close()


class TestPositionAbstractor(unittest.TestCase):
    """Test position abstraction utilities"""

    def setUp(self):
        self.abstractor = PositionAbstractor()

    def test_abstract_starting_position(self):
        """Test abstracting the starting position"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = self.abstractor.abstract_position(fen)

        self.assertIsInstance(features, dict)
        self.assertIn('material_balance', features)
        self.assertIn('piece_count', features)
        self.assertIn('king_safety', features)
        self.assertIn('center_control', features)
        self.assertIn('development', features)
        self.assertIn('pawn_structure', features)

    def test_material_balance_equal(self):
        """Test material balance in equal position"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = self.abstractor.abstract_position(fen)
        self.assertEqual(features['material_balance'], 0)

    def test_piece_count(self):
        """Test piece counting"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = self.abstractor.abstract_position(fen)

        counts = features['piece_count']
        self.assertEqual(counts['P'], 16, "Should have 16 pawns")
        self.assertEqual(counts['N'], 4, "Should have 4 knights")
        self.assertEqual(counts['K'], 2, "Should have 2 kings")

    def test_king_safety_features(self):
        """Test king safety feature extraction"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = self.abstractor.abstract_position(fen)

        safety = features['king_safety']
        self.assertIn('white_castled', safety)
        self.assertIn('black_castled', safety)
        self.assertIn('white_can_castle', safety)
        self.assertIn('black_can_castle', safety)

        # Starting position: not castled but can castle
        self.assertFalse(safety['white_castled'])
        self.assertTrue(safety['white_can_castle'])


class TestDatabaseIntegration(unittest.TestCase):
    """Test that evaluators can read from the actual database"""

    def setUp(self):
        self.db_path = "rule_discovery.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def test_database_exists(self):
        """Test that the database file exists and is accessible"""
        self.assertTrue(os.path.exists(self.db_path))

    def test_discovered_piece_values_table(self):
        """Test that discovered piece values table exists and has data"""
        self.cursor.execute("""
            SELECT COUNT(*) FROM discovered_piece_values
        """)
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0, "Should have discovered piece values")

    def test_abstract_patterns_table(self):
        """Test that abstract patterns table exists"""
        self.cursor.execute("""
            SELECT COUNT(*) FROM abstract_patterns
        """)
        count = self.cursor.fetchone()[0]
        # May be 0 if no patterns learned yet, but table should exist
        self.assertGreaterEqual(count, 0)

    def test_games_table_has_data(self):
        """Test that games have been played"""
        self.cursor.execute("""
            SELECT COUNT(*) FROM games
        """)
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0, "Should have game data for learning")

    def test_learned_patterns_structure(self):
        """Test that abstract patterns have correct structure"""
        self.cursor.execute("""
            SELECT pattern_type, pattern_description, times_seen,
                   avg_material_lost, win_rate
            FROM abstract_patterns
            LIMIT 1
        """)

        result = self.cursor.fetchone()
        if result:  # If patterns exist
            pattern_type, desc, times_seen, mat_lost, win_rate = result

            self.assertIsInstance(pattern_type, str)
            self.assertIsInstance(times_seen, int)
            self.assertIsInstance(win_rate, (int, float))
            self.assertGreaterEqual(win_rate, 0.0)
            self.assertLessEqual(win_rate, 1.0)

    def tearDown(self):
        self.conn.close()


def run_test_suite():
    """Run the complete test suite"""

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMaterialEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestOpeningEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestGamePhaseDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionAbstractor))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
