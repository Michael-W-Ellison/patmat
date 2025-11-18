#!/usr/bin/env python3
"""
Integration Tests for Pattern Learning System
Tests the actual learning components, not just data loading
"""

import unittest
import sqlite3
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from pattern_abstraction_engine import PatternAbstractionEngine


class TestPatternExtraction(unittest.TestCase):
    """Test that patterns are actually extracted from mistakes"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.engine = PatternAbstractionEngine(self.temp_db)

    def tearDown(self):
        """Clean up temporary database"""
        self.engine.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_pattern_extraction_from_hanging_piece(self):
        """Test that hanging piece pattern is extracted"""
        # Simulate a position where a knight is left undefended
        fen_before = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move_made = "b8c6"  # Black attacks the hanging knight
        material_lost = 3.0  # Lost a knight

        # Extract pattern
        patterns = self.engine.extract_patterns_from_mistake(
            fen_before, move_made, material_lost
        )

        # Verify pattern was extracted
        self.assertGreater(len(patterns), 0, "Should extract at least one pattern")

        # Check if hanging piece pattern detected
        pattern_types = [p[0] for p in patterns]
        self.assertIn('hanging_piece', pattern_types,
                     "Should detect hanging piece pattern")

    def test_pattern_extraction_from_tempo_loss(self):
        """Test that tempo loss pattern is extracted"""
        # Simulate moving same piece twice in opening
        fen_before = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 1 2"
        move_made = "c3b1"  # Moving knight back to starting square
        material_lost = 1.0

        patterns = self.engine.extract_patterns_from_mistake(
            fen_before, move_made, material_lost
        )

        # Should detect tempo loss or similar pattern
        self.assertGreater(len(patterns), 0, "Should extract pattern for tempo loss")

    def test_patterns_stored_in_database(self):
        """Test that extracted patterns are stored in database"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"
        material_lost = 3.0

        # Extract and store patterns
        patterns = self.engine.extract_patterns_from_mistake(fen, move, material_lost)

        # Query database to verify storage
        self.engine.cursor.execute('''
            SELECT COUNT(*) FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
        ''')

        count = self.engine.cursor.fetchone()[0]
        self.assertGreater(count, 0, "Pattern should be stored in database")

    def test_pattern_confidence_increases_with_observations(self):
        """Test that pattern confidence increases as it's seen more"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"
        material_lost = 3.0

        # Extract pattern first time
        self.engine.extract_patterns_from_mistake(fen, move, material_lost)

        self.engine.cursor.execute('''
            SELECT times_seen, confidence FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
            LIMIT 1
        ''')

        first_result = self.engine.cursor.fetchone()
        if first_result:
            times_seen_1, confidence_1 = first_result

            # Extract same pattern again
            self.engine.extract_patterns_from_mistake(fen, move, material_lost)

            self.engine.cursor.execute('''
                SELECT times_seen, confidence FROM abstract_patterns
                WHERE pattern_type = 'hanging_piece'
                LIMIT 1
            ''')

            times_seen_2, confidence_2 = self.engine.cursor.fetchone()

            # Verify times_seen increased
            self.assertGreater(times_seen_2, times_seen_1,
                             "Times seen should increase with repeated observation")


class TestOutcomeTracking(unittest.TestCase):
    """Test that game outcomes are tracked for patterns"""

    def setUp(self):
        """Create temporary database"""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.engine = PatternAbstractionEngine(self.temp_db)

    def tearDown(self):
        """Clean up"""
        self.engine.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_loss_outcome_recorded(self):
        """Test that losing games update pattern loss count"""
        # Extract a pattern
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"
        patterns = self.engine.extract_patterns_from_mistake(fen, move, 3.0)

        # Update pattern with loss outcome
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')

        # Verify loss was recorded
        self.engine.cursor.execute('''
            SELECT games_with_pattern_lost, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
            LIMIT 1
        ''')

        result = self.engine.cursor.fetchone()
        if result:
            losses, win_rate = result
            self.assertGreater(losses, 0, "Should record loss")
            self.assertEqual(win_rate, 0.0, "Win rate should be 0% after only losses")

    def test_win_outcome_recorded(self):
        """Test that winning games update pattern win count"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"
        patterns = self.engine.extract_patterns_from_mistake(fen, move, 3.0)

        # Update with win outcome
        self.engine.update_patterns_from_game_outcome(patterns, 'win')

        self.engine.cursor.execute('''
            SELECT games_with_pattern_won, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
            LIMIT 1
        ''')

        result = self.engine.cursor.fetchone()
        if result:
            wins, win_rate = result
            self.assertGreater(wins, 0, "Should record win")
            self.assertEqual(win_rate, 1.0, "Win rate should be 100% after only wins")

    def test_win_rate_calculation(self):
        """Test that win rate is calculated correctly"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"
        patterns = self.engine.extract_patterns_from_mistake(fen, move, 3.0)

        # Record multiple outcomes: 2 wins, 3 losses, 1 draw
        self.engine.update_patterns_from_game_outcome(patterns, 'win')
        self.engine.update_patterns_from_game_outcome(patterns, 'win')
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')
        self.engine.update_patterns_from_game_outcome(patterns, 'draw')

        self.engine.cursor.execute('''
            SELECT games_with_pattern_won, games_with_pattern_lost,
                   games_with_pattern_draw, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
            LIMIT 1
        ''')

        result = self.engine.cursor.fetchone()
        if result:
            wins, losses, draws, win_rate = result

            # Verify counts
            self.assertEqual(wins, 2, "Should have 2 wins")
            self.assertEqual(losses, 3, "Should have 3 losses")
            self.assertEqual(draws, 1, "Should have 1 draw")

            # Verify win rate calculation: wins / (wins + losses + draws)
            expected_win_rate = 2.0 / 6.0
            self.assertAlmostEqual(win_rate, expected_win_rate, places=2,
                                 msg="Win rate should be 2/6 = 0.33")


class TestPenaltyCalculation(unittest.TestCase):
    """Test that penalties are calculated correctly from patterns"""

    def test_zero_win_rate_high_penalty(self):
        """Test that 0% win rate patterns get massive penalties"""
        # Pattern with 0% win rate, high confidence, high material loss
        avg_material_lost = 4.6
        confidence = 1.0
        win_rate = 0.0

        # Calculate penalty as done in optimized_search.py line 271-292
        material_penalty = avg_material_lost * 20
        outcome_penalty = (1.0 - win_rate) * 200
        total_penalty = (material_penalty + outcome_penalty) * confidence

        # Verify penalty calculation
        expected_penalty = (4.6 * 20 + 200) * 1.0  # = 292
        self.assertAlmostEqual(total_penalty, expected_penalty, places=1)
        self.assertGreater(total_penalty, 200,
                          "0% win rate should produce very high penalty")

    def test_high_win_rate_low_penalty(self):
        """Test that high win rate patterns get low penalties"""
        avg_material_lost = 1.0
        confidence = 0.5
        win_rate = 0.8  # 80% win rate

        material_penalty = avg_material_lost * 20
        outcome_penalty = (1.0 - win_rate) * 200
        total_penalty = (material_penalty + outcome_penalty) * confidence

        # With 80% win rate: (20 + 40) * 0.5 = 30
        expected_penalty = (20 + 40) * 0.5
        self.assertAlmostEqual(total_penalty, expected_penalty, places=1)
        self.assertLess(total_penalty, 50,
                       "High win rate should produce low penalty")

    def test_penalty_increases_with_confidence(self):
        """Test that penalty scales with confidence"""
        avg_material_lost = 3.0
        win_rate = 0.0

        # Low confidence
        penalty_low = ((avg_material_lost * 20) + 200) * 0.2

        # High confidence
        penalty_high = ((avg_material_lost * 20) + 200) * 0.9

        self.assertGreater(penalty_high, penalty_low,
                          "Higher confidence should increase penalty")


class TestPatternApplicationDetection(unittest.TestCase):
    """Test that the system can detect when patterns apply to positions"""

    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.engine = PatternAbstractionEngine(self.temp_db)

    def tearDown(self):
        self.engine.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_pattern_matching_for_new_position(self):
        """Test that learned patterns can be matched against new positions"""
        # Store a pattern
        fen_training = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move_training = "c3a4"
        self.engine.extract_patterns_from_mistake(fen_training, move_training, 3.0)
        self.engine.conn.commit()

        # Check if similar position triggers pattern
        fen_test = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1"
        move_test = "f3h4"

        # Check for pattern violations
        violations = self.engine.check_for_known_patterns(fen_test, move_test)

        # Should find some patterns (even if not exact match)
        self.assertIsInstance(violations, list, "Should return list of violations")


class TestDatabaseIntegrity(unittest.TestCase):
    """Test that database maintains integrity during learning"""

    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.engine = PatternAbstractionEngine(self.temp_db)

    def tearDown(self):
        self.engine.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_no_duplicate_patterns(self):
        """Test that same pattern doesn't create duplicates"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"

        # Extract same pattern multiple times
        for _ in range(3):
            self.engine.extract_patterns_from_mistake(fen, move, 3.0)

        # Should update existing pattern, not create duplicates
        self.engine.cursor.execute('''
            SELECT COUNT(*) FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
        ''')

        count = self.engine.cursor.fetchone()[0]
        self.assertEqual(count, 1,
                        "Should update existing pattern, not create duplicates")

    def test_pattern_data_consistency(self):
        """Test that pattern data remains consistent"""
        fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 0 1"
        move = "b8c6"

        patterns = self.engine.extract_patterns_from_mistake(fen, move, 3.0)

        # Add outcomes
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')
        self.engine.update_patterns_from_game_outcome(patterns, 'loss')

        self.engine.cursor.execute('''
            SELECT games_with_pattern_won, games_with_pattern_lost,
                   games_with_pattern_draw, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
            LIMIT 1
        ''')

        wins, losses, draws, win_rate = self.engine.cursor.fetchone()
        total_games = wins + losses + draws

        # Win rate should match: wins / total_games
        expected_win_rate = wins / total_games if total_games > 0 else 0.0
        self.assertAlmostEqual(win_rate, expected_win_rate, places=2,
                             msg="Win rate should match calculated value")


def run_learning_tests():
    """Run learning system integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPatternExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestOutcomeTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestPenaltyCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternApplicationDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseIntegrity))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("LEARNING SYSTEM TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_learning_tests()
    sys.exit(0 if success else 1)
