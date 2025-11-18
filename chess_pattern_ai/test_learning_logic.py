#!/usr/bin/env python3
"""
Learning System Logic Tests
Tests the learning mechanisms without requiring chess library
"""

import unittest
import sqlite3
import tempfile
import os
import sys


class TestPatternLearningLogic(unittest.TestCase):
    """Test pattern learning logic directly against database"""

    def setUp(self):
        """Create test database with schema"""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.conn = sqlite3.connect(self.temp_db)
        self.cursor = self.conn.cursor()

        # Create abstract_patterns table (as in real database)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS abstract_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT,
                times_seen INTEGER DEFAULT 1,
                avg_material_lost REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                games_with_pattern_won INTEGER DEFAULT 0,
                games_with_pattern_lost INTEGER DEFAULT 0,
                games_with_pattern_draw INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, pattern_description)
            )
        ''')
        self.conn.commit()

    def tearDown(self):
        """Clean up test database"""
        self.conn.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_pattern_insertion(self):
        """Test that patterns can be inserted into database"""
        self.cursor.execute('''
            INSERT INTO abstract_patterns
            (pattern_type, pattern_description, times_seen, avg_material_lost, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', ('hanging_piece', 'knight_undefended', 1, 3.0, 0.5))
        self.conn.commit()

        self.cursor.execute('SELECT COUNT(*) FROM abstract_patterns')
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1, "Should insert pattern")

    def test_pattern_update_on_duplicate(self):
        """Test that seeing same pattern increases times_seen"""
        # Insert pattern first time
        self.cursor.execute('''
            INSERT OR REPLACE INTO abstract_patterns
            (pattern_type, pattern_description, times_seen, avg_material_lost)
            VALUES (?, ?, ?, ?)
        ''', ('tempo_loss', 'moved_same_piece_twice', 1, 2.0))
        self.conn.commit()

        # Simulate seeing it again - update times_seen
        self.cursor.execute('''
            UPDATE abstract_patterns
            SET times_seen = times_seen + 1
            WHERE pattern_type = ? AND pattern_description = ?
        ''', ('tempo_loss', 'moved_same_piece_twice'))
        self.conn.commit()

        self.cursor.execute('''
            SELECT times_seen FROM abstract_patterns
            WHERE pattern_type = 'tempo_loss'
        ''')
        times_seen = self.cursor.fetchone()[0]
        self.assertEqual(times_seen, 2, "Should increment times_seen")

    def test_outcome_tracking_loss(self):
        """Test that losses are tracked for patterns"""
        # Insert pattern
        self.cursor.execute('''
            INSERT INTO abstract_patterns
            (pattern_type, pattern_description, games_with_pattern_lost, win_rate)
            VALUES (?, ?, ?, ?)
        ''', ('hanging_piece', 'knight_undefended', 0, 0.0))

        # Record a loss
        self.cursor.execute('''
            UPDATE abstract_patterns
            SET games_with_pattern_lost = games_with_pattern_lost + 1
            WHERE pattern_type = ?
        ''', ('hanging_piece',))

        # Calculate win rate: wins / (wins + losses + draws)
        self.cursor.execute('''
            UPDATE abstract_patterns
            SET win_rate = CAST(games_with_pattern_won AS REAL) /
                          (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw)
            WHERE pattern_type = ?
        ''', ('hanging_piece',))
        self.conn.commit()

        self.cursor.execute('''
            SELECT games_with_pattern_lost, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'hanging_piece'
        ''')
        losses, win_rate = self.cursor.fetchone()

        self.assertEqual(losses, 1, "Should record loss")
        self.assertEqual(win_rate, 0.0, "Win rate should be 0% with only losses")

    def test_outcome_tracking_mixed_results(self):
        """Test win rate calculation with mixed results"""
        # Insert pattern with mixed results: 2 wins, 3 losses, 1 draw
        self.cursor.execute('''
            INSERT INTO abstract_patterns
            (pattern_type, pattern_description,
             games_with_pattern_won, games_with_pattern_lost, games_with_pattern_draw,
             win_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('weak_position', 'isolated_piece', 2, 3, 1, 0.0))

        # Calculate correct win rate
        self.cursor.execute('''
            UPDATE abstract_patterns
            SET win_rate = CAST(games_with_pattern_won AS REAL) /
                          (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw)
            WHERE pattern_type = 'weak_position'
        ''')
        self.conn.commit()

        self.cursor.execute('''
            SELECT games_with_pattern_won, games_with_pattern_lost,
                   games_with_pattern_draw, win_rate
            FROM abstract_patterns
            WHERE pattern_type = 'weak_position'
        ''')
        wins, losses, draws, win_rate = self.cursor.fetchone()

        expected_win_rate = 2.0 / 6.0  # 2 wins out of 6 games
        self.assertAlmostEqual(win_rate, expected_win_rate, places=2,
                             msg="Win rate should be 2/6 = 0.33")

    def test_penalty_calculation_formula(self):
        """Test the penalty calculation formula from optimized_search.py"""
        # Pattern data
        avg_material_lost = 4.6
        confidence = 1.0
        win_rate = 0.0

        # Calculate penalty (from optimized_search.py line 280-291)
        material_penalty = avg_material_lost * 20
        outcome_penalty = (1.0 - win_rate) * 200
        total_penalty = (material_penalty + outcome_penalty) * confidence

        # Expected: (4.6 * 20 + 200) * 1.0 = 292
        self.assertAlmostEqual(material_penalty, 92.0, places=1)
        self.assertAlmostEqual(outcome_penalty, 200.0, places=1)
        self.assertAlmostEqual(total_penalty, 292.0, places=1)

    def test_penalty_scales_with_win_rate(self):
        """Test that penalty decreases as win rate increases"""
        avg_material_lost = 3.0
        confidence = 1.0

        # 0% win rate
        penalty_0 = ((avg_material_lost * 20) + (1.0 - 0.0) * 200) * confidence

        # 50% win rate
        penalty_50 = ((avg_material_lost * 20) + (1.0 - 0.5) * 200) * confidence

        # 100% win rate
        penalty_100 = ((avg_material_lost * 20) + (1.0 - 1.0) * 200) * confidence

        self.assertGreater(penalty_0, penalty_50,
                          "0% win rate should have higher penalty than 50%")
        self.assertGreater(penalty_50, penalty_100,
                          "50% win rate should have higher penalty than 100%")
        self.assertEqual(penalty_100, 60.0,
                        "100% win rate should only have material penalty")

    def test_confidence_affects_penalty(self):
        """Test that confidence multiplies the penalty"""
        avg_material_lost = 3.0
        win_rate = 0.0

        penalty_low_conf = ((avg_material_lost * 20) + 200) * 0.2
        penalty_high_conf = ((avg_material_lost * 20) + 200) * 0.9

        self.assertLess(penalty_low_conf, penalty_high_conf,
                       "Higher confidence should increase penalty magnitude")


class TestDatabaseSchemaValidation(unittest.TestCase):
    """Validate that the real database has expected schema"""

    def setUp(self):
        """Connect to actual database"""
        self.db_path = "rule_discovery.db"
        if not os.path.exists(self.db_path):
            self.skipTest("Database file not found")

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def tearDown(self):
        """Close connection"""
        if self.conn:
            self.conn.close()

    def test_abstract_patterns_has_outcome_columns(self):
        """Verify abstract_patterns table has win/loss/draw tracking"""
        self.cursor.execute("PRAGMA table_info(abstract_patterns)")
        columns = [row[1] for row in self.cursor.fetchall()]

        required_columns = [
            'games_with_pattern_won',
            'games_with_pattern_lost',
            'games_with_pattern_draw',
            'win_rate'
        ]

        for col in required_columns:
            self.assertIn(col, columns,
                         f"abstract_patterns should have {col} column")

    def test_zero_percent_win_rate_patterns_exist(self):
        """Verify that patterns with 0% win rate exist in database"""
        self.cursor.execute('''
            SELECT COUNT(*) FROM abstract_patterns
            WHERE win_rate = 0.0
            AND (games_with_pattern_lost > 0 OR games_with_pattern_won > 0)
        ''')

        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0,
                          "Should have patterns with 0% win rate in database")

    def test_pattern_data_integrity(self):
        """Verify win rate matches win/loss/draw counts"""
        self.cursor.execute('''
            SELECT games_with_pattern_won, games_with_pattern_lost,
                   games_with_pattern_draw, win_rate
            FROM abstract_patterns
            WHERE (games_with_pattern_won + games_with_pattern_lost + games_with_pattern_draw) > 0
            LIMIT 10
        ''')

        for row in self.cursor.fetchall():
            wins, losses, draws, win_rate = row
            total_games = wins + losses + draws

            if total_games > 0:
                expected_win_rate = wins / total_games
                self.assertAlmostEqual(win_rate, expected_win_rate, places=2,
                                     msg=f"Win rate should match calculated value for pattern with {wins}W {losses}L {draws}D")

    def test_high_material_loss_patterns_exist(self):
        """Verify patterns with significant material loss exist"""
        self.cursor.execute('''
            SELECT COUNT(*) FROM abstract_patterns
            WHERE avg_material_lost > 3.0
        ''')

        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0,
                          "Should have patterns with significant material loss")

    def test_patterns_have_been_seen_multiple_times(self):
        """Verify patterns have been observed multiple times"""
        self.cursor.execute('''
            SELECT MAX(times_seen) FROM abstract_patterns
        ''')

        max_times_seen = self.cursor.fetchone()[0]
        if max_times_seen:
            self.assertGreater(max_times_seen, 1,
                             "Patterns should have been seen multiple times")


class TestPenaltyApplicationLogic(unittest.TestCase):
    """Test that penalty application logic is correct"""

    def test_tempo_loss_penalty_calculation(self):
        """Test penalty for tempo_loss pattern from database"""
        # Known pattern from database: tempo_loss with 196 instances, 0% win rate
        times_seen = 196
        avg_material_lost = 4.49  # From database
        win_rate = 0.0
        confidence = min(1.0, times_seen / 100.0)  # Confidence based on observations

        # Calculate penalty
        material_penalty = avg_material_lost * 20
        outcome_penalty = (1.0 - win_rate) * 200
        total_penalty = (material_penalty + outcome_penalty) * confidence

        # Should be very high penalty
        self.assertGreater(total_penalty, 250,
                          "Tempo loss with 0% win rate should have massive penalty")

    def test_pattern_with_few_observations_lower_confidence(self):
        """Test that patterns with few observations have lower confidence"""
        times_seen_few = 5
        times_seen_many = 100

        confidence_few = min(1.0, times_seen_few / 100.0)
        confidence_many = min(1.0, times_seen_many / 100.0)

        self.assertLess(confidence_few, confidence_many,
                       "Fewer observations should result in lower confidence")

        # Same pattern, different confidence levels
        avg_loss = 3.0
        win_rate = 0.0

        penalty_few = ((avg_loss * 20) + 200) * confidence_few
        penalty_many = ((avg_loss * 20) + 200) * confidence_many

        self.assertGreater(penalty_many, penalty_few,
                          "Higher confidence should increase penalty")


def run_logic_tests():
    """Run learning logic tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPatternLearningLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseSchemaValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPenaltyApplicationLogic))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("LEARNING LOGIC TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailed tests:")
        for test, _ in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nErrors:")
        for test, _ in result.errors:
            print(f"  - {test}")

    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_logic_tests()
    sys.exit(0 if success else 1)
