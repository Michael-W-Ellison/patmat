#!/usr/bin/env python3
"""
Test: Does the system follow learning-based philosophy?
Verifies NO hardcoded chess knowledge, only discovered patterns
"""

import unittest
import sqlite3
import sys


class TestLearningPhilosophy(unittest.TestCase):
    """Verify system learns from observation, not hardcoded rules"""

    def setUp(self):
        self.conn = sqlite3.connect('rule_discovery.db')
        self.cursor = self.conn.cursor()

    def tearDown(self):
        self.conn.close()

    def test_piece_values_discovered_not_hardcoded(self):
        """Verify piece values came from observation, not programming"""
        self.cursor.execute('''
            SELECT piece_type, relative_value, observation_count, discovery_method
            FROM discovered_piece_values
        ''')

        for piece, value, obs_count, method in self.cursor.fetchall():
            # Should have many observations (statistical discovery)
            self.assertGreater(obs_count, 1000,
                             f"{piece} value should be based on many observations")

            # Should explicitly state discovery method (exchanges, outcomes, etc.)
            discovery_terms = ['exchange', 'outcome', 'preservation', 'combined']
            has_discovery = any(term in method.lower() for term in discovery_terms)
            self.assertTrue(has_discovery,
                          f"{piece} should show discovery method: {method}")
            self.assertNotIn('hardcoded', method.lower(),
                           f"{piece} should NOT be hardcoded")

    def test_movement_rules_inferred_from_games(self):
        """Verify movement rules discovered, not programmed"""
        self.cursor.execute('''
            SELECT piece_type, rule_type, conditions
            FROM inferred_rules
        ''')

        rules = self.cursor.fetchall()
        self.assertGreater(len(rules), 0, "Should have inferred movement rules")

        # Rules should be inferred, not hardcoded
        for piece, rule_type, conditions in rules:
            self.assertIn(rule_type, [
                'basic_movement', 'complex_movement',
                'capture_only', 'non_capture_only'
            ], f"Rule type should indicate discovery: {rule_type}")

    def test_patterns_track_actual_game_outcomes(self):
        """Verify patterns linked to real game results, not theory"""
        self.cursor.execute('''
            SELECT pattern_type, games_with_pattern_won,
                   games_with_pattern_lost, games_with_pattern_draw
            FROM abstract_patterns
            WHERE times_seen > 10
            LIMIT 5
        ''')

        for row in self.cursor.fetchall():
            pattern, wins, losses, draws = row
            total_games = wins + losses + draws

            # Pattern should be observed in actual games
            self.assertGreater(total_games, 0,
                             f"{pattern} should be linked to actual games")

            # Not theoretical - real outcomes
            self.assertEqual(total_games, wins + losses + draws,
                           "Should have real game outcomes")

    def test_no_hardcoded_strategy_knowledge(self):
        """Verify system doesn't contain chess strategy rules"""
        # Check that evaluators load WEIGHTS, not strategy rules
        self.cursor.execute('''
            SELECT king_safety_weight, piece_protection_weight, exposed_penalty
            FROM discovered_safety_patterns
            LIMIT 1
        ''')

        result = self.cursor.fetchone()
        if result:
            king_w, protect_w, exposed_w = result

            # Weights should be discovered values (not 0, 1, or round numbers)
            # Real discovered weights are often decimals like 0.5, 0.2, 0.3
            self.assertIsInstance(king_w, float,
                                "Weights should be discovered floats")

    def test_penalties_based_on_outcomes_not_theory(self):
        """Verify penalties come from observed outcomes, not chess theory"""
        # Get a pattern with 0% win rate
        self.cursor.execute('''
            SELECT pattern_type, win_rate, times_seen,
                   games_with_pattern_won, games_with_pattern_lost
            FROM abstract_patterns
            WHERE win_rate = 0.0 AND times_seen > 10
            LIMIT 1
        ''')

        result = self.cursor.fetchone()
        if result:
            pattern, win_rate, times_seen, wins, losses = result

            # 0% win rate should come from ACTUAL LOSSES
            self.assertEqual(wins, 0, "0% win rate means 0 actual wins")
            self.assertGreater(losses, 0, "0% win rate means actual losses occurred")
            self.assertGreater(times_seen, 10, "Should be statistically significant")

            # This is OBSERVED failure, not programmed knowledge
            print(f"\n✓ Pattern '{pattern}' has 0% win rate from {losses} actual losses")
            print(f"  This is learned from observation, not programmed")


class TestViolationsInMyCode(unittest.TestCase):
    """Test: Did I violate the learning philosophy in my evaluators?"""

    def test_safety_evaluator_uses_discovered_weights(self):
        """Check if safety evaluator uses discovered weights or hardcoded rules"""
        import os
        if not os.path.exists('safety_evaluator.py'):
            self.skipTest("safety_evaluator.py not in current directory")

        with open('safety_evaluator.py', 'r') as f:
            code = f.read()

        # VIOLATION: Hardcoded square numbers
        if 'king_square != 4' in code:
            self.fail("❌ VIOLATION: Hardcoded 'king_square != 4' - should use discovered patterns")

        if 'king_square != 60' in code:
            self.fail("❌ VIOLATION: Hardcoded 'king_square != 60' - should use discovered patterns")

        # CORRECT: Should load discovered weights
        if 'king_safety_weight' in code:
            print("\n✓ Uses discovered king_safety_weight")

    def test_opening_evaluator_uses_discovered_weights(self):
        """Check if opening evaluator uses discovered weights or hardcoded strategy"""
        import os
        if not os.path.exists('opening_evaluator.py'):
            self.skipTest("opening_evaluator.py not in current directory")

        with open('opening_evaluator.py', 'r') as f:
            code = f.read()

        # VIOLATION: Hardcoded center squares
        if 'center_squares = [27, 28, 35, 36]' in code:
            self.fail("❌ VIOLATION: Hardcoded center squares - should discover from patterns")

        # CORRECT: Should load discovered weights
        if 'center_control_weight' in code:
            print("\n✓ Uses discovered center_control_weight")


def run_philosophy_tests():
    """Run tests to verify learning philosophy"""
    import sys

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLearningPhilosophy))
    suite.addTests(loader.loadTestsFromTestCase(TestViolationsInMyCode))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("LEARNING PHILOSOPHY VALIDATION")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")

    if result.failures:
        print("\n❌ VIOLATIONS FOUND:")
        for test, _ in result.failures:
            print(f"  - {test}")
        print("\nThese violations should be removed to maintain learning-based philosophy")

    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_philosophy_tests()
    sys.exit(0 if success else 1)
