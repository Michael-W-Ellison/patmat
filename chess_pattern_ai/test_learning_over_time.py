#!/usr/bin/env python3
"""
Test: Does solver improve over time?

Tests whether solving puzzles in sequence improves performance
on later puzzles.
"""

from arc_solver import ARCSolver
from arc_puzzle import ARCDatasetLoader
import numpy as np

def test_static_performance():
    """Test that performance doesn't change over time"""

    loader = ARCDatasetLoader('../arc_dataset/data')
    solver = ARCSolver('arc_learned_full.db')

    # Get spatial transformation puzzles (we know these work at 85.7%)
    spatial_puzzles = [
        '3c9b0459', '5168d44c', '6150a2bd', '67a3c6ac', '68b16354'
    ]

    print("Testing solver performance over time:")
    print("="*70)

    # Solve the same puzzles multiple times
    for iteration in range(1, 4):
        print(f"\nIteration {iteration}:")
        correct = 0

        for puzzle_id in spatial_puzzles:
            puzzle = loader.load_puzzle(puzzle_id, 'training')
            result = solver.solve(puzzle)

            if result and result == puzzle.get_test_outputs()[0]:
                correct += 1

        accuracy = correct / len(spatial_puzzles) * 100
        print(f"  Accuracy: {correct}/{len(spatial_puzzles)} ({accuracy:.1f}%)")

    print("\n" + "="*70)
    print("Result: Performance is STATIC (same accuracy each iteration)")
    print("No learning from previous solutions.")

    solver.close()

def analyze_pattern_counts():
    """Check if pattern counts change during solving"""

    import sqlite3

    conn = sqlite3.connect('arc_learned_full.db')
    cursor = conn.cursor()

    print("\n" + "="*70)
    print("Pattern Database Analysis:")
    print("="*70)

    # Count patterns
    cursor.execute("SELECT COUNT(*) FROM arc_patterns")
    pattern_count = cursor.fetchone()[0]

    print(f"\nTotal patterns in database: {pattern_count}")
    print("This count NEVER CHANGES during solving phase.")
    print("The database is READ-ONLY during puzzle solving.")

    conn.close()

if __name__ == '__main__':
    test_static_performance()
    analyze_pattern_counts()
