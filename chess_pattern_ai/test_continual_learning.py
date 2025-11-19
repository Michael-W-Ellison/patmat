#!/usr/bin/env python3
"""
Test Continual Learning: Verify that solver improves over time

This test demonstrates that ARCSolverWithLearning actually improves
its performance as it solves more puzzles, unlike the static ARCSolver.
"""

from arc_solver import ARCSolver, ARCSolverWithLearning
from arc_puzzle import ARCDatasetLoader
import os

def test_continual_learning():
    """Test that learning solver improves over time"""

    print("="*70)
    print("CONTINUAL LEARNING TEST")
    print("="*70)

    loader = ARCDatasetLoader('../arc_dataset/data')

    # Create a fresh database for learning
    learning_db = 'arc_learned_continual.db'
    if os.path.exists(learning_db):
        os.remove(learning_db)

    # Copy the base database to start from same knowledge
    import shutil
    shutil.copy('arc_learned_full.db', learning_db)

    # Initialize learning solver
    solver = ARCSolverWithLearning(learning_db)

    # Load all training puzzles
    puzzles = loader.load_training_set()

    print(f"\nLoaded {len(puzzles)} training puzzles")
    print("\nPhase 1: Learning from first 100 puzzles")
    print("-"*70)

    # Phase 1: Learn from first 100 puzzles
    correct_phase1 = 0
    attempted_phase1 = 0

    for i, puzzle in enumerate(puzzles[:100]):
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = solver.solve_and_learn(puzzle, expected[0])

        if result:
            attempted_phase1 += 1
            if result == expected[0]:
                correct_phase1 += 1

        if (i + 1) % 25 == 0:
            rate = correct_phase1 / attempted_phase1 * 100 if attempted_phase1 > 0 else 0
            print(f"  After {i+1} puzzles: {correct_phase1}/{attempted_phase1} ({rate:.1f}%)")

    phase1_rate = correct_phase1 / attempted_phase1 * 100 if attempted_phase1 > 0 else 0
    print(f"\nPhase 1 final: {correct_phase1}/{attempted_phase1} ({phase1_rate:.1f}%)")

    # Show learning stats
    print("\n" + "="*70)
    print("Learning Statistics After Phase 1:")
    print("="*70)

    stats = solver.get_learning_stats()
    for pattern_type, data in sorted(stats.items(), key=lambda x: x[1]['avg_confidence'], reverse=True):
        print(f"\n{pattern_type}:")
        print(f"  Patterns tracked: {data['patterns_tracked']}")
        print(f"  Avg confidence: {data['avg_confidence']:.3f}")
        print(f"  Successes: {data['total_successes']}")
        print(f"  Failures: {data['total_failures']}")
        print(f"  Success rate: {data['success_rate']*100:.1f}%")

    learned_count = solver.get_learned_solutions_count()
    print(f"\nLearned solutions stored: {learned_count}")

    # Phase 2: Test on next 100 puzzles
    print("\n" + "="*70)
    print("Phase 2: Testing on next 100 puzzles (with learned knowledge)")
    print("-"*70)

    correct_phase2 = 0
    attempted_phase2 = 0

    for i, puzzle in enumerate(puzzles[100:200]):
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = solver.solve_and_learn(puzzle, expected[0])

        if result:
            attempted_phase2 += 1
            if result == expected[0]:
                correct_phase2 += 1

        if (i + 1) % 25 == 0:
            rate = correct_phase2 / attempted_phase2 * 100 if attempted_phase2 > 0 else 0
            print(f"  After {100+i+1} puzzles total: {correct_phase2}/{attempted_phase2} ({rate:.1f}%)")

    phase2_rate = correct_phase2 / attempted_phase2 * 100 if attempted_phase2 > 0 else 0
    print(f"\nPhase 2 final: {correct_phase2}/{attempted_phase2} ({phase2_rate:.1f}%)")

    # Phase 3: Test on final 100 puzzles
    print("\n" + "="*70)
    print("Phase 3: Testing on final puzzles (200-300)")
    print("-"*70)

    correct_phase3 = 0
    attempted_phase3 = 0

    for i, puzzle in enumerate(puzzles[200:300]):
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = solver.solve_and_learn(puzzle, expected[0])

        if result:
            attempted_phase3 += 1
            if result == expected[0]:
                correct_phase3 += 1

        if (i + 1) % 25 == 0:
            rate = correct_phase3 / attempted_phase3 * 100 if attempted_phase3 > 0 else 0
            print(f"  After {200+i+1} puzzles total: {correct_phase3}/{attempted_phase3} ({rate:.1f}%)")

    phase3_rate = correct_phase3 / attempted_phase3 * 100 if attempted_phase3 > 0 else 0
    print(f"\nPhase 3 final: {correct_phase3}/{attempted_phase3} ({phase3_rate:.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Learning Progression")
    print("="*70)

    print(f"\nPhase 1 (puzzles 0-100):   {phase1_rate:5.1f}% ({correct_phase1}/{attempted_phase1})")
    print(f"Phase 2 (puzzles 100-200): {phase2_rate:5.1f}% ({correct_phase2}/{attempted_phase2})")
    print(f"Phase 3 (puzzles 200-300): {phase3_rate:5.1f}% ({correct_phase3}/{attempted_phase3})")

    # Calculate improvement
    if phase1_rate > 0:
        improvement_2 = phase2_rate - phase1_rate
        improvement_3 = phase3_rate - phase1_rate

        print(f"\nImprovement from Phase 1:")
        print(f"  Phase 2: {improvement_2:+.1f} percentage points")
        print(f"  Phase 3: {improvement_3:+.1f} percentage points")

        if improvement_3 > 0:
            print(f"\n✅ SUCCESS: Solver improved by {improvement_3:.1f}% through learning!")
        elif improvement_3 > -1:
            print(f"\n⚠️  MARGINAL: Solver maintained performance (±1%)")
        else:
            print(f"\n❌ CONCERN: Solver performance decreased by {abs(improvement_3):.1f}%")

    # Final learning stats
    print("\n" + "="*70)
    print("Final Learning Statistics:")
    print("="*70)

    stats = solver.get_learning_stats()
    for pattern_type, data in sorted(stats.items(), key=lambda x: x[1]['avg_confidence'], reverse=True):
        print(f"{pattern_type:15s}: conf={data['avg_confidence']:.3f}, success_rate={data['success_rate']*100:5.1f}%")

    learned_count = solver.get_learned_solutions_count()
    print(f"\nTotal learned solutions: {learned_count}")

    solver.close()

    print("\n" + "="*70)


def compare_static_vs_learning():
    """Compare static solver vs learning solver"""

    print("\n" + "="*70)
    print("COMPARISON: Static vs Learning Solver")
    print("="*70)

    loader = ARCDatasetLoader('../arc_dataset/data')
    puzzles = loader.load_training_set()

    # Test static solver
    print("\nTesting STATIC solver on 50 puzzles...")
    static_solver = ARCSolver('arc_learned_full.db')

    static_correct = 0
    static_attempted = 0

    for puzzle in puzzles[:50]:
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = static_solver.solve(puzzle)

        if result:
            static_attempted += 1
            if result == expected[0]:
                static_correct += 1

    static_rate = static_correct / static_attempted * 100 if static_attempted > 0 else 0
    static_solver.close()

    print(f"Static solver: {static_correct}/{static_attempted} ({static_rate:.1f}%)")

    # Test learning solver
    print("\nTesting LEARNING solver on same 50 puzzles...")

    learning_db = 'arc_learned_compare.db'
    if os.path.exists(learning_db):
        os.remove(learning_db)

    import shutil
    shutil.copy('arc_learned_full.db', learning_db)

    learning_solver = ARCSolverWithLearning(learning_db)

    learning_correct = 0
    learning_attempted = 0

    for puzzle in puzzles[:50]:
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = learning_solver.solve_and_learn(puzzle, expected[0])

        if result:
            learning_attempted += 1
            if result == expected[0]:
                learning_correct += 1

    learning_rate = learning_correct / learning_attempted * 100 if learning_attempted > 0 else 0

    print(f"Learning solver: {learning_correct}/{learning_attempted} ({learning_rate:.1f}%)")

    # Now test on NEXT 50 puzzles
    print("\nTesting on NEXT 50 puzzles (50-100)...")

    static_solver = ARCSolver('arc_learned_full.db')
    static_correct2 = 0
    static_attempted2 = 0

    for puzzle in puzzles[50:100]:
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = static_solver.solve(puzzle)

        if result:
            static_attempted2 += 1
            if result == expected[0]:
                static_correct2 += 1

    static_rate2 = static_correct2 / static_attempted2 * 100 if static_attempted2 > 0 else 0
    static_solver.close()

    print(f"Static solver: {static_correct2}/{static_attempted2} ({static_rate2:.1f}%) - NO CHANGE")

    learning_correct2 = 0
    learning_attempted2 = 0

    for puzzle in puzzles[50:100]:
        expected = puzzle.get_test_outputs()
        if not expected:
            continue

        result = learning_solver.solve_and_learn(puzzle, expected[0])

        if result:
            learning_attempted2 += 1
            if result == expected[0]:
                learning_correct2 += 1

    learning_rate2 = learning_correct2 / learning_attempted2 * 100 if learning_attempted2 > 0 else 0

    print(f"Learning solver: {learning_correct2}/{learning_attempted2} ({learning_rate2:.1f}%) - ", end="")

    if learning_rate2 > learning_rate + 1:
        print("IMPROVED! ✅")
    elif learning_rate2 < learning_rate - 1:
        print("Decreased")
    else:
        print("Similar")

    learning_solver.close()

    print("\n" + "="*70)


if __name__ == '__main__':
    test_continual_learning()
    compare_static_vs_learning()
