#!/usr/bin/env python3
"""
Full Continual Learning Test: All 400 training puzzles

Demonstrates improvement over the entire training set
"""

from arc_solver import ARCSolverWithLearning
from arc_puzzle import ARCDatasetLoader
import os

def test_full_learning():
    """Test learning on all 400 puzzles"""

    print("="*70)
    print("FULL CONTINUAL LEARNING TEST - 400 PUZZLES")
    print("="*70)

    loader = ARCDatasetLoader('../arc_dataset/data')

    # Create fresh learning database
    learning_db = 'arc_learned_continual_full.db'
    if os.path.exists(learning_db):
        os.remove(learning_db)

    import shutil
    shutil.copy('arc_learned_full.db', learning_db)

    solver = ARCSolverWithLearning(learning_db)

    puzzles = loader.load_training_set()
    print(f"\nLoaded {len(puzzles)} puzzles")

    # Track performance in buckets
    bucket_size = 50
    num_buckets = len(puzzles) // bucket_size

    bucket_results = []

    for bucket_idx in range(num_buckets):
        start_idx = bucket_idx * bucket_size
        end_idx = start_idx + bucket_size

        correct = 0
        attempted = 0

        for puzzle in puzzles[start_idx:end_idx]:
            expected = puzzle.get_test_outputs()
            if not expected:
                continue

            result = solver.solve_and_learn(puzzle, expected[0])

            if result:
                attempted += 1
                if result == expected[0]:
                    correct += 1

        rate = correct / attempted * 100 if attempted > 0 else 0
        bucket_results.append({
            'bucket': bucket_idx + 1,
            'range': f"{start_idx}-{end_idx}",
            'correct': correct,
            'attempted': attempted,
            'rate': rate
        })

        print(f"Bucket {bucket_idx+1} (puzzles {start_idx:3d}-{end_idx:3d}): {correct:2d}/{attempted:2d} ({rate:5.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("LEARNING CURVE:")
    print("="*70)

    for i, result in enumerate(bucket_results):
        bar_length = int(result['rate'] / 2)  # Scale to fit
        bar = "█" * bar_length
        print(f"Bucket {result['bucket']:2d}: {bar:40s} {result['rate']:5.1f}%")

    # Calculate overall improvement
    first_bucket_rate = bucket_results[0]['rate']
    last_bucket_rate = bucket_results[-1]['rate']
    improvement = last_bucket_rate - first_bucket_rate

    print("\n" + "="*70)
    print(f"First bucket (0-50):      {first_bucket_rate:5.1f}%")
    print(f"Last bucket (350-400):    {last_bucket_rate:5.1f}%")
    print(f"Improvement:              {improvement:+5.1f} percentage points")

    if improvement > 3:
        print(f"\n✅ SUCCESS: Significant improvement through learning!")
    elif improvement > 1:
        print(f"\n✅ MODERATE: Some improvement through learning")
    else:
        print(f"\n⚠️  LIMITED: Minimal improvement")

    # Learning stats
    print("\n" + "="*70)
    print("FINAL LEARNING STATISTICS:")
    print("="*70)

    stats = solver.get_learning_stats()
    for pattern_type, data in sorted(stats.items(), key=lambda x: x[1]['avg_confidence'], reverse=True):
        succ = data['total_successes']
        fail = data['total_failures']
        total = succ + fail
        conf = data['avg_confidence']
        rate = data['success_rate'] * 100

        print(f"{pattern_type:15s}: {succ:3d}/{total:3d} success ({rate:5.1f}%), confidence={conf:.3f}")

    learned_count = solver.get_learned_solutions_count()
    print(f"\nTotal learned solutions: {learned_count}")

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE:")
    print("="*70)

    total_correct = sum(r['correct'] for r in bucket_results)
    total_attempted = sum(r['attempted'] for r in bucket_results)
    overall_rate = total_correct / total_attempted * 100 if total_attempted > 0 else 0

    print(f"Total: {total_correct}/{total_attempted} ({overall_rate:.1f}%)")
    print(f"Baseline (static solver): 10.1%")
    print(f"Improvement: {overall_rate - 10.1:+.1f} percentage points")

    solver.close()

    print("\n" + "="*70)


if __name__ == '__main__':
    test_full_learning()
