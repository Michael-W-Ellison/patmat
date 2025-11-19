#!/usr/bin/env python3
"""
Test solver on simpler, non-scaling patterns
"""

from arc_solver import ARCSolver
from arc_puzzle import ARCDatasetLoader
import numpy as np


def find_puzzles_by_pattern_type(loader, pattern_types, max_puzzles=20):
    """Find puzzles that use specific pattern types"""
    puzzles = loader.load_training_set()
    solver = ARCSolver('arc_learned_full.db')

    matches = []
    for puzzle in puzzles:
        train_pairs = puzzle.get_train_pairs()
        detected = solver._detect_pattern_from_examples(train_pairs)

        if detected and detected['type'] in pattern_types:
            matches.append((puzzle, detected))
            if len(matches) >= max_puzzles:
                break

    solver.close()
    return matches


def main():
    print("="*70)
    print("Testing Solver on Simple Patterns")
    print("="*70)

    loader = ARCDatasetLoader('../arc_dataset/data')

    # Test different pattern types
    pattern_types_to_test = [
        (['spatial'], "Spatial Transformations (rotations/flips)"),
        (['symmetry'], "Symmetry Operations"),
        (['color_mapping'], "Color Remapping"),
        (['tiling'], "Tiling"),
    ]

    overall_results = {}

    for pattern_types, description in pattern_types_to_test:
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"{'='*70}")

        matches = find_puzzles_by_pattern_type(loader, pattern_types, max_puzzles=5)

        if not matches:
            print(f"  No puzzles found with {pattern_types} patterns")
            continue

        print(f"\nFound {len(matches)} puzzles")

        solver = ARCSolver('arc_learned_full.db')
        correct = 0

        for i, (puzzle, detected_pattern) in enumerate(matches, 1):
            test_input = puzzle.get_test_inputs()[0]
            expected_output = puzzle.get_test_outputs()[0] if puzzle.get_test_outputs() else None

            predicted = solver._apply_pattern(test_input, detected_pattern)

            if predicted and expected_output:
                if predicted == expected_output:
                    print(f"  ✓ {puzzle.puzzle_id}: CORRECT")
                    correct += 1
                else:
                    pred_np = np.array(predicted)
                    exp_np = np.array(expected_output)
                    if pred_np.shape == exp_np.shape:
                        diff = np.sum(pred_np != exp_np)
                        total = pred_np.size
                        print(f"  ✗ {puzzle.puzzle_id}: {diff}/{total} cells wrong ({diff/total*100:.1f}%)")
                    else:
                        print(f"  ✗ {puzzle.puzzle_id}: Shape mismatch")
            elif not predicted:
                print(f"  ✗ {puzzle.puzzle_id}: Failed to apply pattern")
            else:
                print(f"  ? {puzzle.puzzle_id}: No expected output")

        rate = (correct / len(matches) * 100) if matches else 0
        print(f"\nSuccess rate: {correct}/{len(matches)} ({rate:.1f}%)")
        overall_results[description] = (correct, len(matches), rate)

        solver.close()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_correct = 0
    total_tested = 0

    for desc, (correct, total, rate) in overall_results.items():
        print(f"{desc:50s}: {correct}/{total} ({rate:5.1f}%)")
        total_correct += correct
        total_tested += total

    if total_tested > 0:
        overall_rate = total_correct / total_tested * 100
        print(f"{'='*70}")
        print(f"{'Overall':50s}: {total_correct}/{total_tested} ({overall_rate:5.1f}%)")


if __name__ == '__main__':
    main()
