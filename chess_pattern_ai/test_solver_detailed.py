#!/usr/bin/env python3
"""
Detailed ARCSolver testing with diagnostics
"""

from arc_solver import ARCSolver
from arc_puzzle import ARCDatasetLoader
import numpy as np


def main():
    print("="*70)
    print("ARC Solver - Detailed Diagnostics")
    print("="*70)

    loader = ARCDatasetLoader('../arc_dataset/data')
    solver = ARCSolver('arc_learned_full.db')

    # Test on puzzles we know have patterns
    puzzle_ids = ['007bbfb7', '10fcaaa3', '46442a0e', '4522001f', 'c3e719e8']

    for puzzle_id in puzzle_ids:
        try:
            puzzle = loader.load_puzzle(puzzle_id, 'training')
            print(f"\n{'='*70}")
            print(f"Puzzle: {puzzle_id}")
            print(f"{'='*70}")

            # Show train examples
            train_pairs = puzzle.get_train_pairs()
            print(f"\nTrain examples: {len(train_pairs)}")
            for i, (inp, out) in enumerate(train_pairs[:2], 1):  # Show first 2
                print(f"  Example {i}: {len(inp)}x{len(inp[0])} → {len(out)}x{len(out[0])}")

            # Detect pattern
            detected = solver._detect_pattern_from_examples(train_pairs)
            if detected:
                print(f"\nDetected pattern:")
                print(f"  Type: {detected['type']}")
                for key, val in detected.items():
                    if key != 'type':
                        print(f"  {key}: {val}")
            else:
                print("\n  ✗ No pattern detected")
                continue

            # Get test input/output
            test_input = puzzle.get_test_inputs()[0]
            expected_output = puzzle.get_test_outputs()[0] if puzzle.get_test_outputs() else None

            print(f"\nTest:")
            print(f"  Input: {len(test_input)}x{len(test_input[0])}")
            if expected_output:
                print(f"  Expected output: {len(expected_output)}x{len(expected_output[0])}")

            # Apply pattern
            predicted = solver._apply_pattern(test_input, detected)

            if predicted:
                print(f"  Predicted output: {len(predicted)}x{len(predicted[0])}")

                if expected_output:
                    # Check if correct
                    if predicted == expected_output:
                        print(f"\n  ✓ CORRECT!")
                    else:
                        print(f"\n  ✗ Incorrect")
                        # Show difference
                        pred_np = np.array(predicted)
                        exp_np = np.array(expected_output)

                        if pred_np.shape == exp_np.shape:
                            diff = np.sum(pred_np != exp_np)
                            total = pred_np.size
                            print(f"  Cells different: {diff}/{total} ({diff/total*100:.1f}%)")
                        else:
                            print(f"  Shape mismatch: {pred_np.shape} vs {exp_np.shape}")
            else:
                print(f"  ✗ Failed to apply pattern")

        except Exception as e:
            print(f"Error: {e}")

    solver.close()
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
