#!/usr/bin/env python3
"""
Test cross-game pattern detection on specific puzzles

Debug why reflection patterns aren't being detected
"""

import numpy as np
from arc_puzzle import ARCDatasetLoader
from arc_meta_pattern_learner import ARCMetaPatternLearner

loader = ARCDatasetLoader('../arc_dataset/data')
learner = ARCMetaPatternLearner()

# Test reflection detection on specific puzzles
test_puzzles = [
    ('0962bcdd', 'Should be reflection'),
    ('6150a2bd', 'Should be reflection'),
]

print("Testing feature extraction and pattern classification:")
print("="*70)

for puzzle_id, expected in test_puzzles:
    puzzle = loader.load_puzzle(puzzle_id, 'training')

    print(f"\nPuzzle {puzzle_id}: {expected}")
    print("-"*70)

    train_pairs = puzzle.get_train_pairs()

    for idx, (inp, out) in enumerate(train_pairs):
        print(f"\nExample {idx+1}:")

        # Extract features
        features = learner.extract_features(inp, out)

        print(f"  Input shape: {features['input_shape']}")
        print(f"  Output shape: {features['output_shape']}")
        print(f"  Size changed: {features.get('size_changed', False)}")

        if not features.get('size_changed'):
            print(f"  Change percentage: {features.get('change_percentage', 0):.1f}%")
            print(f"  Horizontal reflection: {features.get('horizontal_reflection', False)}")
            print(f"  Vertical reflection: {features.get('vertical_reflection', False)}")
            print(f"  Rotation 90: {features.get('rotation_90', False)}")
            print(f"  Rotation 180: {features.get('rotation_180', False)}")
            print(f"  Rotation 270: {features.get('rotation_270', False)}")

            # Check manually
            inp_arr = np.array(inp)
            out_arr = np.array(out)

            h_flip = np.array_equal(out_arr, np.fliplr(inp_arr))
            v_flip = np.array_equal(out_arr, np.flipud(inp_arr))
            rot_90 = np.array_equal(out_arr, np.rot90(inp_arr, 1))
            rot_180 = np.array_equal(out_arr, np.rot90(inp_arr, 2))

            print(f"\n  Manual verification:")
            print(f"    Horizontal flip: {h_flip}")
            print(f"    Vertical flip: {v_flip}")
            print(f"    Rotation 90: {rot_90}")
            print(f"    Rotation 180: {rot_180}")

        # Classify
        pattern_type = learner.classify_transformation(features)
        print(f"\n  Classified as: {pattern_type}")

learner.close()
