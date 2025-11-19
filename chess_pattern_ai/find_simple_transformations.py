#!/usr/bin/env python3
"""
Find ARC puzzles with simple transformations matching universal patterns

Look for:
- Simple horizontal reflection (like chess starting position)
- Simple vertical reflection
- Simple rotation
"""

import numpy as np
from arc_puzzle import ARCDatasetLoader
from arc_meta_pattern_learner import ARCMetaPatternLearner

loader = ARCDatasetLoader('../arc_dataset/data')
learner = ARCMetaPatternLearner()

# Load all training puzzles
puzzles = loader.load_training_set()

simple_reflections = []
simple_rotations = []
simple_fills = []

print("Scanning ARC training set for simple transformations...")
print("="*70)

for puzzle in puzzles[:100]:  # Check first 100 puzzles
    train_pairs = puzzle.get_train_pairs()

    if not train_pairs:
        continue

    inp, out = train_pairs[0]
    features = learner.extract_features(inp, out)

    # Check for simple transformations
    if features.get('horizontal_reflection'):
        simple_reflections.append((puzzle.puzzle_id, 'horizontal', features))
    elif features.get('vertical_reflection'):
        simple_reflections.append((puzzle.puzzle_id, 'vertical', features))
    elif features.get('rotation_90') or features.get('rotation_180') or features.get('rotation_270'):
        rot_type = 'rotation_90' if features.get('rotation_90') else ('rotation_180' if features.get('rotation_180') else 'rotation_270')
        simple_rotations.append((puzzle.puzzle_id, rot_type, features))
    elif features.get('change_percentage', 100) < 10 and 'fill' in learner.classify_transformation(features).lower():
        simple_fills.append((puzzle.puzzle_id, 'fill', features))

print(f"\nSimple Reflections: {len(simple_reflections)}")
for pid, rtype, _ in simple_reflections[:5]:
    print(f"  - {pid}: {rtype}")

print(f"\nSimple Rotations: {len(simple_rotations)}")
for pid, rtype, _ in simple_rotations[:5]:
    print(f"  - {pid}: {rtype}")

print(f"\nSimple Fills: {len(simple_fills)}")
for pid, ftype, feats in simple_fills[:5]:
    print(f"  - {pid}: {feats.get('change_percentage', 0):.1f}% change")

learner.close()

# Test cross-game matching on a simple reflection if found
if simple_reflections:
    print("\n" + "="*70)
    print("Testing cross-game pattern matching on simple reflection:")
    print("="*70)

    from arc_cross_game_learner import ARCCrossGameLearner

    cross_learner = ARCCrossGameLearner()

    puzzle_id, ref_type, _ = simple_reflections[0]
    puzzle = loader.load_puzzle(puzzle_id, 'training')

    print(f"\nPuzzle {puzzle_id} ({ref_type} reflection):")

    matches = cross_learner.match_puzzle_to_patterns(
        puzzle.get_train_pairs(),
        use_universal=True
    )

    print(f"\nTop matches:")
    for i, match in enumerate(matches[:3], 1):
        source_marker = "🌟" if match['source'] == 'universal' else "📊"
        pattern_name = match.get('pattern_name', match.get('type', 'unknown'))

        print(f"  {i}. {source_marker} {pattern_name}")
        print(f"     Source: {match['source']}")
        print(f"     Confidence: {match.get('boosted_confidence', 0.5):.2f}")

        if match['source'] == 'universal':
            print(f"     Games: {', '.join(match.get('games', []))}")
            print(f"     ✓ MATCHED TO CHESS/CHECKERS PATTERN!")

    cross_learner.close()
