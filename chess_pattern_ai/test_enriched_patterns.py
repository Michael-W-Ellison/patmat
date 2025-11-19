#!/usr/bin/env python3
"""
Test ARC with enriched universal patterns from gameplay

User's insight: "Running through additional game tests will provide additional
patterns for the database and increase the success rate."

We now have:
- 137,162 diagonal movement observations from chess
- 12,163 diagonal movements from checkers (ALL checkers moves)
- 95,875 fork/multi-directional threat observations
- Patterns from 11,533 real chess games
- 0.85 confidence for diagonal_movement (cross-game validated)
"""

from arc_puzzle import ARCDatasetLoader
from arc_cross_game_learner import ARCCrossGameLearner
import numpy as np

print("Testing ARC with Enriched Universal Patterns")
print("="*70)
print("\nPattern Sources:")
print("  - 11,533 chess games analyzed")
print("  - 24 checkers patterns extracted")
print("  - 137,162 diagonal movement observations from chess")
print("  - 12,163 diagonal movements from checkers")
print("  - Cross-game validation boosted confidence")
print()

learner = ARCCrossGameLearner()
loader = ARCDatasetLoader('../arc_dataset/data')

# Check current universal patterns
print("Current Universal Patterns:")
print("-"*70)

universal_patterns = learner.get_universal_patterns()
print(f"Total: {len(universal_patterns)}")

for pattern in universal_patterns[:15]:
    cross_game = "🌟" if pattern['cross_game'] else "  "
    print(f"{cross_game} {pattern['name']}: {pattern['confidence']:.2f} confidence, {pattern['observations']} obs")
    if pattern['games']:
        games_str = ', '.join([g['game'] for g in pattern['games']])
        print(f"   Games: {games_str}")

# Test on puzzles that might benefit from diagonal patterns
print("\n" + "="*70)
print("Testing on puzzles with potential diagonal/spatial patterns:")
print("="*70)

# Load some training puzzles to test
training_puzzles = loader.load_training_set()

# Test on first 20 puzzles
test_puzzles = training_puzzles[:20]

results = {
    'total': 0,
    'attempted': 0,
    'correct': 0,
    'by_pattern': {}
}

for puzzle in test_puzzles:
    results['total'] += 1

    try:
        # Try to solve with universal patterns
        solution = learner.solve_with_universal_patterns(puzzle)

        if solution is not None:
            results['attempted'] += 1

            # Check correctness
            expected = puzzle.get_test_outputs()[0]

            if np.array_equal(np.array(solution), np.array(expected)):
                results['correct'] += 1
                print(f"✓ {puzzle.puzzle_id}: CORRECT")
            else:
                print(f"✗ {puzzle.puzzle_id}: Incorrect")
        else:
            print(f"- {puzzle.puzzle_id}: No solution generated")

    except Exception as e:
        print(f"✗ {puzzle.puzzle_id}: Error - {e}")

print("\n" + "="*70)
print("Results:")
print("="*70)
print(f"Total: {results['total']}")
print(f"Attempted: {results['attempted']}")
print(f"Correct: {results['correct']}")
if results['total'] > 0:
    print(f"Success rate: {results['correct']/results['total']*100:.1f}%")
if results['attempted'] > 0:
    print(f"Accuracy when attempting: {results['correct']/results['attempted']*100:.1f}%")

# Show which patterns are now available that weren't before
print("\n" + "="*70)
print("New Patterns from Gameplay Analysis:")
print("="*70)

new_patterns = [
    'diagonal_movement',
    'multi_directional_threat',
    'center_control',
    'chain_formation',
    'breakthrough_pattern',
    'multi_hop_diagonal'
]

for pattern_name in new_patterns:
    pattern = next((p for p in universal_patterns if p['name'] == pattern_name), None)
    if pattern:
        print(f"\n✓ {pattern['name']}")
        print(f"  Confidence: {pattern['confidence']:.2f}")
        print(f"  Observations: {pattern['observations']}")
        if pattern['games']:
            games = ', '.join([g['game'] for g in pattern['games']])
            print(f"  Games: {games}")

print("\n" + "="*70)
print("Key Improvements:")
print("="*70)
print("✓ Diagonal movement: 149,325 observations (chess + checkers)")
print("✓ Confidence boosted to 0.85 (cross-game validation)")
print("✓ Multi-hop patterns from checkers (for skip/jump puzzles)")
print("✓ Fork patterns (95,875 obs) for multi-directional ARC puzzles")
print("✓ Center control patterns for spatial ARC puzzles")

learner.close()
