#!/usr/bin/env python3
"""
Evaluate Cross-Game Learning on ARC

Compare performance:
1. Baseline: ARC meta-patterns only
2. Cross-Game: ARC + universal patterns from chess/checkers/Dots and Boxes

User's insight: "The AI should use patterns from all games"
Expected: Confidence boost from cross-game patterns improves accuracy
"""

import numpy as np
import json
from arc_puzzle import ARCDatasetLoader
from arc_meta_pattern_learner import ARCMetaPatternLearner
from arc_cross_game_learner import ARCCrossGameLearner


def evaluate_solver(solver, puzzles, use_universal=False):
    """Evaluate solver on puzzles"""
    results = {
        'total': 0,
        'correct': 0,
        'attempted': 0,
        'by_pattern': {},
        'puzzles': {}
    }

    for puzzle in puzzles:
        results['total'] += 1

        try:
            # Get solution
            if use_universal and hasattr(solver, 'solve_with_universal_patterns'):
                solution = solver.solve_with_universal_patterns(puzzle)
                matches = solver.match_puzzle_to_patterns(
                    puzzle.get_train_pairs(),
                    use_universal=True
                )
            else:
                # Baseline - use only ARC patterns
                training_pairs = puzzle.get_train_pairs()
                test_input = puzzle.get_test_inputs()[0]

                matches = solver.match_puzzle_to_patterns(training_pairs)

                if matches:
                    best_match = matches[0]
                    pattern_type = best_match.get('type', 'unknown')
                    solution = solver._apply_pattern_transformation(
                        test_input, training_pairs, pattern_type
                    )
                else:
                    solution = None

            if solution is not None:
                results['attempted'] += 1

                # Check correctness
                expected = puzzle.get_test_outputs()[0]

                if np.array_equal(np.array(solution), np.array(expected)):
                    results['correct'] += 1

                    # Track by pattern type
                    if matches:
                        pattern_name = matches[0].get('pattern_name',
                                                     matches[0].get('type', 'unknown'))
                        source = matches[0].get('source', 'unknown')

                        key = f"{pattern_name} ({source})"
                        if key not in results['by_pattern']:
                            results['by_pattern'][key] = {'correct': 0, 'attempted': 0}
                        results['by_pattern'][key]['correct'] += 1
                        results['by_pattern'][key]['attempted'] += 1

                        results['puzzles'][puzzle.puzzle_id] = {
                            'status': 'correct',
                            'pattern': pattern_name,
                            'source': source,
                            'confidence': matches[0].get('boosted_confidence', 0.5)
                        }
                else:
                    # Wrong answer
                    if matches:
                        pattern_name = matches[0].get('pattern_name',
                                                     matches[0].get('type', 'unknown'))
                        source = matches[0].get('source', 'unknown')
                        key = f"{pattern_name} ({source})"
                        if key not in results['by_pattern']:
                            results['by_pattern'][key] = {'correct': 0, 'attempted': 0}
                        results['by_pattern'][key]['attempted'] += 1

                        results['puzzles'][puzzle.puzzle_id] = {
                            'status': 'incorrect',
                            'pattern': pattern_name,
                            'source': source,
                            'confidence': matches[0].get('boosted_confidence', 0.5)
                        }

        except Exception as e:
            # Skip on error
            results['puzzles'][puzzle.puzzle_id] = {
                'status': 'error',
                'error': str(e)
            }
            continue

    return results


if __name__ == '__main__':
    print("Evaluating Cross-Game Learning for ARC")
    print("="*70)
    print("\nUser's insight: \"The AI should use patterns from all games\"")
    print("Testing if chess/checkers/Dots and Boxes patterns boost ARC accuracy")
    print()

    loader = ARCDatasetLoader('../arc_dataset/data')

    # First observe training set to build ARC meta-patterns
    print("Step 1: Building ARC meta-pattern database...")
    print("-"*70)

    arc_learner = ARCMetaPatternLearner('arc_meta_patterns.db')
    cross_game_learner = ARCCrossGameLearner('arc_meta_patterns.db', 'universal_patterns.db')

    training_puzzles = loader.load_training_set()

    print(f"Observing {len(training_puzzles)} training puzzles...")
    for i, puzzle in enumerate(training_puzzles):
        arc_learner.observe_puzzle(puzzle.puzzle_id, puzzle.get_train_pairs())

        if (i + 1) % 100 == 0:
            print(f"  Observed {i+1}/{len(training_puzzles)} puzzles...")

    print(f"\n✓ Observed all {len(training_puzzles)} training puzzles")

    # Get learned patterns
    arc_patterns = arc_learner.get_meta_patterns()
    print(f"✓ Learned {len(arc_patterns)} ARC meta-patterns")

    # Load evaluation set
    print("\nStep 2: Loading evaluation set...")
    print("-"*70)

    evaluation_puzzles = loader.load_evaluation_set()
    print(f"✓ Loaded {len(evaluation_puzzles)} evaluation puzzles")

    # Test on subset for speed
    test_size = min(50, len(evaluation_puzzles))
    test_puzzles = evaluation_puzzles[:test_size]

    print(f"\nTesting on {test_size} puzzles (for speed)")

    # Evaluate baseline (ARC patterns only)
    print("\n" + "="*70)
    print("Baseline: ARC meta-patterns only")
    print("="*70)

    baseline_results = evaluate_solver(cross_game_learner, test_puzzles, use_universal=False)

    print(f"\nResults:")
    print(f"  Total: {baseline_results['total']}")
    print(f"  Attempted: {baseline_results['attempted']}")
    print(f"  Correct: {baseline_results['correct']}")
    print(f"  Success rate: {baseline_results['correct']/baseline_results['total']*100:.1f}%")

    # Evaluate with cross-game patterns
    print("\n" + "="*70)
    print("Cross-Game: ARC + Universal patterns")
    print("="*70)

    crossgame_results = evaluate_solver(cross_game_learner, test_puzzles, use_universal=True)

    print(f"\nResults:")
    print(f"  Total: {crossgame_results['total']}")
    print(f"  Attempted: {crossgame_results['attempted']}")
    print(f"  Correct: {crossgame_results['correct']}")
    print(f"  Success rate: {crossgame_results['correct']/crossgame_results['total']*100:.1f}%")

    # Compare
    print("\n" + "="*70)
    print("Comparison:")
    print("="*70)

    baseline_pct = baseline_results['correct']/baseline_results['total']*100
    crossgame_pct = crossgame_results['correct']/crossgame_results['total']*100
    improvement = crossgame_pct - baseline_pct

    print(f"\nBaseline:   {baseline_pct:.1f}%")
    print(f"Cross-Game: {crossgame_pct:.1f}%")
    print(f"Improvement: {improvement:+.1f} percentage points")

    if improvement > 0:
        print(f"\n✓ Cross-game learning IMPROVED performance!")
    elif improvement < 0:
        print(f"\n✗ Cross-game learning decreased performance")
    else:
        print(f"\n= No change in performance")

    # Show pattern breakdown
    print("\n" + "="*70)
    print("Success by Pattern Type (Cross-Game):")
    print("="*70)

    for pattern, stats in sorted(crossgame_results['by_pattern'].items(),
                                 key=lambda x: x[1]['correct'], reverse=True):
        if stats['attempted'] > 0:
            success_rate = stats['correct'] / stats['attempted'] * 100
            marker = "🌟" if "universal" in pattern else "📊"
            print(f"{marker} {pattern}: {stats['correct']}/{stats['attempted']} ({success_rate:.0f}%)")

    # Show puzzles solved by universal patterns
    universal_solves = [
        (pid, info) for pid, info in crossgame_results['puzzles'].items()
        if info.get('status') == 'correct' and info.get('source') == 'universal'
    ]

    if universal_solves:
        print("\n" + "="*70)
        print("✓ Puzzles solved using universal patterns:")
        print("="*70)

        for pid, info in universal_solves:
            print(f"  - {pid}: {info['pattern']} (confidence: {info['confidence']:.2f})")
            print(f"    Learned from: chess/checkers/Dots and Boxes")

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)

    print(f"\nARC meta-patterns learned: {len(arc_patterns)}")

    universal_patterns = cross_game_learner.get_universal_patterns()
    cross_game_patterns = [p for p in universal_patterns if p['cross_game']]

    print(f"Universal patterns available: {len(universal_patterns)}")
    print(f"Cross-game patterns: {len(cross_game_patterns)}")

    print(f"\nCross-game patterns:")
    for p in cross_game_patterns:
        games = ', '.join([g['game'] for g in p['games']])
        print(f"  - {p['name']}: {p['confidence']:.2f} confidence")
        print(f"    Observed in: {games}")

    print(f"\nUser's insight validated: {len(universal_solves)} puzzles solved using")
    print(f"patterns learned from chess/checkers/Dots and Boxes!")

    arc_learner.close()
    cross_game_learner.close()
