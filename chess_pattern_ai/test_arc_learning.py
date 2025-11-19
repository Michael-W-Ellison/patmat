#!/usr/bin/env python3
"""
Test ARCObserver on larger dataset to validate pattern learning
"""

from arc_observer import ARCObserver
from arc_puzzle import ARCDatasetLoader


def main():
    print("="*70)
    print("ARC PATTERN LEARNING - Extended Test (50 puzzles)")
    print("="*70)

    # Initialize observer
    observer = ARCObserver('arc_learned_extended.db')

    # Load dataset
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Train on first 50 puzzles
    print("\nLoading first 50 training puzzles...")
    puzzles = loader.load_training_set()[:50]

    print(f"\nObserving {len(puzzles)} puzzles to learn patterns...\n")

    total_examples = 0
    for i, puzzle in enumerate(puzzles, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(puzzles)} puzzles...")
        observer.observe_puzzle(puzzle)
        total_examples += len(puzzle.train_examples)

    print(f"\nTotal examples observed: {total_examples}")

    # Display learned patterns
    observer.get_learned_patterns()

    # Test specific pattern types
    print("\n" + "="*70)
    print("Pattern Type Distribution:")
    print("="*70)

    observer.cursor.execute('''
        SELECT pattern_type, COUNT(*), SUM(times_seen)
        FROM transformation_patterns
        GROUP BY pattern_type
    ''')

    for pattern_type, unique_count, total_seen in observer.cursor.fetchall():
        print(f"{pattern_type:<20} {unique_count:>3} unique patterns, {total_seen:>3} total occurrences")

    print("\n" + "="*70)
    print("Phase 1 Extended Test Complete!")
    print("="*70)

    observer.close()


if __name__ == '__main__':
    main()
