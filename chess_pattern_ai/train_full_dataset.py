#!/usr/bin/env python3
"""
Train ARCObserver on the complete ARC-AGI training dataset

This will process all 400 training puzzles to build a comprehensive
pattern database.
"""

from arc_observer import ARCObserver
from arc_puzzle import ARCDatasetLoader
import time


def main():
    print("="*70)
    print("ARC PATTERN LEARNING - Full Dataset Training")
    print("="*70)

    # Initialize observer with new database for full training
    print("\nInitializing ARCObserver...")
    observer = ARCObserver('arc_learned_full.db')

    # Load complete dataset
    print("Loading complete training dataset...")
    loader = ARCDatasetLoader('../arc_dataset/data')

    puzzles = loader.load_training_set()
    print(f"Loaded {len(puzzles)} training puzzles")

    # Train on all puzzles
    print(f"\nObserving {len(puzzles)} puzzles to learn patterns...")
    print("This may take a minute...\n")

    start_time = time.time()
    total_examples = 0
    patterns_detected = 0

    for i, puzzle in enumerate(puzzles, 1):
        # Progress indicator every 50 puzzles
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"Progress: {i}/{len(puzzles)} puzzles ({rate:.1f} puzzles/sec)")

        # Count patterns before
        observer.cursor.execute('SELECT COUNT(*) FROM transformation_patterns')
        patterns_before = observer.cursor.fetchone()[0]

        # Observe puzzle
        observer.observe_puzzle(puzzle)

        # Count patterns after
        observer.cursor.execute('SELECT COUNT(*) FROM transformation_patterns')
        patterns_after = observer.cursor.fetchone()[0]

        if patterns_after > patterns_before:
            patterns_detected += (patterns_after - patterns_before)

        total_examples += len(puzzle.train_examples)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Processing rate: {len(puzzles)/elapsed_time:.1f} puzzles/sec")
    print(f"Total examples observed: {total_examples}")
    print(f"Average examples per puzzle: {total_examples/len(puzzles):.2f}")

    # Display learned patterns
    observer.get_learned_patterns()

    # Get detailed statistics
    print("\n" + "="*70)
    print("Detailed Pattern Analysis:")
    print("="*70)

    # Pattern type distribution
    observer.cursor.execute('''
        SELECT pattern_type, COUNT(*), SUM(times_seen)
        FROM transformation_patterns
        GROUP BY pattern_type
        ORDER BY SUM(times_seen) DESC
    ''')

    print("\nPattern Type Distribution:")
    print(f"{'Type':<20} {'Unique':<10} {'Total Seen':<15} {'Avg per Pattern':<15}")
    print("-" * 70)

    for pattern_type, unique_count, total_seen in observer.cursor.fetchall():
        avg = total_seen / unique_count if unique_count > 0 else 0
        print(f"{pattern_type:<20} {unique_count:<10} {total_seen:<15} {avg:<15.2f}")

    # Coverage analysis
    observer.cursor.execute('SELECT COUNT(*) FROM observed_transformations')
    total_observations = observer.cursor.fetchone()[0]

    observer.cursor.execute('''
        SELECT COUNT(*) FROM observed_transformations
        WHERE transformation_type IS NOT NULL
    ''')
    identified_transformations = observer.cursor.fetchone()[0]

    coverage = (identified_transformations / total_observations * 100) if total_observations > 0 else 0

    print(f"\n{'='*70}")
    print("Coverage Analysis:")
    print(f"{'='*70}")
    print(f"Total observations: {total_observations}")
    print(f"Identified patterns: {identified_transformations}")
    print(f"Unidentified: {total_observations - identified_transformations}")
    print(f"Coverage: {coverage:.1f}%")

    # Find most common patterns
    print(f"\n{'='*70}")
    print("Top 10 Most Common Patterns:")
    print(f"{'='*70}")

    observer.cursor.execute('''
        SELECT pattern_name, pattern_description, times_seen
        FROM transformation_patterns
        ORDER BY times_seen DESC
        LIMIT 10
    ''')

    print(f"{'Pattern':<30} {'Count':<10} {'Description':<40}")
    print("-" * 70)

    for name, desc, count in observer.cursor.fetchall():
        desc_short = desc[:37] + "..." if len(desc) > 40 else desc
        print(f"{name:<30} {count:<10} {desc_short:<40}")

    # Identify puzzles with no detected patterns
    observer.cursor.execute('''
        SELECT puzzle_id
        FROM observed_transformations
        WHERE transformation_type IS NULL
        GROUP BY puzzle_id
        LIMIT 20
    ''')

    unidentified_puzzles = [row[0] for row in observer.cursor.fetchall()]

    if unidentified_puzzles:
        print(f"\n{'='*70}")
        print(f"Sample Puzzles with No Detected Patterns ({len(unidentified_puzzles)} shown):")
        print(f"{'='*70}")
        for puzzle_id in unidentified_puzzles[:20]:
            print(f"  - {puzzle_id}")
        print("\nThese puzzles likely require more sophisticated pattern detectors.")

    print(f"\n{'='*70}")
    print("Phase 2: Full Dataset Training Complete!")
    print(f"{'='*70}")

    observer.close()


if __name__ == '__main__':
    main()
