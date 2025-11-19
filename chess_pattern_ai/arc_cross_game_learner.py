#!/usr/bin/env python3
"""
ARC Cross-Game Pattern Learner - Integrates universal patterns from all games

User's insight: "The AI should use patterns from all games when it encounters a new game."

This extends ARCMetaPatternLearner to:
1. Query universal patterns from chess/checkers/Dots and Boxes
2. Boost confidence for patterns seen across multiple games
3. Prioritize cross-game patterns over ARC-only patterns
"""

import numpy as np
import sqlite3
import json
from typing import List, Tuple, Dict, Optional
from arc_meta_pattern_learner import ARCMetaPatternLearner


class ARCCrossGameLearner(ARCMetaPatternLearner):
    """
    ARC learner that leverages universal patterns from all games

    Like user described:
    - Use reflection patterns learned from chess (100% confidence)
    - Use boundary patterns from Dots and Boxes
    - Boost confidence for cross-game pattern matches
    """

    def __init__(self, arc_db_path='arc_meta_patterns.db',
                 universal_db_path='universal_patterns.db'):
        # Initialize base ARC meta-pattern learner
        super().__init__(arc_db_path)

        # Connect to universal patterns database
        self.universal_conn = sqlite3.connect(universal_db_path)
        self.universal_cursor = self.universal_conn.cursor()

    def get_universal_patterns(self) -> List[Dict]:
        """Get all universal patterns from cross-game database"""

        self.universal_cursor.execute('''
            SELECT
                p.pattern_id,
                p.pattern_name,
                p.pattern_category,
                p.confidence,
                p.observation_count,
                p.description
            FROM universal_patterns p
            ORDER BY p.confidence DESC, p.observation_count DESC
        ''')

        patterns = []
        for row in self.universal_cursor.fetchall():
            pattern_id, name, category, confidence, obs_count, description = row

            # Get games showing this pattern
            self.universal_cursor.execute('''
                SELECT game_type, frequency, observation_count
                FROM game_to_pattern
                WHERE pattern_id = ?
            ''', (pattern_id,))

            games = []
            for game_row in self.universal_cursor.fetchall():
                games.append({
                    'game': game_row[0],
                    'frequency': game_row[1],
                    'observations': game_row[2]
                })

            # Get features
            self.universal_cursor.execute('''
                SELECT feature_name, feature_value
                FROM universal_pattern_features
                WHERE pattern_id = ?
            ''', (pattern_id,))

            features = {}
            for fname, fvalue in self.universal_cursor.fetchall():
                features[fname] = json.loads(fvalue)

            patterns.append({
                'pattern_id': pattern_id,
                'name': name,
                'category': category,
                'confidence': confidence,
                'observations': obs_count,
                'description': description,
                'games': games,
                'features': features,
                'cross_game': len(games) > 1,
                'source': 'universal'
            })

        return patterns

    def match_to_universal_patterns(self, features: Dict) -> List[Dict]:
        """
        Match ARC puzzle features to universal patterns

        Like user described:
        - Chess reflection pattern → Match to ARC reflection puzzle
        - Boost confidence from cross-game observation
        """
        matches = []

        pattern_type = self.classify_transformation(features)

        # Map ARC pattern types to universal pattern names
        type_mapping = {
            'horizontal_reflection': 'horizontal_reflection',
            'vertical_reflection': 'vertical_reflection',
            'rotation_90': 'rotation',
            'rotation_180': 'rotation',
            'rotation_270': 'rotation',
            'fill_enclosed_region': 'boundary_completion'
        }

        universal_pattern_name = type_mapping.get(pattern_type)

        if universal_pattern_name:
            # Query universal patterns
            self.universal_cursor.execute('''
                SELECT
                    p.pattern_id,
                    p.pattern_name,
                    p.confidence,
                    p.observation_count,
                    p.description
                FROM universal_patterns p
                WHERE p.pattern_name = ?
            ''', (universal_pattern_name,))

            result = self.universal_cursor.fetchone()

            if result:
                pattern_id, name, confidence, obs_count, description = result

                # Get games showing this pattern
                self.universal_cursor.execute('''
                    SELECT game_type, frequency
                    FROM game_to_pattern
                    WHERE pattern_id = ?
                ''', (pattern_id,))

                games = [row[0] for row in self.universal_cursor.fetchall()]

                # Cross-game confidence boost
                cross_game_boost = len(games) * 0.1  # +0.1 per game

                matches.append({
                    'pattern_name': name,
                    'type': pattern_type,
                    'confidence': confidence,
                    'base_confidence': confidence,
                    'cross_game_boost': cross_game_boost,
                    'boosted_confidence': min(1.0, confidence + cross_game_boost),
                    'observations': obs_count,
                    'games': games,
                    'description': description,
                    'source': 'universal'
                })

        return matches

    def match_puzzle_to_patterns(self, training_pairs: List[Tuple],
                                  use_universal=True) -> List[Dict]:
        """
        Match puzzle to both ARC patterns and universal patterns

        Priority:
        1. Universal cross-game patterns (highest confidence)
        2. ARC-specific patterns with multiple observations
        3. Low-confidence ARC patterns
        """

        if not training_pairs:
            return []

        # Extract features
        features = self.extract_features(training_pairs[0][0], training_pairs[0][1])

        all_matches = []

        # 1. Check universal patterns (from chess/checkers/etc)
        if use_universal:
            universal_matches = self.match_to_universal_patterns(features)
            all_matches.extend(universal_matches)

        # 2. Check ARC-specific patterns
        arc_matches = super().match_puzzle_to_patterns(training_pairs)

        # Convert to common format
        for match in arc_matches:
            match['source'] = 'arc'
            match['boosted_confidence'] = match.get('confidence', 0.5)
            all_matches.append(match)

        # Sort by boosted confidence (universal patterns get priority)
        all_matches.sort(key=lambda m: (
            m.get('boosted_confidence', 0.5),
            m.get('observations', 0)
        ), reverse=True)

        return all_matches

    def solve_with_universal_patterns(self, puzzle) -> Optional[List[List[int]]]:
        """
        Solve ARC puzzle using universal + ARC patterns

        Returns solution or None
        """

        training_pairs = puzzle.get_train_pairs()
        test_input = puzzle.get_test_inputs()[0]

        # Match to patterns (including universal)
        matches = self.match_puzzle_to_patterns(training_pairs, use_universal=True)

        if not matches:
            return None

        # Try highest-confidence pattern
        best_match = matches[0]

        print(f"Best match: {best_match.get('pattern_name', best_match.get('type'))}")
        print(f"  Source: {best_match['source']}")
        print(f"  Confidence: {best_match.get('boosted_confidence', 0.5):.2f}")

        if best_match['source'] == 'universal':
            print(f"  Games: {', '.join(best_match['games'])}")
            print(f"  ✓ Cross-game pattern - HIGH CONFIDENCE!")

        # Apply transformation based on pattern type
        pattern_type = best_match.get('pattern_name', best_match.get('type'))

        return self._apply_pattern_transformation(test_input, training_pairs, pattern_type)

    def _apply_pattern_transformation(self, test_input, training_pairs,
                                     pattern_type: str) -> Optional[List[List[int]]]:
        """
        Apply transformation based on pattern type

        Implements transformations for universal patterns
        """
        inp = np.array(test_input)

        # Reflection patterns (from chess/checkers)
        if pattern_type == 'horizontal_reflection':
            result = np.fliplr(inp)
            return result.tolist()

        elif pattern_type == 'vertical_reflection':
            result = np.flipud(inp)
            return result.tolist()

        # Rotation patterns
        elif 'rotation_90' in pattern_type:
            result = np.rot90(inp, 1)
            return result.tolist()

        elif 'rotation_180' in pattern_type:
            result = np.rot90(inp, 2)
            return result.tolist()

        elif 'rotation_270' in pattern_type:
            result = np.rot90(inp, 3)
            return result.tolist()

        # Boundary completion (from Dots and Boxes)
        elif pattern_type == 'boundary_completion' or 'fill_enclosed' in pattern_type:
            return self._apply_fill_enclosed(test_input, training_pairs)

        return None

    def _apply_fill_enclosed(self, test_input, training_pairs) -> Optional[List[List[int]]]:
        """
        Apply fill enclosed pattern (learned from Dots and Boxes)

        Like Dots and Boxes: Complete boundary → Fill interior
        """
        # Learn fill color from training examples
        train_input, train_output = training_pairs[0]
        inp = np.array(train_input)
        out = np.array(train_output)

        changed = (inp != out)
        if np.sum(changed) == 0:
            return None

        # Find fill color
        fill_colors = set(out[changed].flatten()) - set(inp[changed].flatten())
        if not fill_colors:
            return None

        fill_color = list(fill_colors)[0]

        # Apply to test: Flood fill from edges to find exterior
        test_arr = np.array(test_input)
        h, w = test_arr.shape

        # Mark all background cells reachable from edges
        exterior = np.zeros_like(test_arr, dtype=bool)
        queue = []

        # Start from all edge background cells
        for i in range(h):
            if test_arr[i, 0] == 0:
                queue.append((i, 0))
            if test_arr[i, w-1] == 0:
                queue.append((i, w-1))
        for j in range(w):
            if test_arr[0, j] == 0:
                queue.append((0, j))
            if test_arr[h-1, j] == 0:
                queue.append((h-1, j))

        # Flood fill
        while queue:
            r, c = queue.pop(0)
            if exterior[r, c] or test_arr[r, c] != 0:
                continue

            exterior[r, c] = True

            # Add neighbors
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not exterior[nr, nc]:
                    queue.append((nr, nc))

        # Interior = background NOT reachable from edges
        interior = (test_arr == 0) & ~exterior

        # Fill interior
        result = test_arr.copy()
        result[interior] = fill_color

        return result.tolist()

    def close(self):
        """Close both databases"""
        super().close()
        self.universal_conn.close()


if __name__ == '__main__':
    from arc_puzzle import ARCDatasetLoader

    print("ARC Cross-Game Pattern Learning")
    print("="*70)
    print("\nUser's insight: \"The AI should use patterns from all games\"")
    print("Testing with universal patterns from chess/checkers/Dots and Boxes")
    print()

    learner = ARCCrossGameLearner()
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Test on puzzles that should match universal patterns
    test_puzzles = [
        ('00d62c1b', 'fill_enclosed', 'Should match Dots and Boxes boundary completion'),
        ('0962bcdd', 'reflection', 'Should match chess/checkers reflection'),
        ('6150a2bd', 'reflection', 'Should match chess/checkers reflection'),
    ]

    print("Testing cross-game pattern matching:")
    print("="*70)

    for puzzle_id, expected_type, reason in test_puzzles:
        try:
            puzzle = loader.load_puzzle(puzzle_id, 'training')

            print(f"\nPuzzle {puzzle_id}:")
            print(f"  Expected: {expected_type}")
            print(f"  Reason: {reason}")
            print()

            # Match to patterns
            matches = learner.match_puzzle_to_patterns(
                puzzle.get_train_pairs(),
                use_universal=True
            )

            if matches:
                print(f"  Top 3 matches:")
                for i, match in enumerate(matches[:3], 1):
                    source_marker = "🌟" if match['source'] == 'universal' else "📊"
                    pattern_name = match.get('pattern_name', match.get('type', 'unknown'))

                    print(f"    {i}. {source_marker} {pattern_name}")
                    print(f"       Source: {match['source']}")
                    print(f"       Confidence: {match.get('boosted_confidence', 0.5):.2f}")

                    if match['source'] == 'universal':
                        print(f"       Games: {', '.join(match.get('games', []))}")

                # Try to solve
                print()
                print(f"  Attempting solution with highest-confidence pattern...")
                solution = learner.solve_with_universal_patterns(puzzle)

                if solution:
                    print(f"  ✓ Solution generated!")
                else:
                    print(f"  ✗ Could not generate solution")

            else:
                print(f"  ✗ No pattern matches found")

            print("-"*70)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print("-"*70)

    # Show statistics
    print("\n" + "="*70)
    print("Universal Pattern Statistics:")
    print("="*70)

    universal_patterns = learner.get_universal_patterns()
    print(f"Total universal patterns: {len(universal_patterns)}")
    print(f"Cross-game patterns: {sum(1 for p in universal_patterns if p['cross_game'])}")
    print(f"Average confidence: {np.mean([p['confidence'] for p in universal_patterns]):.2f}")

    print("\nCross-game patterns available for ARC:")
    for pattern in universal_patterns:
        if pattern['cross_game']:
            print(f"  - {pattern['name']}: {pattern['confidence']:.2f} confidence")
            print(f"    Games: {', '.join([g['game'] for g in pattern['games']])}")

    learner.close()
