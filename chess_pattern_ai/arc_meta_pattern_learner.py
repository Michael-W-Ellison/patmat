#!/usr/bin/env python3
"""
Meta-Pattern Learner for ARC - GameObserver Approach

User's insight: "The AI should look for meta-patterns like it does during chess games"

This implements the SAME approach as GameObserver:
1. Observe transformations across many puzzles
2. Detect recurring meta-patterns (like chess openings)
3. Build confidence scores from observations
4. Match new puzzles to learned meta-patterns
5. Apply highest-confidence transformation
"""

import numpy as np
import sqlite3
import json
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class ARCMetaPatternLearner:
    """
    Learn meta-patterns across ARC puzzles

    Like GameObserver for chess:
    - Observe many games/puzzles
    - Extract recurring patterns
    - Build pattern database with confidence scores
    - Match new positions/puzzles to learned patterns
    """

    def __init__(self, db_path='arc_meta_patterns.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_database()

    def _init_database(self):
        """Initialize meta-pattern database (like chess position DB)"""

        # Meta-patterns table - stores learned patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                signature TEXT,
                confidence REAL DEFAULT 0.5,
                observation_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Pattern features - specific features of each meta-pattern
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_features (
                pattern_id INTEGER,
                feature_name TEXT,
                feature_value TEXT,
                FOREIGN KEY (pattern_id) REFERENCES meta_patterns(pattern_id)
            )
        ''')

        # Puzzle-pattern mapping - which puzzles match which patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzle_patterns (
                puzzle_id TEXT,
                pattern_id INTEGER,
                match_confidence REAL,
                FOREIGN KEY (pattern_id) REFERENCES meta_patterns(pattern_id)
            )
        ''')

        self.conn.commit()

    def extract_features(self, input_grid, output_grid) -> Dict:
        """
        Extract observable features from transformation

        Like extracting features from a chess position:
        - Not semantic ("what does this mean?")
        - Observable ("what do I see?")
        - Comparable ("can I match this to previous observations?")
        """
        inp = np.array(input_grid)
        out = np.array(output_grid)

        features = {}

        # Size features
        features['input_shape'] = inp.shape
        features['output_shape'] = out.shape
        features['size_changed'] = (inp.shape != out.shape)

        if inp.shape == out.shape:
            # Same-size transformation features
            changed = (inp != out)
            features['cells_changed'] = int(np.sum(changed))
            features['change_percentage'] = float(np.sum(changed) / inp.size * 100)

            # Color features
            inp_colors = set(inp.flatten()) - {0}
            out_colors = set(out.flatten()) - {0}

            features['input_colors'] = sorted(list(inp_colors))
            features['output_colors'] = sorted(list(out_colors))
            features['new_colors'] = sorted(list(out_colors - inp_colors))
            features['removed_colors'] = sorted(list(inp_colors - out_colors))

            # Spatial features - where do changes occur?
            if np.sum(changed) > 0:
                changed_coords = np.argwhere(changed)
                features['change_region_size'] = len(changed_coords)

                # Are changes localized or distributed?
                if len(changed_coords) > 0:
                    y_coords = changed_coords[:, 0]
                    x_coords = changed_coords[:, 1]

                    y_span = y_coords.max() - y_coords.min() + 1
                    x_span = x_coords.max() - x_coords.min() + 1

                    features['change_span'] = (int(y_span), int(x_span))

                    # Localized if changes fit in <1/4 of grid
                    is_localized = (y_span * x_span) < (inp.shape[0] * inp.shape[1] / 4)
                    features['localized'] = bool(is_localized)

            # Check for reflection symmetry
            horizontal_flip = np.array_equal(out, np.fliplr(inp))
            vertical_flip = np.array_equal(out, np.flipud(inp))

            features['horizontal_reflection'] = horizontal_flip
            features['vertical_reflection'] = vertical_flip

            # Check for rotation
            rotate_90 = np.array_equal(out, np.rot90(inp, 1))
            rotate_180 = np.array_equal(out, np.rot90(inp, 2))
            rotate_270 = np.array_equal(out, np.rot90(inp, 3))

            features['rotation_90'] = rotate_90
            features['rotation_180'] = rotate_180
            features['rotation_270'] = rotate_270

        else:
            # Size-change transformation
            features['scale_factor'] = (
                out.shape[0] / inp.shape[0],
                out.shape[1] / inp.shape[1]
            )

        return features

    def classify_transformation(self, features: Dict) -> str:
        """
        Classify transformation based on features

        Like classifying a chess position as "opening" vs "middlegame" vs "endgame"
        based on observable features (piece count, pawn structure, etc.)
        """

        # Simple geometric transformations (easy to classify)
        if features.get('horizontal_reflection'):
            return 'horizontal_reflection'
        if features.get('vertical_reflection'):
            return 'vertical_reflection'
        if features.get('rotation_90'):
            return 'rotation_90'
        if features.get('rotation_180'):
            return 'rotation_180'
        if features.get('rotation_270'):
            return 'rotation_270'

        # Size-change transformations
        if features.get('size_changed'):
            scale = features.get('scale_factor', (1, 1))
            if scale[0] > 1 or scale[1] > 1:
                return 'scaling_or_tiling'
            else:
                return 'cropping_or_extraction'

        # Same-size transformations
        change_pct = features.get('change_percentage', 0)

        if change_pct < 10:
            # Small changes - likely filling or detail adding
            if features.get('localized', False):
                return 'fill_enclosed_region'
            else:
                return 'sparse_modification'

        elif change_pct > 60:
            # Large changes - likely major transformation
            return 'major_transformation'

        else:
            # Moderate changes
            return 'moderate_transformation'

    def observe_puzzle(self, puzzle_id: str, training_pairs: List[Tuple]):
        """
        Observe a puzzle and update meta-pattern database

        Like GameObserver.observe_game() for chess:
        - Extract features from observations
        - Classify transformation type
        - Update pattern database
        - Increase confidence for recurring patterns
        """

        for idx, (input_grid, output_grid) in enumerate(training_pairs):
            # Extract features
            features = self.extract_features(input_grid, output_grid)

            # Classify transformation
            pattern_type = self.classify_transformation(features)

            # Create signature (like chess position hash)
            signature_parts = [
                pattern_type,
                str(features.get('input_shape')),
                str(features.get('output_shape')),
                f"change_{features.get('change_percentage', 0):.1f}pct"
            ]
            signature = "_".join(signature_parts)

            # Find or create meta-pattern
            self.cursor.execute('''
                SELECT pattern_id, observation_count FROM meta_patterns
                WHERE signature = ?
            ''', (signature,))

            result = self.cursor.fetchone()

            if result:
                # Existing pattern - increase observation count
                pattern_id, obs_count = result

                self.cursor.execute('''
                    UPDATE meta_patterns
                    SET observation_count = observation_count + 1,
                        confidence = MIN(1.0, confidence + 0.05)
                    WHERE pattern_id = ?
                ''', (pattern_id,))

            else:
                # New pattern - create it
                self.cursor.execute('''
                    INSERT INTO meta_patterns (pattern_type, signature, confidence, observation_count)
                    VALUES (?, ?, 0.5, 1)
                ''', (pattern_type, signature))

                pattern_id = self.cursor.lastrowid

                # Store features
                for feature_name, feature_value in features.items():
                    # Convert numpy types to Python types for JSON
                    if isinstance(feature_value, np.integer):
                        feature_value = int(feature_value)
                    elif isinstance(feature_value, np.floating):
                        feature_value = float(feature_value)
                    elif isinstance(feature_value, np.ndarray):
                        feature_value = feature_value.tolist()
                    elif isinstance(feature_value, (list, tuple)):
                        feature_value = [int(x) if isinstance(x, np.integer) else
                                       float(x) if isinstance(x, np.floating) else x
                                       for x in feature_value]

                    self.cursor.execute('''
                        INSERT INTO pattern_features (pattern_id, feature_name, feature_value)
                        VALUES (?, ?, ?)
                    ''', (pattern_id, feature_name, json.dumps(feature_value)))

            # Link puzzle to pattern
            self.cursor.execute('''
                INSERT OR REPLACE INTO puzzle_patterns (puzzle_id, pattern_id, match_confidence)
                VALUES (?, ?, 1.0)
            ''', (f"{puzzle_id}_ex{idx}", pattern_id))

        self.conn.commit()

    def get_meta_patterns(self) -> List[Dict]:
        """Get all learned meta-patterns sorted by confidence"""

        self.cursor.execute('''
            SELECT pattern_id, pattern_type, signature, confidence, observation_count
            FROM meta_patterns
            ORDER BY confidence DESC, observation_count DESC
        ''')

        patterns = []
        for row in self.cursor.fetchall():
            pattern_id, ptype, signature, confidence, obs_count = row

            # Get features
            self.cursor.execute('''
                SELECT feature_name, feature_value
                FROM pattern_features
                WHERE pattern_id = ?
            ''', (pattern_id,))

            features = {name: json.loads(value) for name, value in self.cursor.fetchall()}

            patterns.append({
                'pattern_id': pattern_id,
                'type': ptype,
                'signature': signature,
                'confidence': confidence,
                'observations': obs_count,
                'features': features
            })

        return patterns

    def match_puzzle_to_patterns(self, training_pairs: List[Tuple]) -> List[Dict]:
        """
        Match a new puzzle to learned meta-patterns

        Like matching a chess position to known openings in database
        """

        # Extract features from first training example
        if not training_pairs:
            return []

        features = self.extract_features(training_pairs[0][0], training_pairs[0][1])
        pattern_type = self.classify_transformation(features)

        # Find matching patterns
        self.cursor.execute('''
            SELECT pattern_id, pattern_type, confidence, observation_count
            FROM meta_patterns
            WHERE pattern_type = ?
            ORDER BY confidence DESC, observation_count DESC
            LIMIT 5
        ''', (pattern_type,))

        matches = []
        for row in self.cursor.fetchall():
            matches.append({
                'pattern_id': row[0],
                'type': row[1],
                'confidence': row[2],
                'observations': row[3]
            })

        return matches

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == '__main__':
    from arc_puzzle import ARCDatasetLoader

    print("Meta-Pattern Learning - GameObserver Approach")
    print("="*70)

    learner = ARCMetaPatternLearner()
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Observe first 50 puzzles (like observing chess games)
    print("\nObserving 50 training puzzles...")

    puzzles = loader.load_training_set()[:50]

    for i, puzzle in enumerate(puzzles):
        learner.observe_puzzle(puzzle.puzzle_id, puzzle.get_train_pairs())

        if (i + 1) % 10 == 0:
            print(f"  Observed {i+1}/50 puzzles...")

    # Show learned meta-patterns
    print("\n" + "="*70)
    print("Learned Meta-Patterns (like chess opening database):")
    print("="*70)

    patterns = learner.get_meta_patterns()

    for pattern in patterns[:15]:  # Show top 15
        print(f"\n{pattern['type']:30s} | Confidence: {pattern['confidence']:.2f} | Observations: {pattern['observations']}")
        print(f"  Signature: {pattern['signature']}")

    print(f"\n{'='*70}")
    print(f"Total meta-patterns learned: {len(patterns)}")

    # Test matching on a new puzzle
    print("\n" + "="*70)
    print("Testing pattern matching on new puzzle:")
    print("="*70)

    test_puzzle = loader.load_puzzle('00d62c1b', 'training')
    matches = learner.match_puzzle_to_patterns(test_puzzle.get_train_pairs())

    print(f"\nPuzzle 00d62c1b matches:")
    for match in matches:
        print(f"  {match['type']:30s} | Confidence: {match['confidence']:.2f} | Observations: {match['observations']}")

    learner.close()
