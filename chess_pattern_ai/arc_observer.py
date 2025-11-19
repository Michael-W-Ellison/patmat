#!/usr/bin/env python3
"""
ARC Observer - Learn transformation patterns from ARC puzzles

Extends the universal_game_learner.GameObserver to learn visual transformation
patterns instead of game move patterns.

Philosophy: Learn transformation rules from observation, not hardcoded rules
"""

import sqlite3
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from arc_puzzle import ARCPuzzle, ARCDatasetLoader
from universal_game_learner import GameObserver
from arc_object_detector import ObjectTransformationDetector
from arc_symmetry_detector import SymmetryDetector, RepetitionDetector


class ARCObserver(GameObserver):
    """
    Learns ARC puzzle transformation patterns through observation

    Treats puzzles as games:
    - Starting state = Input grid
    - "Move" = Transformation
    - Winning state = Output grid
    """

    def __init__(self, db_path='arc_learned.db'):
        # Initialize parent class
        super().__init__('arc', db_path)

        # Create ARC-specific tables
        self._init_arc_tables()

        # Initialize detectors
        self.object_detector = ObjectTransformationDetector()
        self.symmetry_detector = SymmetryDetector()
        self.repetition_detector = RepetitionDetector()

    def _init_arc_tables(self):
        """Create tables for ARC-specific pattern learning"""

        # Store observed transformations
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS observed_transformations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT NOT NULL,
                example_idx INTEGER NOT NULL,
                input_grid TEXT NOT NULL,        -- JSON serialized
                output_grid TEXT NOT NULL,       -- JSON serialized
                input_shape TEXT NOT NULL,       -- "(rows, cols)"
                output_shape TEXT NOT NULL,
                transformation_type TEXT,        -- Detected pattern type
                times_observed INTEGER DEFAULT 1,

                UNIQUE(puzzle_id, example_idx)
            )
        ''')

        # Store learned transformation patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transformation_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,      -- 'reflection', 'rotation', 'tiling', etc.
                pattern_name TEXT NOT NULL,
                pattern_description TEXT,
                times_seen INTEGER DEFAULT 0,
                times_successful INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,

                -- Pattern-specific parameters (JSON)
                parameters TEXT,

                UNIQUE(pattern_type, pattern_name)
            )
        ''')

        # Store spatial features for pattern matching
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spatial_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT NOT NULL,
                feature_type TEXT NOT NULL,      -- 'dimension_scale', 'color_distribution', etc.
                feature_value TEXT NOT NULL,     -- JSON serialized
                pattern_type TEXT,               -- Associated pattern

                UNIQUE(puzzle_id, feature_type)
            )
        ''')

        self.conn.commit()

    def observe_puzzle(self, puzzle: ARCPuzzle):
        """
        Watch a complete puzzle and learn from all training examples

        Args:
            puzzle: ARCPuzzle instance with train examples
        """
        print(f"Observing puzzle: {puzzle.puzzle_id} ({len(puzzle.train_examples)} examples)")

        for idx, (input_grid, output_grid) in enumerate(puzzle.get_train_pairs()):
            self._observe_transformation(puzzle.puzzle_id, idx, input_grid, output_grid)

        # After observing all examples, detect overall pattern
        self._detect_puzzle_pattern(puzzle)

    def _observe_transformation(self, puzzle_id: str, example_idx: int,
                               input_grid: List[List[int]],
                               output_grid: List[List[int]]):
        """
        Record a single input→output transformation

        This is analogous to observing a chess move
        """
        input_shape = (len(input_grid), len(input_grid[0]) if input_grid else 0)
        output_shape = (len(output_grid), len(output_grid[0]) if output_grid else 0)

        # Serialize grids to JSON
        input_json = json.dumps(input_grid)
        output_json = json.dumps(output_grid)

        # Store transformation
        self.cursor.execute('''
            INSERT INTO observed_transformations
                (puzzle_id, example_idx, input_grid, output_grid,
                 input_shape, output_shape, times_observed)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(puzzle_id, example_idx)
            DO UPDATE SET times_observed = times_observed + 1
        ''', (
            puzzle_id, example_idx, input_json, output_json,
            str(input_shape), str(output_shape)
        ))

        self.conn.commit()

    def _detect_puzzle_pattern(self, puzzle: ARCPuzzle):
        """
        Analyze all examples in a puzzle to detect the transformation pattern

        This is where the learning happens - discovering what the transformation IS
        """
        train_pairs = puzzle.get_train_pairs()

        if not train_pairs:
            return

        # Run pattern detectors
        patterns_detected = []

        # 1. Check for dimension scaling
        scale_pattern = self._detect_scaling_pattern(train_pairs)
        if scale_pattern:
            patterns_detected.append(scale_pattern)

        # 2. Check for reflection/rotation
        spatial_pattern = self._detect_spatial_transformation(train_pairs)
        if spatial_pattern:
            patterns_detected.append(spatial_pattern)

        # 3. Check for tiling/repetition
        tiling_pattern = self._detect_tiling_pattern(train_pairs)
        if tiling_pattern:
            patterns_detected.append(tiling_pattern)

        # 4. Check for color transformations
        color_pattern = self._detect_color_transformation(train_pairs)
        if color_pattern:
            patterns_detected.append(color_pattern)

        # 5. Check for object-based transformations
        object_pattern = self._detect_object_transformation(train_pairs)
        if object_pattern:
            patterns_detected.append(object_pattern)

        # 6. Check for symmetry operations
        symmetry_pattern = self._detect_symmetry_operations(train_pairs)
        if symmetry_pattern:
            patterns_detected.append(symmetry_pattern)

        # 7. Check for repetition operations
        repetition_pattern = self._detect_repetition_operations(train_pairs)
        if repetition_pattern:
            patterns_detected.append(repetition_pattern)

        # Record discovered patterns
        for pattern in patterns_detected:
            self._record_pattern(puzzle.puzzle_id, pattern)

    def _detect_scaling_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect if output is scaled version of input"""
        scales = []

        for input_grid, output_grid in train_pairs:
            in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

            if in_h > 0 and in_w > 0:
                scale_h = out_h / in_h
                scale_w = out_w / in_w
                scales.append((scale_h, scale_w))

        # Check if all examples have same scale
        if scales and len(set(scales)) == 1:
            scale_h, scale_w = scales[0]

            if scale_h == scale_w and scale_h > 1:
                return {
                    'type': 'scaling',
                    'name': f'uniform_scale_{int(scale_h)}x',
                    'description': f'Output is {int(scale_h)}x scaled version of input',
                    'parameters': {'scale_factor': scale_h}
                }
            elif scale_h != 1 or scale_w != 1:
                return {
                    'type': 'scaling',
                    'name': f'anisotropic_scale_{scale_h}x{scale_w}',
                    'description': f'Output is scaled {scale_h}x{scale_w}',
                    'parameters': {'scale_h': scale_h, 'scale_w': scale_w}
                }

        return None

    def _detect_spatial_transformation(self, train_pairs) -> Optional[Dict]:
        """Detect reflection, rotation, or transposition"""

        # Check all examples for consistent transformation
        transformations = []

        for input_grid, output_grid in train_pairs:
            # Skip if different dimensions
            if (len(input_grid), len(input_grid[0])) != (len(output_grid), len(output_grid[0])):
                continue

            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            # Check for horizontal flip
            if np.array_equal(np.fliplr(input_np), output_np):
                transformations.append('horizontal_flip')

            # Check for vertical flip
            elif np.array_equal(np.flipud(input_np), output_np):
                transformations.append('vertical_flip')

            # Check for 90° rotation
            elif np.array_equal(np.rot90(input_np, 1), output_np):
                transformations.append('rotate_90')

            # Check for 180° rotation
            elif np.array_equal(np.rot90(input_np, 2), output_np):
                transformations.append('rotate_180')

            # Check for 270° rotation
            elif np.array_equal(np.rot90(input_np, 3), output_np):
                transformations.append('rotate_270')

            # Check for transpose
            elif np.array_equal(input_np.T, output_np):
                transformations.append('transpose')

            else:
                transformations.append('unknown')

        # If all examples show same transformation
        if transformations and len(set(transformations)) == 1 and transformations[0] != 'unknown':
            trans_type = transformations[0]
            return {
                'type': 'spatial',
                'name': trans_type,
                'description': f'Apply {trans_type.replace("_", " ")} to input',
                'parameters': {'transformation': trans_type}
            }

        return None

    def _detect_tiling_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect if output is tiled/repeated version of input or sub-pattern"""

        for input_grid, output_grid in train_pairs:
            in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

            # Check if output dimensions are multiples of input
            if in_h > 0 and in_w > 0 and out_h % in_h == 0 and out_w % in_w == 0:
                tile_h = out_h // in_h
                tile_w = out_w // in_w

                # Check if output is actually tiled input
                if self._is_tiled_pattern(input_grid, output_grid, tile_h, tile_w):
                    return {
                        'type': 'tiling',
                        'name': f'tile_{tile_h}x{tile_w}',
                        'description': f'Tile input in {tile_h}x{tile_w} pattern',
                        'parameters': {'tile_h': tile_h, 'tile_w': tile_w}
                    }

        return None

    def _is_tiled_pattern(self, input_grid, output_grid, tile_h, tile_w) -> bool:
        """Check if output is tiled version of input"""
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)

        in_h, in_w = input_np.shape
        out_h, out_w = output_np.shape

        # Check each tile position
        for i in range(tile_h):
            for j in range(tile_w):
                tile = output_np[i*in_h:(i+1)*in_h, j*in_w:(j+1)*in_w]

                # Check if tile matches input (allowing for partial matches near edges)
                if tile.shape == input_np.shape:
                    if not np.array_equal(tile, input_np):
                        # Not a simple tiling pattern
                        return False

        return True

    def _detect_color_transformation(self, train_pairs) -> Optional[Dict]:
        """Detect color mapping or recoloring patterns"""

        # Check if there's a consistent color mapping
        color_maps = []

        for input_grid, output_grid in train_pairs:
            if (len(input_grid), len(input_grid[0])) != (len(output_grid), len(output_grid[0])):
                continue

            # Build color mapping
            color_map = {}
            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            for (i, j), in_color in np.ndenumerate(input_np):
                out_color = output_np[i, j]

                # Convert numpy types to Python native types for JSON serialization
                in_color = int(in_color)
                out_color = int(out_color)

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        # Inconsistent mapping
                        color_map = None
                        break
                else:
                    color_map[in_color] = out_color

            if color_map:
                color_maps.append(frozenset(color_map.items()))

        # If all examples have same color mapping
        if color_maps and len(set(color_maps)) == 1:
            color_map = dict(color_maps[0])

            # Check if it's not identity mapping
            if not all(k == v for k, v in color_map.items()):
                return {
                    'type': 'color_mapping',
                    'name': 'color_remap',
                    'description': f'Remap colors: {color_map}',
                    'parameters': {'mapping': color_map}
                }

        return None

    def _detect_object_transformation(self, train_pairs) -> Optional[Dict]:
        """Detect object-based transformations (movement, copying, recoloring)"""

        # Analyze transformations for each example
        transformations_by_type = {}

        for input_grid, output_grid in train_pairs:
            result = self.object_detector.detect_transformation(input_grid, output_grid)

            if result['transformations']:
                for trans in result['transformations']:
                    trans_type = trans['type']
                    if trans_type not in transformations_by_type:
                        transformations_by_type[trans_type] = []
                    transformations_by_type[trans_type].append(trans)

        # Check if consistent transformation across all examples
        for trans_type, trans_list in transformations_by_type.items():
            if len(trans_list) == len(train_pairs):
                # This transformation appears in all examples

                if trans_type == 'uniform_movement':
                    # Check if all movements have same delta
                    deltas = [t['delta'] for t in trans_list]
                    if len(set(deltas)) == 1:
                        delta = deltas[0]
                        return {
                            'type': 'object',
                            'name': f'object_move_{delta[0]:.0f}_{delta[1]:.0f}',
                            'description': f'Move all objects by ({delta[0]:.0f}, {delta[1]:.0f})',
                            'parameters': {'delta': delta, 'operation': 'movement'}
                        }

                elif trans_type == 'object_copying':
                    # Objects were copied
                    copies = trans_list[0]['copies']  # Assume consistent
                    return {
                        'type': 'object',
                        'name': 'object_copying',
                        'description': f'Copy objects: {copies}',
                        'parameters': {'copies': copies, 'operation': 'copying'}
                    }

                elif trans_type == 'object_recoloring':
                    # Check if color mapping is consistent
                    color_maps = [t['color_map'] for t in trans_list]
                    # Convert to comparable format
                    color_maps_str = [str(sorted(cm.items())) for cm in color_maps]
                    if len(set(color_maps_str)) == 1:
                        color_map = color_maps[0]
                        return {
                            'type': 'object',
                            'name': 'object_recoloring',
                            'description': f'Recolor objects: {color_map}',
                            'parameters': {'color_map': color_map, 'operation': 'recoloring'}
                        }

                elif trans_type == 'object_scaling':
                    # Objects were scaled
                    scales = [(t['scale_h'], t['scale_w']) for t in trans_list]
                    if len(set(scales)) == 1:
                        scale_h, scale_w = scales[0]
                        return {
                            'type': 'object',
                            'name': f'object_scale_{scale_h}x{scale_w}',
                            'description': f'Scale objects by {scale_h}x{scale_w}',
                            'parameters': {'scale_h': scale_h, 'scale_w': scale_w, 'operation': 'scaling'}
                        }

        return None

    def _detect_symmetry_operations(self, train_pairs) -> Optional[Dict]:
        """Detect symmetry-based transformations (reflection, pattern completion)"""

        # Check all examples for consistent symmetry operation
        symmetries = []

        for input_grid, output_grid in train_pairs:
            result = self.symmetry_detector.detect_symmetry_transformation(input_grid, output_grid)
            if result:
                symmetries.append(result)

        # If same symmetry operation appears in all examples
        if symmetries and len(symmetries) == len(train_pairs):
            # Check if all examples show same operation
            operations = [s['name'] for s in symmetries]
            if len(set(operations)) == 1:
                # All examples have same symmetry operation
                sym = symmetries[0]
                return {
                    'type': sym['type'],
                    'name': sym['name'],
                    'description': sym['description'],
                    'parameters': sym.get('parameters', {})
                }

        return None

    def _detect_repetition_operations(self, train_pairs) -> Optional[Dict]:
        """Detect repetition-based transformations (tiling, duplication)"""

        # Check all examples for consistent repetition operation
        repetitions = []

        for input_grid, output_grid in train_pairs:
            result = self.repetition_detector.detect_repetition(input_grid, output_grid)
            if result:
                repetitions.append(result)

        # If same repetition operation appears in all examples
        if repetitions and len(repetitions) == len(train_pairs):
            # Check if all examples show same operation
            operations = [r['name'] for r in repetitions]
            if len(set(operations)) == 1:
                # All examples have same repetition operation
                rep = repetitions[0]
                return {
                    'type': rep['type'],
                    'name': rep['name'],
                    'description': rep['description'],
                    'parameters': rep.get('parameters', {})
                }

        return None

    def _record_pattern(self, puzzle_id: str, pattern: Dict):
        """Record a discovered pattern in the database"""

        self.cursor.execute('''
            INSERT INTO transformation_patterns
                (pattern_type, pattern_name, pattern_description, parameters, times_seen)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(pattern_type, pattern_name)
            DO UPDATE SET times_seen = times_seen + 1
        ''', (
            pattern['type'],
            pattern['name'],
            pattern['description'],
            json.dumps(pattern.get('parameters', {}))
        ))

        # Store spatial features
        self.cursor.execute('''
            INSERT INTO spatial_features
                (puzzle_id, feature_type, feature_value, pattern_type)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(puzzle_id, feature_type)
            DO NOTHING
        ''', (
            puzzle_id,
            pattern['type'],
            json.dumps(pattern.get('parameters', {})),
            pattern['name']
        ))

        self.conn.commit()

        print(f"  → Detected: {pattern['description']}")

    def get_learned_patterns(self):
        """Display all learned transformation patterns"""

        print(f"\n{'='*70}")
        print(f"LEARNED PATTERNS: ARC TRANSFORMATIONS")
        print(f"{'='*70}")

        self.cursor.execute('''
            SELECT pattern_type, pattern_name, pattern_description, times_seen
            FROM transformation_patterns
            ORDER BY times_seen DESC
        ''')

        patterns_by_type = {}
        for pattern_type, name, desc, count in self.cursor.fetchall():
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append((name, desc, count))

        for pattern_type, patterns in patterns_by_type.items():
            print(f"\n{pattern_type.upper()} PATTERNS:")
            print(f"{'Pattern':<30} {'Count':<10} {'Description':<40}")
            print("-" * 70)

            for name, desc, count in patterns:
                print(f"{name:<30} {count:<10} {desc:<40}")

        # Total observations
        self.cursor.execute('SELECT COUNT(*) FROM observed_transformations')
        total_obs = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT COUNT(*) FROM transformation_patterns')
        total_patterns = self.cursor.fetchone()[0]

        print(f"\n{'='*70}")
        print(f"Total: {total_patterns} unique patterns, {total_obs} observations")
        print(f"{'='*70}\n")

    def parse_move(self, move_notation):
        """Not used for ARC - transformations are handled differently"""
        pass

    def extract_board_features(self, position):
        """Not used for ARC - we extract grid features instead"""
        pass


def main():
    """Demo: Learn patterns from ARC training set"""
    print("="*70)
    print("ARC OBSERVER - Pattern Learning Demo")
    print("="*70)

    # Initialize observer
    observer = ARCObserver('arc_learned.db')

    # Load dataset
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Train on first 10 puzzles
    print("\nLoading first 10 training puzzles...")
    puzzles = loader.load_training_set()[:10]

    print(f"\nObserving {len(puzzles)} puzzles to learn patterns...\n")

    for puzzle in puzzles:
        observer.observe_puzzle(puzzle)

    # Display learned patterns
    observer.get_learned_patterns()

    print("\n" + "="*70)
    print("Phase 1 Complete: ARCObserver successfully learning patterns!")
    print("="*70)

    observer.close()


if __name__ == '__main__':
    main()
