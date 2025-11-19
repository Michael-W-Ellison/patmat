#!/usr/bin/env python3
"""
ARC Solver - Apply learned patterns to solve puzzles

Uses the patterns learned by ARCObserver to actually solve ARC puzzles.

Architecture:
1. Pattern Matcher: Find which learned pattern(s) match this puzzle
2. Transformation Applier: Apply the pattern to test input
3. Solution Generator: Create output grid
4. Validator: Check solution correctness
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from arc_puzzle import ARCPuzzle, ARCDatasetLoader
from arc_observer import ARCObserver
from arc_object_detector import ARCObjectDetector
from arc_symmetry_detector import SymmetryDetector, RepetitionDetector


class ARCSolver:
    """
    Solves ARC puzzles using learned transformation patterns

    Workflow:
    1. Analyze train examples to find matching patterns
    2. Select best matching pattern from database
    3. Apply pattern to test input
    4. Generate output grid
    """

    def __init__(self, db_path: str = 'arc_learned_full.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Initialize detectors (for applying transformations)
        self.observer = ARCObserver(db_path)

    def solve(self, puzzle: ARCPuzzle) -> Optional[List[List[int]]]:
        """
        Solve an ARC puzzle

        Args:
            puzzle: ARCPuzzle with train examples and test input

        Returns:
            Predicted output grid, or None if no solution found
        """
        # Step 1: Detect pattern in train examples
        train_pairs = puzzle.get_train_pairs()
        detected_pattern = self._detect_pattern_from_examples(train_pairs)

        if not detected_pattern:
            return None

        # Step 2: Get test input
        test_inputs = puzzle.get_test_inputs()
        if not test_inputs:
            return None

        test_input = test_inputs[0]  # Use first test case

        # Step 3: Apply pattern to test input
        output = self._apply_pattern(test_input, detected_pattern)

        return output

    def _detect_pattern_from_examples(self, train_pairs: List[Tuple]) -> Optional[Dict]:
        """
        Detect which pattern this puzzle follows based on train examples

        Returns:
            Pattern dictionary with type, name, and parameters
        """
        # Try each detector type
        detectors = [
            ('scaling', self._detect_scaling_pattern),
            ('spatial', self._detect_spatial_pattern),
            ('tiling', self._detect_tiling_pattern),
            ('color', self._detect_color_pattern),
            ('object', self._detect_object_pattern),
            ('symmetry', self._detect_symmetry_pattern),
            ('repetition', self._detect_repetition_pattern),
        ]

        for detector_type, detector_func in detectors:
            pattern = detector_func(train_pairs)
            if pattern:
                return pattern

        return None

    def _detect_scaling_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect scaling pattern"""
        scales = []
        for input_grid, output_grid in train_pairs:
            in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

            if in_h > 0 and in_w > 0:
                scale_h = out_h / in_h
                scale_w = out_w / in_w
                scales.append((scale_h, scale_w))

        if scales and len(set(scales)) == 1:
            scale_h, scale_w = scales[0]
            if scale_h != 1 or scale_w != 1:
                return {
                    'type': 'scaling',
                    'scale_h': scale_h,
                    'scale_w': scale_w
                }
        return None

    def _detect_spatial_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect spatial transformation"""
        transformations = []

        for input_grid, output_grid in train_pairs:
            if (len(input_grid), len(input_grid[0])) != (len(output_grid), len(output_grid[0])):
                continue

            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            if np.array_equal(np.fliplr(input_np), output_np):
                transformations.append('horizontal_flip')
            elif np.array_equal(np.flipud(input_np), output_np):
                transformations.append('vertical_flip')
            elif np.array_equal(np.rot90(input_np, 1), output_np):
                transformations.append('rotate_90')
            elif np.array_equal(np.rot90(input_np, 2), output_np):
                transformations.append('rotate_180')
            elif np.array_equal(np.rot90(input_np, 3), output_np):
                transformations.append('rotate_270')
            elif np.array_equal(input_np.T, output_np):
                transformations.append('transpose')

        if transformations and len(set(transformations)) == 1:
            return {
                'type': 'spatial',
                'operation': transformations[0]
            }
        return None

    def _detect_tiling_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect tiling pattern"""
        for input_grid, output_grid in train_pairs:
            in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

            if in_h > 0 and in_w > 0 and out_h % in_h == 0 and out_w % in_w == 0:
                tile_h = out_h // in_h
                tile_w = out_w // in_w

                # Verify it's actually tiled
                input_np = np.array(input_grid)
                output_np = np.array(output_grid)

                is_tiled = True
                for i in range(tile_h):
                    for j in range(tile_w):
                        tile = output_np[i*in_h:(i+1)*in_h, j*in_w:(j+1)*in_w]
                        if tile.shape == input_np.shape and not np.array_equal(tile, input_np):
                            is_tiled = False
                            break
                    if not is_tiled:
                        break

                if is_tiled:
                    return {
                        'type': 'tiling',
                        'tile_h': tile_h,
                        'tile_w': tile_w
                    }
        return None

    def _detect_color_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect color mapping"""
        color_maps = []

        for input_grid, output_grid in train_pairs:
            if (len(input_grid), len(input_grid[0])) != (len(output_grid), len(output_grid[0])):
                continue

            color_map = {}
            input_np = np.array(input_grid)
            output_np = np.array(output_grid)

            for (i, j), in_color in np.ndenumerate(input_np):
                out_color = output_np[i, j]
                in_color = int(in_color)
                out_color = int(out_color)

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        color_map = None
                        break
                else:
                    color_map[in_color] = out_color

            if color_map:
                color_maps.append(frozenset(color_map.items()))

        if color_maps and len(set(color_maps)) == 1:
            color_map = dict(color_maps[0])
            if not all(k == v for k, v in color_map.items()):
                return {
                    'type': 'color_mapping',
                    'mapping': color_map
                }
        return None

    def _detect_object_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect object-based transformation"""
        from arc_object_detector import ObjectTransformationDetector

        detector = ObjectTransformationDetector()
        transformations_by_type = {}

        for input_grid, output_grid in train_pairs:
            result = detector.detect_transformation(input_grid, output_grid)

            if result['transformations']:
                for trans in result['transformations']:
                    trans_type = trans['type']
                    if trans_type not in transformations_by_type:
                        transformations_by_type[trans_type] = []
                    transformations_by_type[trans_type].append(trans)

        # Find consistent transformation
        for trans_type, trans_list in transformations_by_type.items():
            if len(trans_list) == len(train_pairs):
                if trans_type == 'uniform_movement':
                    deltas = [t['delta'] for t in trans_list]
                    if len(set(deltas)) == 1:
                        return {
                            'type': 'object',
                            'operation': 'movement',
                            'delta': deltas[0]
                        }
                elif trans_type == 'object_copying':
                    return {
                        'type': 'object',
                        'operation': 'copying',
                        'copies': trans_list[0]['copies']
                    }
                elif trans_type == 'object_recoloring':
                    color_maps = [t['color_map'] for t in trans_list]
                    if len(set([str(sorted(cm.items())) for cm in color_maps])) == 1:
                        return {
                            'type': 'object',
                            'operation': 'recoloring',
                            'color_map': color_maps[0]
                        }
        return None

    def _detect_symmetry_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect symmetry transformation"""
        detector = SymmetryDetector()
        symmetries = []

        for input_grid, output_grid in train_pairs:
            result = detector.detect_symmetry_transformation(input_grid, output_grid)
            if result:
                symmetries.append(result)

        if symmetries and len(symmetries) == len(train_pairs):
            operations = [s['name'] for s in symmetries]
            if len(set(operations)) == 1:
                sym = symmetries[0]
                return {
                    'type': 'symmetry',
                    'operation': sym['name'],
                    'parameters': sym.get('parameters', {})
                }
        return None

    def _detect_repetition_pattern(self, train_pairs) -> Optional[Dict]:
        """Detect repetition transformation"""
        detector = RepetitionDetector()
        repetitions = []

        for input_grid, output_grid in train_pairs:
            result = detector.detect_repetition(input_grid, output_grid)
            if result:
                repetitions.append(result)

        if repetitions and len(repetitions) == len(train_pairs):
            operations = [r['name'] for r in repetitions]
            if len(set(operations)) == 1:
                rep = repetitions[0]
                return {
                    'type': 'repetition',
                    'operation': rep['name'],
                    'parameters': rep.get('parameters', {})
                }
        return None

    def _apply_pattern(self, input_grid: List[List[int]],
                      pattern: Dict) -> Optional[List[List[int]]]:
        """
        Apply detected pattern to input grid

        Args:
            input_grid: Test input grid
            pattern: Detected pattern dictionary

        Returns:
            Output grid after transformation
        """
        pattern_type = pattern['type']

        if pattern_type == 'scaling':
            return self._apply_scaling(input_grid, pattern)
        elif pattern_type == 'spatial':
            return self._apply_spatial(input_grid, pattern)
        elif pattern_type == 'tiling':
            return self._apply_tiling(input_grid, pattern)
        elif pattern_type == 'color_mapping':
            return self._apply_color_mapping(input_grid, pattern)
        elif pattern_type == 'object':
            return self._apply_object_transformation(input_grid, pattern)
        elif pattern_type == 'symmetry':
            return self._apply_symmetry(input_grid, pattern)
        elif pattern_type == 'repetition':
            return self._apply_repetition(input_grid, pattern)

        return None

    def _apply_scaling(self, grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply scaling transformation - tries fractal substitution first"""
        input_np = np.array(grid)
        scale_h = pattern['scale_h']
        scale_w = pattern['scale_w']

        in_h, in_w = input_np.shape
        out_h = int(in_h * scale_h)
        out_w = int(in_w * scale_w)

        # Try fractal substitution for uniform integer scaling
        if scale_h == scale_w and scale_h == int(scale_h) and scale_h >= 2:
            scale_factor = int(scale_h)

            # Fractal substitution: replace each cell with scaled copy of entire input
            # Background (0) stays 0, non-background gets the input pattern
            output_np = np.zeros((out_h, out_w), dtype=int)

            for i in range(in_h):
                for j in range(in_w):
                    color = input_np[i, j]
                    block_i = i * scale_factor
                    block_j = j * scale_factor

                    if color != 0:
                        # Non-background: place copy of entire input
                        # Only if input fits in the block
                        if in_h <= scale_factor and in_w <= scale_factor:
                            output_np[block_i:block_i+in_h,
                                     block_j:block_j+in_w] = input_np

            return output_np.tolist()

        # Fallback: simple nearest-neighbor scaling for non-uniform or fractional scaling
        output_np = np.zeros((out_h, out_w), dtype=int)
        for i in range(out_h):
            for j in range(out_w):
                src_i = int(i / scale_h)
                src_j = int(j / scale_w)
                output_np[i, j] = input_np[src_i, src_j]

        return output_np.tolist()

    def _apply_spatial(self, grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply spatial transformation"""
        input_np = np.array(grid)
        operation = pattern['operation']

        if operation == 'horizontal_flip':
            output_np = np.fliplr(input_np)
        elif operation == 'vertical_flip':
            output_np = np.flipud(input_np)
        elif operation == 'rotate_90':
            output_np = np.rot90(input_np, 1)
        elif operation == 'rotate_180':
            output_np = np.rot90(input_np, 2)
        elif operation == 'rotate_270':
            output_np = np.rot90(input_np, 3)
        elif operation == 'transpose':
            output_np = input_np.T
        else:
            return grid

        return output_np.tolist()

    def _apply_tiling(self, grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply tiling transformation"""
        input_np = np.array(grid)
        tile_h = pattern['tile_h']
        tile_w = pattern['tile_w']

        output_np = np.tile(input_np, (tile_h, tile_w))
        return output_np.tolist()

    def _apply_color_mapping(self, grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply color mapping"""
        input_np = np.array(grid)
        mapping = pattern['mapping']

        output_np = input_np.copy()
        for old_color, new_color in mapping.items():
            output_np[input_np == old_color] = new_color

        return output_np.tolist()

    def _apply_object_transformation(self, grid: List[List[int]], pattern: Dict) -> Optional[List[List[int]]]:
        """Apply object transformation - simplified for common cases"""
        # Note: Full object transformation is complex
        # This is a simplified version for the most common case (copying)

        operation = pattern.get('operation')

        if operation == 'copying':
            # For copying, we'd need to detect objects and duplicate them
            # This is complex - for now return None (unsupported)
            return None
        elif operation == 'recoloring':
            # Apply color map to objects
            color_map = pattern['color_map']
            return self._apply_color_mapping(grid, {'mapping': color_map})
        elif operation == 'movement':
            # Move objects - also complex, return None for now
            return None

        return None

    def _apply_symmetry(self, grid: List[List[int]], pattern: Dict) -> Optional[List[List[int]]]:
        """Apply symmetry transformation"""
        input_np = np.array(grid)
        operation = pattern['operation']

        if operation == 'horizontal_reflection_expansion':
            # Create [input | flipped_input]
            flipped = np.fliplr(input_np)
            output_np = np.hstack([input_np, flipped])
        elif operation == 'vertical_reflection_expansion':
            # Create [input; flipped_input]
            flipped = np.flipud(input_np)
            output_np = np.vstack([input_np, flipped])
        elif operation == 'quadrant_reflection':
            # Create 2x2 grid of reflections
            tl = input_np
            tr = np.fliplr(input_np)
            bl = np.flipud(input_np)
            br = np.flipud(np.fliplr(input_np))
            top = np.hstack([tl, tr])
            bottom = np.hstack([bl, br])
            output_np = np.vstack([top, bottom])
        elif operation == 'create_vertical_symmetry':
            # Mirror top half
            h, w = input_np.shape
            mid = h // 2
            top_half = input_np[:mid, :]
            output_np = np.vstack([top_half, np.flipud(top_half)])
        elif operation == 'create_horizontal_symmetry':
            # Mirror left half
            h, w = input_np.shape
            mid = w // 2
            left_half = input_np[:, :mid]
            output_np = np.hstack([left_half, np.fliplr(left_half)])
        elif operation == 'pattern_completion':
            # Complete partial symmetric pattern
            params = pattern.get('parameters', {})
            symmetry_types = params.get('symmetry_types', [])

            output_np = input_np.copy()

            # Apply the most specific symmetry type
            if 'vertical' in symmetry_types:
                # Complete vertical symmetry - mirror top to bottom
                h, w = input_np.shape
                mid = h // 2

                # Copy non-zero values from top to bottom (mirror)
                for i in range(mid):
                    mirror_i = h - 1 - i
                    for j in range(w):
                        if output_np[i, j] != 0 and output_np[mirror_i, j] == 0:
                            output_np[mirror_i, j] = output_np[i, j]
                        elif output_np[mirror_i, j] != 0 and output_np[i, j] == 0:
                            output_np[i, j] = output_np[mirror_i, j]

            if 'horizontal' in symmetry_types:
                # Complete horizontal symmetry - mirror left to right
                h, w = input_np.shape
                mid = w // 2

                for j in range(mid):
                    mirror_j = w - 1 - j
                    for i in range(h):
                        if output_np[i, j] != 0 and output_np[i, mirror_j] == 0:
                            output_np[i, mirror_j] = output_np[i, j]
                        elif output_np[i, mirror_j] != 0 and output_np[i, j] == 0:
                            output_np[i, j] = output_np[i, mirror_j]
        else:
            return None

        return output_np.tolist()

    def _apply_repetition(self, grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply repetition transformation"""
        params = pattern.get('parameters', {})

        if 'repetitions' in params:
            reps = params['repetitions']
            axis = params.get('axis', 'horizontal')
            input_np = np.array(grid)

            if axis == 'horizontal':
                output_np = np.tile(input_np, (1, reps))
            else:  # vertical
                output_np = np.tile(input_np, (reps, 1))
        elif 'rows' in params and 'cols' in params:
            # Grid tiling
            input_np = np.array(grid)
            output_np = np.tile(input_np, (params['rows'], params['cols']))
        else:
            return grid

        return output_np.tolist()

    def close(self):
        """Close database connection"""
        self.conn.close()


class ARCSolverWithLearning(ARCSolver):
    """
    ARC Solver with continual learning capabilities

    Improves performance over time by:
    1. Tracking pattern success/failure rates
    2. Updating pattern confidence based on results
    3. Learning from successful solutions
    4. Prioritizing high-confidence patterns
    """

    def __init__(self, db_path: str = 'arc_learned_full.db'):
        super().__init__(db_path)
        self._init_learning_tables()
        self._last_used_pattern = None
        self._last_puzzle_id = None

    def _init_learning_tables(self):
        """Initialize database tables for learning"""
        # Pattern success tracking
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_performance (
                pattern_hash TEXT PRIMARY KEY,
                pattern_type TEXT,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Puzzle solutions for retraining
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_solutions (
                puzzle_id TEXT PRIMARY KEY,
                pattern_hash TEXT,
                input_grid TEXT,
                output_grid TEXT,
                verified INTEGER DEFAULT 0,
                added_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    def _pattern_hash(self, pattern: Dict) -> str:
        """Create unique hash for a pattern"""
        import hashlib
        pattern_str = json.dumps(pattern, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()

    def _get_pattern_confidence(self, pattern_hash: str) -> float:
        """Get confidence score for a pattern"""
        self.cursor.execute('''
            SELECT confidence FROM pattern_performance WHERE pattern_hash = ?
        ''', (pattern_hash,))

        result = self.cursor.fetchone()
        return result[0] if result else 0.5  # Default confidence

    def _update_pattern_confidence(self, pattern_hash: str, pattern_type: str, success: bool):
        """Update pattern confidence based on success/failure"""
        # Get current stats
        self.cursor.execute('''
            SELECT successes, failures FROM pattern_performance WHERE pattern_hash = ?
        ''', (pattern_hash,))

        result = self.cursor.fetchone()

        if result:
            successes, failures = result
        else:
            successes, failures = 0, 0

        # Update counts
        if success:
            successes += 1
        else:
            failures += 1

        # Calculate new confidence (success rate with smoothing)
        total = successes + failures
        confidence = (successes + 1) / (total + 2)  # Laplace smoothing

        # Store updated stats
        self.cursor.execute('''
            INSERT OR REPLACE INTO pattern_performance
            (pattern_hash, pattern_type, successes, failures, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (pattern_hash, pattern_type, successes, failures, confidence))

        self.conn.commit()

    def _add_solution_to_training(self, puzzle_id: str, pattern_hash: str,
                                   input_grid: List[List[int]], output_grid: List[List[int]]):
        """Add a successful solution to the training set"""
        input_json = json.dumps(input_grid)
        output_json = json.dumps(output_grid)

        self.cursor.execute('''
            INSERT OR REPLACE INTO learned_solutions
            (puzzle_id, pattern_hash, input_grid, output_grid, verified)
            VALUES (?, ?, ?, ?, 1)
        ''', (puzzle_id, pattern_hash, input_json, output_json))

        self.conn.commit()

    def solve_and_learn(self, puzzle: ARCPuzzle, expected_output: Optional[List[List[int]]] = None) -> Optional[List[List[int]]]:
        """
        Solve puzzle and learn from the result

        Args:
            puzzle: ARCPuzzle to solve
            expected_output: Expected output for validation (optional)

        Returns:
            Predicted output grid, or None if no solution found
        """
        # Store puzzle ID for tracking
        self._last_puzzle_id = puzzle.puzzle_id

        # Solve the puzzle
        result = self.solve(puzzle)

        # Learn from the result if we have expected output
        if expected_output and self._last_used_pattern:
            pattern_hash = self._pattern_hash(self._last_used_pattern)
            pattern_type = self._last_used_pattern.get('type', 'unknown')

            if result:
                is_correct = (result == expected_output)

                # Update pattern confidence
                self._update_pattern_confidence(pattern_hash, pattern_type, is_correct)

                if is_correct:
                    # Add successful solution to training set
                    test_input = puzzle.get_test_inputs()[0]
                    self._add_solution_to_training(puzzle.puzzle_id, pattern_hash,
                                                   test_input, result)
            else:
                # Pattern detected but failed to apply
                self._update_pattern_confidence(pattern_hash, pattern_type, False)

        return result

    def _apply_pattern(self, input_grid: List[List[int]], pattern: Dict) -> Optional[List[List[int]]]:
        """Apply pattern and track which pattern was used"""
        self._last_used_pattern = pattern
        return super()._apply_pattern(input_grid, pattern)

    def _detect_pattern_from_examples(self, train_pairs: List[Tuple]) -> Optional[Dict]:
        """
        Detect pattern with confidence-based prioritization

        Tries detectors in order of their learned confidence scores
        """
        # Get confidence scores for each pattern type
        pattern_type_confidence = {}

        self.cursor.execute('''
            SELECT pattern_type, AVG(confidence) as avg_conf
            FROM pattern_performance
            GROUP BY pattern_type
        ''')

        for row in self.cursor.fetchall():
            pattern_type, avg_conf = row
            pattern_type_confidence[pattern_type] = avg_conf

        # Define detectors
        detectors = [
            ('scaling', self._detect_scaling_pattern),
            ('spatial', self._detect_spatial_pattern),
            ('tiling', self._detect_tiling_pattern),
            ('color_mapping', self._detect_color_pattern),
            ('object', self._detect_object_pattern),
            ('symmetry', self._detect_symmetry_pattern),
            ('repetition', self._detect_repetition_pattern),
        ]

        # Sort detectors by confidence (highest first)
        # Use default confidence of 0.5 for unseen pattern types
        detectors_sorted = sorted(
            detectors,
            key=lambda x: pattern_type_confidence.get(x[0], 0.5),
            reverse=True
        )

        # Try each detector in confidence order
        for detector_type, detector_func in detectors_sorted:
            pattern = detector_func(train_pairs)
            if pattern:
                return pattern

        return None

    def get_learning_stats(self) -> Dict:
        """Get statistics about learning progress"""
        self.cursor.execute('''
            SELECT
                pattern_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                SUM(successes) as total_successes,
                SUM(failures) as total_failures
            FROM pattern_performance
            GROUP BY pattern_type
            ORDER BY avg_confidence DESC
        ''')

        stats = {}
        for row in self.cursor.fetchall():
            pattern_type, count, avg_conf, successes, failures = row
            stats[pattern_type] = {
                'patterns_tracked': count,
                'avg_confidence': avg_conf,
                'total_successes': successes,
                'total_failures': failures,
                'success_rate': successes / (successes + failures) if (successes + failures) > 0 else 0
            }

        return stats

    def get_learned_solutions_count(self) -> int:
        """Get count of learned solutions"""
        self.cursor.execute('SELECT COUNT(*) FROM learned_solutions WHERE verified = 1')
        return self.cursor.fetchone()[0]


def main():
    """Demo: Solve sample puzzles"""
    print("="*70)
    print("ARC Solver Demo")
    print("="*70)

    # Load dataset and solver
    loader = ARCDatasetLoader('../arc_dataset/data')
    solver = ARCSolver('arc_learned_full.db')

    # Try to solve first 10 puzzles
    puzzles = loader.load_training_set()[:10]

    solved = 0
    for i, puzzle in enumerate(puzzles, 1):
        print(f"\nPuzzle {i}: {puzzle.puzzle_id}")

        solution = solver.solve(puzzle)

        if solution:
            # Check if correct (if we have expected output)
            expected = puzzle.get_test_outputs()
            if expected and len(expected) > 0:
                if solution == expected[0]:
                    print("  ✓ SOLVED CORRECTLY!")
                    solved += 1
                else:
                    print("  ✗ Solution found but incorrect")
            else:
                print("  ? Solution generated (no expected output to verify)")
                solved += 1
        else:
            print("  ✗ No solution found")

    print(f"\n{'='*70}")
    print(f"Results: {solved}/{len(puzzles)} puzzles solved")
    print(f"Success rate: {solved/len(puzzles)*100:.1f}%")
    print(f"{'='*70}")

    solver.close()


if __name__ == '__main__':
    main()
