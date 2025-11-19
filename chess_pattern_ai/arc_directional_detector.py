#!/usr/bin/env python3
"""
Directional Pattern Detection for ARC

User's insight: "Find starting point, pick direction, count angle changes.
Compare Image 1 vs Image 2 - what pattern emerges?"

This uses the SAME observation-based approach as GameObserver for chess/checkers:
- No semantic understanding
- Just compare sequences and detect patterns
- Build confidence from observations
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque

class DirectionalPatternDetector:
    """
    Detect transformation patterns by comparing directional features

    Like chess pattern detection:
    - Observe configuration in input
    - Observe configuration in output
    - Detect what changed (and how)
    - Store pattern for future matching
    """

    def __init__(self):
        self.observed_patterns = {}

    def trace_boundary(self, grid: np.ndarray, start_pos: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Trace boundary of non-zero region, recording directions

        Like user described:
        - Find starting point
        - Pick direction
        - Count changes in direction
        - Return sequence: ['R', 'R', 'D', 'D', 'L', 'L', 'U', 'U']
        """
        h, w = grid.shape

        # Find non-zero pixels
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) == 0:
            return []

        # Find starting point (top-left non-zero pixel)
        if start_pos is None:
            start_pos = tuple(nonzero[0])

        # Trace boundary using 4-directional moves
        directions = []
        visited = set()
        current = start_pos

        # Direction mapping
        DIR_MAP = {
            (0, 1): 'R',   # Right
            (0, -1): 'L',  # Left
            (1, 0): 'D',   # Down
            (-1, 0): 'U'   # Up
        }

        # BFS to trace connected component
        queue = deque([current])
        boundary_points = []

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or grid[r, c] == 0:
                continue

            visited.add((r, c))

            # Check if this is a boundary point (has background neighbor)
            is_boundary = False
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] == 0:
                        is_boundary = True
                    elif (nr, nc) not in visited:
                        queue.append((nr, nc))

            if is_boundary:
                boundary_points.append((r, c))

        # Sort boundary points to create consistent tracing order
        if not boundary_points:
            return []

        # Start from top-left boundary point
        boundary_points.sort()
        current = boundary_points[0]
        traced = [current]
        remaining = set(boundary_points) - {current}

        # Trace boundary by finding nearest neighbors
        while remaining:
            r, c = current
            best_next = None
            best_dist = float('inf')

            # Find nearest unvisited boundary point
            for nr, nc in remaining:
                dist = abs(nr - r) + abs(nc - c)
                if dist < best_dist and dist <= 2:  # Must be adjacent or diagonal
                    best_dist = dist
                    best_next = (nr, nc)

            if best_next is None:
                break

            # Record direction to next point
            dr = best_next[0] - current[0]
            dc = best_next[1] - current[1]

            # Normalize to cardinal direction
            if abs(dr) >= abs(dc):
                direction = (1 if dr > 0 else -1, 0) if dr != 0 else (0, 1 if dc > 0 else -1)
            else:
                direction = (0, 1 if dc > 0 else -1)

            if direction in DIR_MAP:
                directions.append(DIR_MAP[direction])

            current = best_next
            traced.append(current)
            remaining.remove(current)

        return directions

    def compare_directional_sequences(self, seq1: List[str], seq2: List[str]) -> Dict:
        """
        Compare two directional sequences to detect pattern

        Like user described:
        - Same sequence → No change
        - Reversed horizontal → Horizontal reflection
        - Reversed vertical → Vertical reflection
        - Rotated → Rotation transformation
        """
        if not seq1 or not seq2 or len(seq1) != len(seq2):
            return {'pattern': 'unknown', 'confidence': 0.0}

        # Check for horizontal reversal (R↔L, U/D same)
        h_reversed = all(
            (s1 == 'R' and s2 == 'L') or
            (s1 == 'L' and s2 == 'R') or
            (s1 in ['U', 'D'] and s1 == s2)
            for s1, s2 in zip(seq1, seq2)
        )

        if h_reversed:
            return {
                'pattern': 'horizontal_reflection',
                'confidence': 1.0,
                'description': 'Left-right directions reversed'
            }

        # Check for vertical reversal (U↔D, L/R same)
        v_reversed = all(
            (s1 == 'U' and s2 == 'D') or
            (s1 == 'D' and s2 == 'U') or
            (s1 in ['L', 'R'] and s1 == s2)
            for s1, s2 in zip(seq1, seq2)
        )

        if v_reversed:
            return {
                'pattern': 'vertical_reflection',
                'confidence': 1.0,
                'description': 'Up-down directions reversed'
            }

        # Check for rotation (all directions rotated 90°)
        ROTATE_90 = {'R': 'D', 'D': 'L', 'L': 'U', 'U': 'R'}
        rotated_90 = all(ROTATE_90.get(s1) == s2 for s1, s2 in zip(seq1, seq2))

        if rotated_90:
            return {
                'pattern': 'rotation_90',
                'confidence': 1.0,
                'description': 'Rotated 90 degrees clockwise'
            }

        # Check for 180° rotation
        ROTATE_180 = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}
        rotated_180 = all(ROTATE_180.get(s1) == s2 for s1, s2 in zip(seq1, seq2))

        if rotated_180:
            return {
                'pattern': 'rotation_180',
                'confidence': 1.0,
                'description': 'Rotated 180 degrees'
            }

        # Check if sequences are identical
        if seq1 == seq2:
            return {
                'pattern': 'no_change',
                'confidence': 1.0,
                'description': 'Boundary unchanged'
            }

        return {'pattern': 'unknown', 'confidence': 0.0}

    def detect_transformation_pattern(self, input_grid, output_grid) -> Dict:
        """
        Detect transformation by comparing directional features

        GameObserver approach:
        1. Extract features from input
        2. Extract features from output
        3. Compare to find pattern
        4. Return pattern with confidence
        """
        inp = np.array(input_grid)
        out = np.array(output_grid)

        # If sizes differ, it's a scaling/tiling pattern
        if inp.shape != out.shape:
            return {
                'pattern': 'size_change',
                'confidence': 1.0,
                'details': {
                    'from': inp.shape,
                    'to': out.shape,
                    'ratio': (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
                }
            }

        # Trace directional features
        input_directions = self.trace_boundary(inp)
        output_directions = self.trace_boundary(out)

        # Compare sequences
        pattern = self.compare_directional_sequences(input_directions, output_directions)

        # Add metadata
        pattern['input_sequence'] = input_directions[:10]  # First 10 for debugging
        pattern['output_sequence'] = output_directions[:10]

        return pattern

    def observe_and_learn(self, puzzle_id: str, training_pairs: List[Tuple]) -> Dict:
        """
        Observe puzzle and learn meta-pattern

        Like GameObserver for chess:
        - Observe multiple examples
        - Find consistent pattern
        - Store with confidence score
        """
        patterns = []

        for input_grid, output_grid in training_pairs:
            pattern = self.detect_transformation_pattern(input_grid, output_grid)
            patterns.append(pattern)

        # Find consensus pattern
        if not patterns:
            return {'pattern': 'unknown', 'confidence': 0.0}

        # Check if all examples show same pattern
        pattern_types = [p['pattern'] for p in patterns]
        if len(set(pattern_types)) == 1:
            # All examples agree!
            consensus = patterns[0].copy()
            consensus['confidence'] = 1.0
            consensus['examples'] = len(patterns)
            return consensus
        else:
            # Mixed patterns - lower confidence
            from collections import Counter
            most_common = Counter(pattern_types).most_common(1)[0]
            return {
                'pattern': most_common[0],
                'confidence': most_common[1] / len(patterns),
                'examples': len(patterns)
            }


if __name__ == '__main__':
    from arc_puzzle import ARCDatasetLoader

    detector = DirectionalPatternDetector()
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Test on reflection puzzle (user's example)
    print("Testing directional pattern detection:")
    print("="*70)

    test_puzzles = [
        ('3c9b0459', 'rotation'),
        ('0962bcdd', 'reflection'),
        ('6150a2bd', 'flip'),
    ]

    for puzzle_id, expected in test_puzzles:
        puzzle = loader.load_puzzle(puzzle_id, 'training')

        print(f"\nPuzzle {puzzle_id} (expected: {expected}):")
        print("-"*70)

        pattern = detector.observe_and_learn(puzzle_id, puzzle.get_train_pairs())

        print(f"Detected pattern: {pattern['pattern']}")
        print(f"Confidence: {pattern['confidence']:.2f}")
        if 'description' in pattern:
            print(f"Description: {pattern['description']}")
        if 'input_sequence' in pattern:
            print(f"Input directions: {pattern['input_sequence']}")
            print(f"Output directions: {pattern['output_sequence']}")

        if pattern['pattern'] != 'unknown':
            print(f"✓ Pattern detected!")
        else:
            print(f"✗ Pattern unclear")
