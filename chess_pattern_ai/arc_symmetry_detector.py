#!/usr/bin/env python3
"""
ARC Symmetry Detector

Detects symmetry operations and pattern completion in ARC puzzles.
Many ARC puzzles involve:
- Reflecting patterns across axes
- Completing partial symmetric patterns
- Creating mirror images
- Rotational symmetry
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set


class SymmetryDetector:
    """
    Detects symmetry patterns in grids and transformations

    Symmetry types:
    - Horizontal reflection (flip left/right)
    - Vertical reflection (flip up/down)
    - Diagonal reflection (transpose)
    - Rotational symmetry (90°, 180°, 270°)
    - Pattern completion (fill missing symmetric parts)
    """

    def __init__(self):
        pass

    def detect_symmetry_transformation(self, input_grid: List[List[int]],
                                      output_grid: List[List[int]]) -> Optional[Dict]:
        """
        Detect if output is symmetrically transformed version of input

        Returns:
            Dictionary describing symmetry transformation, or None
        """
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)

        # Check for reflection to create larger output
        reflection = self._detect_reflection_expansion(input_np, output_np)
        if reflection:
            return reflection

        # Check for pattern completion
        completion = self._detect_pattern_completion(input_np, output_np)
        if completion:
            return completion

        # Check for symmetry creation
        symmetry = self._detect_symmetry_creation(input_np, output_np)
        if symmetry:
            return symmetry

        return None

    def _detect_reflection_expansion(self, input_grid: np.ndarray,
                                    output_grid: np.ndarray) -> Optional[Dict]:
        """
        Detect if input was reflected and combined with itself to create output

        Example: [ABC] → [ABCCBA] (horizontal reflection expansion)
        """
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Check horizontal reflection expansion (double width)
        if out_w == 2 * in_w and out_h == in_h:
            # Check if output = [input | flipped_input]
            left_half = output_grid[:, :in_w]
            right_half = output_grid[:, in_w:]

            if np.array_equal(left_half, input_grid) and \
               np.array_equal(right_half, np.fliplr(input_grid)):
                return {
                    'type': 'symmetry',
                    'name': 'horizontal_reflection_expansion',
                    'description': 'Expand input by adding horizontal reflection',
                    'parameters': {'axis': 'horizontal', 'operation': 'reflection_expansion'}
                }

        # Check vertical reflection expansion (double height)
        if out_h == 2 * in_h and out_w == in_w:
            # Check if output = [input; flipped_input]
            top_half = output_grid[:in_h, :]
            bottom_half = output_grid[in_h:, :]

            if np.array_equal(top_half, input_grid) and \
               np.array_equal(bottom_half, np.flipud(input_grid)):
                return {
                    'type': 'symmetry',
                    'name': 'vertical_reflection_expansion',
                    'description': 'Expand input by adding vertical reflection',
                    'parameters': {'axis': 'vertical', 'operation': 'reflection_expansion'}
                }

        # Check 4-way reflection (2x2 expansion)
        if out_h == 2 * in_h and out_w == 2 * in_w:
            # Check if output is 2x2 grid of reflections
            tl = output_grid[:in_h, :in_w]  # top-left
            tr = output_grid[:in_h, in_w:]  # top-right
            bl = output_grid[in_h:, :in_w]  # bottom-left
            br = output_grid[in_h:, in_w:]  # bottom-right

            if np.array_equal(tl, input_grid) and \
               np.array_equal(tr, np.fliplr(input_grid)) and \
               np.array_equal(bl, np.flipud(input_grid)) and \
               np.array_equal(br, np.flipud(np.fliplr(input_grid))):
                return {
                    'type': 'symmetry',
                    'name': 'quadrant_reflection',
                    'description': 'Create 4 reflected quadrants from input',
                    'parameters': {'operation': '4way_reflection'}
                }

        return None

    def _detect_pattern_completion(self, input_grid: np.ndarray,
                                   output_grid: np.ndarray) -> Optional[Dict]:
        """
        Detect if output completes a partial symmetric pattern in input

        Example: Input has half a symmetric pattern, output completes it
        """
        # Same dimensions required for completion
        if input_grid.shape != output_grid.shape:
            return None

        # Check if output is symmetric while input is not
        input_symmetric = self._check_symmetry(input_grid)
        output_symmetric = self._check_symmetry(output_grid)

        if not input_symmetric['any'] and output_symmetric['any']:
            # Output is symmetric but input is not - likely completion

            # Check which cells changed
            changed = input_grid != output_grid
            num_changed = np.sum(changed)
            total_cells = input_grid.size

            # If less than 50% of cells changed, it's likely completion
            if num_changed < total_cells * 0.5:
                symmetry_type = []
                if output_symmetric['horizontal']:
                    symmetry_type.append('horizontal')
                if output_symmetric['vertical']:
                    symmetry_type.append('vertical')
                if output_symmetric['diagonal']:
                    symmetry_type.append('diagonal')

                return {
                    'type': 'symmetry',
                    'name': 'pattern_completion',
                    'description': f'Complete {"/".join(symmetry_type)} symmetric pattern',
                    'parameters': {
                        'symmetry_types': symmetry_type,
                        'operation': 'completion',
                        'cells_modified': int(num_changed)
                    }
                }

        return None

    def _detect_symmetry_creation(self, input_grid: np.ndarray,
                                  output_grid: np.ndarray) -> Optional[Dict]:
        """
        Detect if output creates symmetric version of input

        Example: Asymmetric input → symmetric output (by mirroring one half)
        """
        if input_grid.shape != output_grid.shape:
            return None

        h, w = input_grid.shape

        # Check if output is horizontally symmetric
        if w >= 2 and np.array_equal(output_grid, np.fliplr(output_grid)):
            # Output is horizontally symmetric
            # Check if left half of output matches left half of input
            mid = w // 2
            if np.array_equal(output_grid[:, :mid], input_grid[:, :mid]):
                return {
                    'type': 'symmetry',
                    'name': 'create_horizontal_symmetry',
                    'description': 'Mirror left half to create horizontal symmetry',
                    'parameters': {'axis': 'horizontal', 'operation': 'symmetry_creation'}
                }
            # Check if right half of output matches right half of input
            elif np.array_equal(output_grid[:, -mid:], input_grid[:, -mid:]):
                return {
                    'type': 'symmetry',
                    'name': 'create_horizontal_symmetry',
                    'description': 'Mirror right half to create horizontal symmetry',
                    'parameters': {'axis': 'horizontal', 'operation': 'symmetry_creation'}
                }

        # Check if output is vertically symmetric
        if h >= 2 and np.array_equal(output_grid, np.flipud(output_grid)):
            # Output is vertically symmetric
            mid = h // 2
            if np.array_equal(output_grid[:mid, :], input_grid[:mid, :]):
                return {
                    'type': 'symmetry',
                    'name': 'create_vertical_symmetry',
                    'description': 'Mirror top half to create vertical symmetry',
                    'parameters': {'axis': 'vertical', 'operation': 'symmetry_creation'}
                }
            elif np.array_equal(output_grid[-mid:, :], input_grid[-mid:, :]):
                return {
                    'type': 'symmetry',
                    'name': 'create_vertical_symmetry',
                    'description': 'Mirror bottom half to create vertical symmetry',
                    'parameters': {'axis': 'vertical', 'operation': 'symmetry_creation'}
                }

        return None

    def _check_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """
        Check if grid has various types of symmetry

        Returns:
            Dictionary with symmetry flags
        """
        h, w = grid.shape

        # Horizontal symmetry (left-right mirror)
        horizontal = np.array_equal(grid, np.fliplr(grid)) if w >= 2 else False

        # Vertical symmetry (top-bottom mirror)
        vertical = np.array_equal(grid, np.flipud(grid)) if h >= 2 else False

        # Diagonal symmetry (transpose)
        diagonal = False
        if h == w:  # Only square grids can have diagonal symmetry
            diagonal = np.array_equal(grid, grid.T)

        # Rotational symmetry (180°)
        rotational_180 = np.array_equal(grid, np.rot90(grid, 2))

        return {
            'horizontal': horizontal,
            'vertical': vertical,
            'diagonal': diagonal,
            'rotational_180': rotational_180,
            'any': horizontal or vertical or diagonal or rotational_180
        }


class RepetitionDetector:
    """
    Detects counting and repetition operations

    Patterns:
    - Repeat input N times
    - Stack objects multiple times
    - Generate pattern based on counts
    """

    def __init__(self):
        pass

    def detect_repetition(self, input_grid: List[List[int]],
                         output_grid: List[List[int]]) -> Optional[Dict]:
        """
        Detect if output is repetition of input

        Returns:
            Dictionary describing repetition, or None
        """
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)

        # Check for horizontal repetition
        h_rep = self._detect_horizontal_repetition(input_np, output_np)
        if h_rep:
            return h_rep

        # Check for vertical repetition
        v_rep = self._detect_vertical_repetition(input_np, output_np)
        if v_rep:
            return v_rep

        # Check for grid repetition (NxM tiling)
        grid_rep = self._detect_grid_repetition(input_np, output_np)
        if grid_rep:
            return grid_rep

        return None

    def _detect_horizontal_repetition(self, input_grid: np.ndarray,
                                     output_grid: np.ndarray) -> Optional[Dict]:
        """Detect if input is repeated horizontally N times"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Check if output height matches and width is multiple
        if out_h != in_h or out_w % in_w != 0:
            return None

        n_reps = out_w // in_w

        # Check if each segment matches input
        for i in range(n_reps):
            segment = output_grid[:, i*in_w:(i+1)*in_w]
            if not np.array_equal(segment, input_grid):
                return None

        return {
            'type': 'repetition',
            'name': f'horizontal_repeat_{n_reps}x',
            'description': f'Repeat input {n_reps} times horizontally',
            'parameters': {'repetitions': int(n_reps), 'axis': 'horizontal'}
        }

    def _detect_vertical_repetition(self, input_grid: np.ndarray,
                                   output_grid: np.ndarray) -> Optional[Dict]:
        """Detect if input is repeated vertically N times"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Check if output width matches and height is multiple
        if out_w != in_w or out_h % in_h != 0:
            return None

        n_reps = out_h // in_h

        # Check if each segment matches input
        for i in range(n_reps):
            segment = output_grid[i*in_h:(i+1)*in_h, :]
            if not np.array_equal(segment, input_grid):
                return None

        return {
            'type': 'repetition',
            'name': f'vertical_repeat_{n_reps}x',
            'description': f'Repeat input {n_reps} times vertically',
            'parameters': {'repetitions': int(n_reps), 'axis': 'vertical'}
        }

    def _detect_grid_repetition(self, input_grid: np.ndarray,
                               output_grid: np.ndarray) -> Optional[Dict]:
        """Detect if input is repeated in NxM grid"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Check if output dimensions are multiples
        if out_h % in_h != 0 or out_w % in_w != 0:
            return None

        n_rows = out_h // in_h
        n_cols = out_w // in_w

        # Must be at least 2x2 to be interesting
        if n_rows * n_cols < 4:
            return None

        # Check if each tile matches input
        for i in range(n_rows):
            for j in range(n_cols):
                tile = output_grid[i*in_h:(i+1)*in_h, j*in_w:(j+1)*in_w]
                if not np.array_equal(tile, input_grid):
                    return None

        return {
            'type': 'repetition',
            'name': f'grid_repeat_{n_rows}x{n_cols}',
            'description': f'Tile input in {n_rows}×{n_cols} grid',
            'parameters': {'rows': int(n_rows), 'cols': int(n_cols), 'operation': 'tiling'}
        }


def main():
    """Demo: Symmetry and repetition detection"""
    print("="*70)
    print("Symmetry and Repetition Detection Demo")
    print("="*70)

    sym_detector = SymmetryDetector()
    rep_detector = RepetitionDetector()

    # Example 1: Horizontal reflection expansion
    print("\nExample 1: Horizontal reflection expansion")
    input1 = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    output1 = [
        [1, 2, 3, 3, 2, 1],
        [4, 5, 6, 6, 5, 4]
    ]

    result = sym_detector.detect_symmetry_transformation(input1, output1)
    if result:
        print(f"  → {result['description']}")

    # Example 2: Vertical repetition
    print("\nExample 2: Vertical repetition")
    input2 = [
        [1, 2],
        [3, 4]
    ]
    output2 = [
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4]
    ]

    result = rep_detector.detect_repetition(input2, output2)
    if result:
        print(f"  → {result['description']}")

    # Example 3: 4-way reflection
    print("\nExample 3: 4-way quadrant reflection")
    input3 = [
        [1, 2],
        [3, 0]
    ]
    output3 = [
        [1, 2, 2, 1],
        [3, 0, 0, 3],
        [3, 0, 0, 3],
        [1, 2, 2, 1]
    ]

    result = sym_detector.detect_symmetry_transformation(input3, output3)
    if result:
        print(f"  → {result['description']}")

    # Example 4: Grid tiling
    print("\nExample 4: Grid tiling (3x3)")
    input4 = [[5]]
    output4 = [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5]
    ]

    result = rep_detector.detect_repetition(input4, output4)
    if result:
        print(f"  → {result['description']}")

    print("\n" + "="*70)
    print("Symmetry and Repetition Detection Ready!")
    print("="*70)


if __name__ == '__main__':
    main()
