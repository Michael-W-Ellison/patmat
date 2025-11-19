#!/usr/bin/env python3
"""
Test: Simple comparison-based approach to object transformations

Key insight from user: Don't try to understand "hollow shapes" semantically.
Instead, just compare input vs output and find the pattern:

1. Find what stays the same (boundary pixels)
2. Find what changes (interior pixels)
3. Detect the rule: enclosed regions get filled
4. Apply to test case
"""

from arc_puzzle import ARCDatasetLoader
from arc_object_detector import ARCObjectDetector
import numpy as np

def analyze_transformation_by_comparison(input_grid, output_grid):
    """
    Compare input and output to find transformation pattern

    Returns:
        - unchanged_mask: pixels that stayed the same
        - changed_mask: pixels that changed
        - change_pattern: what kind of change (fill, boundary, etc.)
    """
    inp = np.array(input_grid)
    out = np.array(output_grid)

    if inp.shape != out.shape:
        return None  # Can't compare different sizes

    # Find what changed
    changed_mask = (inp != out)
    unchanged_mask = (inp == out)

    # Analyze the changes
    analysis = {
        'total_cells': inp.size,
        'changed_cells': np.sum(changed_mask),
        'changed_percentage': np.sum(changed_mask) / inp.size * 100,
        'unchanged_cells': np.sum(unchanged_mask),
    }

    # Find colors in unchanged regions
    unchanged_colors = set(inp[unchanged_mask].flatten()) - {0}

    # Find new colors in changed regions
    input_changed_colors = set(inp[changed_mask].flatten())
    output_changed_colors = set(out[changed_mask].flatten())

    analysis['boundary_colors'] = unchanged_colors
    analysis['changed_from'] = input_changed_colors
    analysis['changed_to'] = output_changed_colors

    # Check if changed regions are enclosed by unchanged regions
    # Simple test: are changed pixels surrounded by non-zero unchanged pixels?

    return analysis, changed_mask, unchanged_mask

def detect_enclosed_region_fill(training_pairs):
    """
    Detect if pattern is: fill enclosed regions with new color

    Pattern:
    - Some pixels stay the same (form boundary)
    - Interior pixels (enclosed by boundary) change color
    - Exterior pixels stay black
    """

    for input_grid, output_grid in training_pairs:
        analysis, changed, unchanged = analyze_transformation_by_comparison(input_grid, output_grid)

        if analysis is None:
            continue

        inp = np.array(input_grid)
        out = np.array(output_grid)

        print(f"\nTraining example analysis:")
        print(f"  Changed: {analysis['changed_cells']}/{analysis['total_cells']} ({analysis['changed_percentage']:.1f}%)")
        print(f"  Boundary colors (unchanged): {analysis['boundary_colors']}")
        print(f"  Changed from: {analysis['changed_from']}")
        print(f"  Changed to: {analysis['changed_to']}")

        # Check if changed regions are enclosed
        # Use flood fill from edges - if we can't reach a changed pixel, it's enclosed
        h, w = inp.shape

        # Mark exterior (reachable from edges via background)
        exterior_mask = np.zeros((h, w), dtype=bool)
        visited = np.zeros((h, w), dtype=bool)

        # Flood fill from all edges
        from collections import deque
        queue = deque()

        # Add all edge cells with value 0
        for i in range(h):
            if inp[i, 0] == 0:
                queue.append((i, 0))
            if inp[i, w-1] == 0:
                queue.append((i, w-1))
        for j in range(w):
            if inp[0, j] == 0:
                queue.append((0, j))
            if inp[h-1, j] == 0:
                queue.append((h-1, j))

        # Flood fill to find all exterior cells
        while queue:
            i, j = queue.popleft()
            if visited[i, j] or inp[i, j] != 0:
                continue

            visited[i, j] = True
            exterior_mask[i, j] = True

            # Add neighbors
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                    queue.append((ni, nj))

        # Interior = background pixels NOT reachable from edges
        interior_mask = (inp == 0) & ~exterior_mask

        print(f"  Interior cells: {np.sum(interior_mask)}")
        print(f"  Exterior cells: {np.sum(exterior_mask)}")

        # Check if changed cells match interior cells
        changed_is_interior = np.array_equal(changed & (inp == 0), interior_mask)

        print(f"  Changed cells are enclosed interior: {changed_is_interior}")

        if changed_is_interior:
            # Found the pattern!
            fill_color = list(analysis['changed_to'] - {0})[0] if len(analysis['changed_to'] - {0}) > 0 else None
            print(f"  ✓ PATTERN DETECTED: Fill interior with color {fill_color}")

            return {
                'type': 'fill_enclosed_regions',
                'fill_color': fill_color,
                'boundary_colors': list(analysis['boundary_colors'])
            }

    return None

def apply_fill_enclosed_regions(test_input, pattern):
    """Apply fill enclosed regions transformation"""
    inp = np.array(test_input)
    h, w = inp.shape

    # Flood fill from edges to find exterior
    from collections import deque
    exterior_mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)

    queue = deque()

    # Add all edge cells with value 0
    for i in range(h):
        if inp[i, 0] == 0:
            queue.append((i, 0))
        if inp[i, w-1] == 0:
            queue.append((i, w-1))
    for j in range(w):
        if inp[0, j] == 0:
            queue.append((0, j))
        if inp[h-1, j] == 0:
            queue.append((h-1, j))

    while queue:
        i, j = queue.popleft()
        if visited[i, j] or inp[i, j] != 0:
            continue

        visited[i, j] = True
        exterior_mask[i, j] = True

        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                queue.append((ni, nj))

    # Interior = background pixels NOT reachable from edges
    interior_mask = (inp == 0) & ~exterior_mask

    # Fill interior
    output = inp.copy()
    fill_color = pattern.get('fill_color')

    if fill_color is not None:
        output[interior_mask] = fill_color

    return output.tolist()


if __name__ == '__main__':
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Test on "fill hollow shapes" puzzle
    puzzle = loader.load_puzzle('00d62c1b', 'training')

    print("="*70)
    print("Testing comparison-based approach on puzzle 00d62c1b")
    print("="*70)

    # Detect pattern from training
    pattern = detect_enclosed_region_fill(puzzle.get_train_pairs())

    if pattern:
        print("\n" + "="*70)
        print("Applying to test case:")
        print("="*70)

        test_input = puzzle.get_test_inputs()[0]
        expected_output = puzzle.get_test_outputs()[0]

        predicted = apply_fill_enclosed_regions(test_input, pattern)

        if predicted == expected_output:
            print("✓ CORRECT!")
        else:
            pred_np = np.array(predicted)
            exp_np = np.array(expected_output)
            diff = np.sum(pred_np != exp_np)
            accuracy = (1 - diff / pred_np.size) * 100
            print(f"✗ Incorrect: {accuracy:.1f}% accurate ({diff}/{pred_np.size} cells wrong)")

            print("\nExpected:")
            print(exp_np)
            print("\nPredicted:")
            print(pred_np)
    else:
        print("✗ Pattern not detected")
