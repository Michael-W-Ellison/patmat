#!/usr/bin/env python3
"""
Transformation Signatures: Identify transformation type by comparing input/output

Key insight: Different transformations have different "signatures"
- Fill enclosed: Small % changes, all in enclosed regions
- Reflection: ~50% changes, mirror pattern
- Tiling: Large % changes, repetitive pattern
- Color remap: 100% changes for specific colors
"""

from arc_puzzle import ARCDatasetLoader
import numpy as np

def get_transformation_signature(input_grid, output_grid):
    """
    Analyze input/output to determine transformation type

    Returns signature dictionary with:
    - changed_percentage
    - spatial_pattern (clustered, symmetric, uniform, etc.)
    - color_pattern (what colors changed how)
    - size_change (did grid size change?)
    """
    inp = np.array(input_grid)
    out = np.array(output_grid)

    signature = {}

    # Size change?
    if inp.shape != out.shape:
        signature['size_change'] = {
            'from': inp.shape,
            'to': out.shape,
            'ratio': (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
        }
        signature['type_hints'] = ['scaling', 'tiling', 'expansion']
        return signature

    # No size change - analyze cell-by-cell
    changed = (inp != out)
    signature['changed_percentage'] = (np.sum(changed) / inp.size) * 100
    signature['size_change'] = None

    # Color analysis
    inp_colors = set(inp[changed].flatten())
    out_colors = set(out[changed].flatten())

    signature['colors_changed_from'] = inp_colors
    signature['colors_changed_to'] = out_colors

    # Spatial pattern of changes
    if np.sum(changed) < inp.size * 0.1:  # <10% changed
        signature['spatial_pattern'] = 'localized'  # Small region
        signature['type_hints'] = ['fill_enclosed', 'add_details']

    elif np.sum(changed) > inp.size * 0.4:  # >40% changed
        signature['spatial_pattern'] = 'widespread'  # Large region
        signature['type_hints'] = ['tiling', 'reflection', 'transformation']

    else:  # 10-40% changed
        signature['spatial_pattern'] = 'moderate'
        signature['type_hints'] = ['partial_fill', 'object_modification']

    # Symmetry check - are changes symmetric?
    changed_coords = np.argwhere(changed)
    if len(changed_coords) > 0:
        center_y, center_x = inp.shape[0] / 2, inp.shape[1] / 2

        # Check if changes are symmetric around center
        symmetric_matches = 0
        for y, x in changed_coords:
            mirror_y = int(2 * center_y - y)
            mirror_x = int(2 * center_x - x)
            if 0 <= mirror_y < inp.shape[0] and 0 <= mirror_x < inp.shape[1]:
                if changed[mirror_y, mirror_x]:
                    symmetric_matches += 1

        symmetry_ratio = symmetric_matches / len(changed_coords)
        signature['symmetry_ratio'] = symmetry_ratio

        if symmetry_ratio > 0.7:
            signature['type_hints'].append('reflection')
            signature['type_hints'].append('symmetry')

    return signature

def classify_transformation(training_pairs):
    """Classify what type of transformation this is"""

    signatures = []
    for inp, out in training_pairs:
        sig = get_transformation_signature(inp, out)
        signatures.append(sig)

    # Find common patterns
    first_sig = signatures[0]

    print("Transformation Signature Analysis:")
    print("="*70)
    print(f"Changed: {first_sig['changed_percentage']:.1f}%")
    print(f"Spatial pattern: {first_sig['spatial_pattern']}")
    print(f"Type hints: {first_sig['type_hints']}")

    if 'symmetry_ratio' in first_sig:
        print(f"Symmetry ratio: {first_sig['symmetry_ratio']:.2f}")

    if first_sig['size_change']:
        print(f"Size change: {first_sig['size_change']}")

    return first_sig['type_hints']

# Test on various puzzles
loader = ARCDatasetLoader('../arc_dataset/data')

puzzles_to_test = [
    ('00d62c1b', 'fill_enclosed'),
    ('0962bcdd', 'reflection'),
    ('05269061', 'tiling'),
    ('3c9b0459', 'rotation'),
]

print("\nClassifying multiple puzzles:")
print("="*70)

for puzzle_id, expected_type in puzzles_to_test:
    puzzle = loader.load_puzzle(puzzle_id, 'training')

    print(f"\nPuzzle {puzzle_id} (expected: {expected_type}):")
    print("-"*70)

    type_hints = classify_transformation(puzzle.get_train_pairs())

    if expected_type in [hint.lower().replace('_', '') for hint in type_hints]:
        print(f"✓ Correctly identified as {expected_type}")
    else:
        print(f"✗ Classified as {type_hints}, expected {expected_type}")
