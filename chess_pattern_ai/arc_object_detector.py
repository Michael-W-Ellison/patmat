#!/usr/bin/env python3
"""
ARC Object Detector - Connected Component Analysis

Detects discrete objects in ARC grids using connected component labeling.
Objects are defined as connected regions of the same color (excluding background).

This is the foundation for object-based pattern detection, which is crucial
for solving most ARC puzzles.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import deque


class ARCObject:
    """Represents a detected object in an ARC grid"""

    def __init__(self, cells: Set[Tuple[int, int]], color: int, grid_shape: Tuple[int, int]):
        self.cells = cells  # Set of (row, col) positions
        self.color = color
        self.grid_shape = grid_shape

        # Compute bounding box
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        self.min_row = min(rows)
        self.max_row = max(rows)
        self.min_col = min(cols)
        self.max_col = max(cols)

        self.height = self.max_row - self.min_row + 1
        self.width = self.max_col - self.min_col + 1
        self.size = len(cells)

    def __repr__(self):
        return f"Object(color={self.color}, size={self.size}, bbox=({self.height}x{self.width}))"

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Returns (min_row, min_col, height, width)"""
        return (self.min_row, self.min_col, self.height, self.width)

    def get_center(self) -> Tuple[float, float]:
        """Returns center position (row, col)"""
        row_center = sum(r for r, c in self.cells) / len(self.cells)
        col_center = sum(c for r, c in self.cells) / len(self.cells)
        return (row_center, col_center)

    def get_mask(self) -> np.ndarray:
        """Returns binary mask of object in its bounding box"""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for r, c in self.cells:
            mask[r - self.min_row, c - self.min_col] = True
        return mask

    def get_shape_grid(self) -> np.ndarray:
        """Returns grid showing object shape in its bounding box"""
        grid = np.zeros((self.height, self.width), dtype=int)
        for r, c in self.cells:
            grid[r - self.min_row, c - self.min_col] = self.color
        return grid

    def overlaps(self, other: 'ARCObject') -> bool:
        """Check if this object overlaps with another"""
        return bool(self.cells & other.cells)

    def distance_to(self, other: 'ARCObject') -> float:
        """Euclidean distance between object centers"""
        r1, c1 = self.get_center()
        r2, c2 = other.get_center()
        return ((r2 - r1)**2 + (c2 - c1)**2)**0.5

    def is_same_shape(self, other: 'ARCObject') -> bool:
        """Check if objects have same shape (ignoring position and color)"""
        if self.height != other.height or self.width != other.width:
            return False

        mask1 = self.get_mask()
        mask2 = other.get_mask()
        return np.array_equal(mask1, mask2)


class ARCObjectDetector:
    """
    Detects objects in ARC grids using connected component analysis

    Objects are defined as connected regions of the same non-background color.
    Background is typically color 0, but can be customized.
    """

    def __init__(self, background_color: int = 0, connectivity: int = 4):
        """
        Args:
            background_color: Color to treat as background (default 0)
            connectivity: 4 or 8 (4-connected or 8-connected)
        """
        self.background_color = background_color
        self.connectivity = connectivity

    def detect_objects(self, grid: List[List[int]]) -> List[ARCObject]:
        """
        Detect all objects in a grid

        Args:
            grid: 2D list of integers representing the grid

        Returns:
            List of ARCObject instances
        """
        grid_np = np.array(grid)
        height, width = grid_np.shape

        visited = np.zeros((height, width), dtype=bool)
        objects = []

        # Find all connected components
        for r in range(height):
            for c in range(width):
                if not visited[r, c]:
                    color = grid_np[r, c]

                    # Skip background
                    if color == self.background_color:
                        visited[r, c] = True
                        continue

                    # Flood fill to find connected component
                    cells = self._flood_fill(grid_np, r, c, color, visited)

                    if cells:
                        obj = ARCObject(cells, color, (height, width))
                        objects.append(obj)

        return objects

    def _flood_fill(self, grid: np.ndarray, start_r: int, start_c: int,
                    target_color: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Flood fill algorithm to find connected component

        Uses BFS to find all cells of target_color connected to start position
        """
        height, width = grid.shape
        cells = set()
        queue = deque([(start_r, start_c)])

        while queue:
            r, c = queue.popleft()

            # Skip if out of bounds or already visited
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            if visited[r, c]:
                continue

            # Skip if wrong color
            if grid[r, c] != target_color:
                continue

            # Add cell
            visited[r, c] = True
            cells.add((r, c))

            # Add neighbors
            if self.connectivity == 4:
                # 4-connected: up, down, left, right
                queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            else:
                # 8-connected: includes diagonals
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr != 0 or dc != 0:
                            queue.append((r + dr, c + dc))

        return cells

    def detect_and_analyze(self, grid: List[List[int]]) -> Dict:
        """
        Detect objects and return analysis

        Returns:
            Dictionary with object statistics
        """
        objects = self.detect_objects(grid)

        # Group by color
        by_color = {}
        for obj in objects:
            if obj.color not in by_color:
                by_color[obj.color] = []
            by_color[obj.color].append(obj)

        # Analyze
        analysis = {
            'num_objects': len(objects),
            'num_colors': len(by_color),
            'objects_by_color': {color: len(objs) for color, objs in by_color.items()},
            'total_cells': sum(obj.size for obj in objects),
            'avg_object_size': sum(obj.size for obj in objects) / len(objects) if objects else 0,
            'objects': objects
        }

        return analysis


class ObjectTransformationDetector:
    """
    Detects transformations between objects in input and output grids

    This is used to learn object-based patterns like:
    - Objects moved to new positions
    - Objects copied/duplicated
    - Objects recolored
    - Objects scaled
    - Objects rotated
    """

    def __init__(self, detector: ARCObjectDetector = None):
        self.detector = detector or ARCObjectDetector()

    def detect_transformation(self, input_grid: List[List[int]],
                            output_grid: List[List[int]]) -> Dict:
        """
        Detect transformation from input to output

        Returns:
            Dictionary describing the transformation
        """
        input_objects = self.detector.detect_objects(input_grid)
        output_objects = self.detector.detect_objects(output_grid)

        transformations = []

        # Check for object movement
        movement = self._detect_movement(input_objects, output_objects)
        if movement:
            transformations.append(movement)

        # Check for object copying
        copying = self._detect_copying(input_objects, output_objects)
        if copying:
            transformations.append(copying)

        # Check for object recoloring
        recoloring = self._detect_recoloring(input_objects, output_objects)
        if recoloring:
            transformations.append(recoloring)

        # Check for object scaling
        scaling = self._detect_object_scaling(input_objects, output_objects)
        if scaling:
            transformations.append(scaling)

        return {
            'input_object_count': len(input_objects),
            'output_object_count': len(output_objects),
            'transformations': transformations
        }

    def _detect_movement(self, input_objs: List[ARCObject],
                        output_objs: List[ARCObject]) -> Dict:
        """Detect if objects moved to new positions"""

        # Check if same number of objects
        if len(input_objs) != len(output_objs):
            return None

        # Try to match objects by shape
        movements = []
        matched_output = set()

        for in_obj in input_objs:
            for i, out_obj in enumerate(output_objs):
                if i in matched_output:
                    continue

                # Check if same shape and color
                if in_obj.color == out_obj.color and in_obj.is_same_shape(out_obj):
                    # Calculate movement
                    in_center = in_obj.get_center()
                    out_center = out_obj.get_center()
                    delta_r = float(out_center[0] - in_center[0])
                    delta_c = float(out_center[1] - in_center[1])

                    movements.append({
                        'from': (float(in_center[0]), float(in_center[1])),
                        'to': (float(out_center[0]), float(out_center[1])),
                        'delta': (delta_r, delta_c)
                    })
                    matched_output.add(i)
                    break

        # Check if all objects moved by same delta
        if movements and len(movements) == len(input_objs):
            deltas = [m['delta'] for m in movements]
            if len(set(deltas)) == 1:
                # All objects moved by same amount
                return {
                    'type': 'uniform_movement',
                    'delta': deltas[0],
                    'description': f"All objects moved by {deltas[0]}"
                }

        return None

    def _detect_copying(self, input_objs: List[ARCObject],
                       output_objs: List[ARCObject]) -> Dict:
        """Detect if objects were copied/duplicated"""

        if len(output_objs) <= len(input_objs):
            return None

        # Count output objects that match each input object
        copies = {}
        for in_obj in input_objs:
            count = sum(1 for out_obj in output_objs
                       if in_obj.color == out_obj.color and in_obj.is_same_shape(out_obj))
            if count > 1:
                copies[int(in_obj.color)] = int(count)

        if copies:
            return {
                'type': 'object_copying',
                'copies': copies,
                'description': f"Objects copied: {copies}"
            }

        return None

    def _detect_recoloring(self, input_objs: List[ARCObject],
                          output_objs: List[ARCObject]) -> Dict:
        """Detect if objects were recolored"""

        if len(input_objs) != len(output_objs):
            return None

        # Try to match by shape (ignoring color)
        color_changes = []
        matched_output = set()

        for in_obj in input_objs:
            for i, out_obj in enumerate(output_objs):
                if i in matched_output:
                    continue

                if in_obj.is_same_shape(out_obj) and in_obj.color != out_obj.color:
                    color_changes.append({
                        'from_color': int(in_obj.color),
                        'to_color': int(out_obj.color)
                    })
                    matched_output.add(i)
                    break

        if color_changes:
            # Check if consistent color mapping
            color_map = {}
            for change in color_changes:
                from_c = change['from_color']
                to_c = change['to_color']
                if from_c in color_map and color_map[from_c] != to_c:
                    # Inconsistent mapping
                    return None
                color_map[int(from_c)] = int(to_c)

            return {
                'type': 'object_recoloring',
                'color_map': color_map,
                'description': f"Objects recolored: {color_map}"
            }

        return None

    def _detect_object_scaling(self, input_objs: List[ARCObject],
                              output_objs: List[ARCObject]) -> Dict:
        """Detect if objects were scaled"""

        if len(input_objs) != len(output_objs):
            return None

        # Check if object sizes changed proportionally
        scale_factors = []

        for in_obj, out_obj in zip(input_objs, output_objs):
            if in_obj.color == out_obj.color:
                scale_h = float(out_obj.height / in_obj.height) if in_obj.height > 0 else 0.0
                scale_w = float(out_obj.width / in_obj.width) if in_obj.width > 0 else 0.0
                scale_factors.append((scale_h, scale_w))

        if scale_factors and len(set(scale_factors)) == 1:
            scale_h, scale_w = scale_factors[0]
            if scale_h != 1 or scale_w != 1:
                return {
                    'type': 'object_scaling',
                    'scale_h': scale_h,
                    'scale_w': scale_w,
                    'description': f"Objects scaled {scale_h}x{scale_w}"
                }

        return None


def main():
    """Demo: Object detection on sample grids"""
    print("="*70)
    print("ARC Object Detection Demo")
    print("="*70)

    # Example 1: Simple objects
    grid1 = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2]
    ]

    print("\nExample 1: Two square objects")
    detector = ARCObjectDetector()
    objects = detector.detect_objects(grid1)

    for i, obj in enumerate(objects, 1):
        print(f"  Object {i}: {obj}")
        print(f"    Center: {obj.get_center()}")
        print(f"    Bounding box: {obj.get_bounding_box()}")

    # Example 2: Multiple objects of same color
    grid2 = [
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 0, 3],
        [3, 0, 0, 0, 3]
    ]

    print("\nExample 2: Multiple objects (same color)")
    analysis = detector.detect_and_analyze(grid2)
    print(f"  Total objects: {analysis['num_objects']}")
    print(f"  Objects by color: {analysis['objects_by_color']}")
    print(f"  Average object size: {analysis['avg_object_size']:.1f} cells")

    # Example 3: Object transformation
    input_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    output_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    print("\nExample 3: Object movement detection")
    transform_detector = ObjectTransformationDetector()
    result = transform_detector.detect_transformation(input_grid, output_grid)
    print(f"  Input objects: {result['input_object_count']}")
    print(f"  Output objects: {result['output_object_count']}")
    if result['transformations']:
        for t in result['transformations']:
            print(f"  → {t['description']}")

    print("\n" + "="*70)
    print("Object Detection System Ready!")
    print("="*70)


if __name__ == '__main__':
    main()
