#!/usr/bin/env python3
"""
ARC Puzzle Data Structures

Represents ARC-AGI puzzles and provides utilities for loading and manipulating them.
Follows the same observation-based learning philosophy as the chess/checkers learners.
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path


class ARCPuzzle:
    """
    Represents a single ARC puzzle task

    Structure:
    - train: List of input-output example pairs (typically 2-10, usually 3)
    - test: List of test cases with input (and sometimes output for validation)
    """

    def __init__(self, puzzle_id: str, data: Dict):
        self.puzzle_id = puzzle_id
        self.train_examples = data.get('train', [])
        self.test_examples = data.get('test', [])

    def __repr__(self):
        return (f"ARCPuzzle(id={self.puzzle_id}, "
                f"train_examples={len(self.train_examples)}, "
                f"test_examples={len(self.test_examples)})")

    def get_train_pairs(self) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """
        Get training input-output pairs

        Returns:
            List of (input_grid, output_grid) tuples
        """
        return [(ex['input'], ex['output']) for ex in self.train_examples]

    def get_test_inputs(self) -> List[List[List[int]]]:
        """Get test case inputs"""
        return [ex['input'] for ex in self.test_examples]

    def get_test_outputs(self) -> List[List[List[int]]]:
        """Get test case outputs (if available)"""
        return [ex.get('output') for ex in self.test_examples
                if 'output' in ex]

    def get_grid_dimensions(self, grid: List[List[int]]) -> Tuple[int, int]:
        """Get grid dimensions (rows, cols)"""
        if not grid:
            return (0, 0)
        return (len(grid), len(grid[0]) if grid else 0)

    def analyze_transformation(self) -> Dict:
        """
        Analyze the transformation characteristics of this puzzle

        Returns:
            Dictionary with transformation metadata
        """
        analysis = {
            'puzzle_id': self.puzzle_id,
            'num_train_examples': len(self.train_examples),
            'num_test_examples': len(self.test_examples),
            'input_dimensions': [],
            'output_dimensions': [],
            'dimension_changes': [],
            'color_counts_input': [],
            'color_counts_output': []
        }

        for input_grid, output_grid in self.get_train_pairs():
            in_dims = self.get_grid_dimensions(input_grid)
            out_dims = self.get_grid_dimensions(output_grid)

            analysis['input_dimensions'].append(in_dims)
            analysis['output_dimensions'].append(out_dims)

            # Check if dimensions changed
            if in_dims != out_dims:
                scale_h = out_dims[0] / in_dims[0] if in_dims[0] > 0 else 0
                scale_w = out_dims[1] / in_dims[1] if in_dims[1] > 0 else 0
                analysis['dimension_changes'].append({
                    'from': in_dims,
                    'to': out_dims,
                    'scale_h': scale_h,
                    'scale_w': scale_w
                })

            # Count colors (0-9)
            analysis['color_counts_input'].append(self._count_colors(input_grid))
            analysis['color_counts_output'].append(self._count_colors(output_grid))

        return analysis

    def _count_colors(self, grid: List[List[int]]) -> Dict[int, int]:
        """Count occurrences of each color in grid"""
        counts = {}
        for row in grid:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        return counts

    def visualize_example(self, example_idx: int = 0, use_colors: bool = False):
        """
        Print a training example in human-readable format

        Args:
            example_idx: Which training example to show (0-indexed)
            use_colors: If True, use ANSI colors (requires terminal support)
        """
        if example_idx >= len(self.train_examples):
            print(f"Example {example_idx} not found (only {len(self.train_examples)} examples)")
            return

        example = self.train_examples[example_idx]
        input_grid = example['input']
        output_grid = example['output']

        print(f"\n{'='*60}")
        print(f"Puzzle: {self.puzzle_id} - Example {example_idx + 1}")
        print(f"{'='*60}")

        print(f"\nINPUT ({len(input_grid)}x{len(input_grid[0]) if input_grid else 0}):")
        self._print_grid(input_grid, use_colors)

        print(f"\nOUTPUT ({len(output_grid)}x{len(output_grid[0]) if output_grid else 0}):")
        self._print_grid(output_grid, use_colors)

        print(f"\n{'='*60}\n")

    def _print_grid(self, grid: List[List[int]], use_colors: bool = False):
        """Print a grid in readable format"""
        if use_colors:
            # ANSI color codes for visualization
            colors = {
                0: '\033[40m  \033[0m',  # Black
                1: '\033[44m  \033[0m',  # Blue
                2: '\033[41m  \033[0m',  # Red
                3: '\033[42m  \033[0m',  # Green
                4: '\033[43m  \033[0m',  # Yellow
                5: '\033[45m  \033[0m',  # Magenta
                6: '\033[46m  \033[0m',  # Cyan
                7: '\033[47m  \033[0m',  # White
                8: '\033[100m  \033[0m', # Gray
                9: '\033[101m  \033[0m', # Light Red
            }
            for row in grid:
                print(''.join(colors.get(cell, f'{cell} ') for cell in row))
        else:
            for row in grid:
                print(' '.join(str(cell) for cell in row))


class ARCDatasetLoader:
    """
    Loads and manages the ARC-AGI dataset

    Usage:
        loader = ARCDatasetLoader('arc_dataset/data')
        puzzles = loader.load_training_set()
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / 'training'
        self.evaluation_dir = self.data_dir / 'evaluation'

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

    def load_puzzle(self, puzzle_id: str, dataset: str = 'training') -> ARCPuzzle:
        """
        Load a single puzzle by ID

        Args:
            puzzle_id: Puzzle ID (e.g., '007bbfb7')
            dataset: 'training' or 'evaluation'

        Returns:
            ARCPuzzle instance
        """
        dataset_dir = self.training_dir if dataset == 'training' else self.evaluation_dir
        puzzle_path = dataset_dir / f"{puzzle_id}.json"

        if not puzzle_path.exists():
            raise ValueError(f"Puzzle not found: {puzzle_path}")

        with open(puzzle_path, 'r') as f:
            data = json.load(f)

        return ARCPuzzle(puzzle_id, data)

    def load_training_set(self) -> List[ARCPuzzle]:
        """Load all training puzzles"""
        return self._load_dataset(self.training_dir)

    def load_evaluation_set(self) -> List[ARCPuzzle]:
        """Load all evaluation puzzles"""
        return self._load_dataset(self.evaluation_dir)

    def _load_dataset(self, dataset_dir: Path) -> List[ARCPuzzle]:
        """Load all puzzles from a directory"""
        puzzles = []

        for puzzle_file in sorted(dataset_dir.glob('*.json')):
            puzzle_id = puzzle_file.stem

            try:
                with open(puzzle_file, 'r') as f:
                    data = json.load(f)
                puzzles.append(ARCPuzzle(puzzle_id, data))
            except Exception as e:
                print(f"Error loading {puzzle_id}: {e}")

        return puzzles

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        train_puzzles = self.load_training_set()
        eval_puzzles = self.load_evaluation_set()

        stats = {
            'num_training_puzzles': len(train_puzzles),
            'num_evaluation_puzzles': len(eval_puzzles),
            'total_puzzles': len(train_puzzles) + len(eval_puzzles),
            'training_examples': sum(len(p.train_examples) for p in train_puzzles),
            'evaluation_examples': sum(len(p.train_examples) for p in eval_puzzles),
            'total_examples': 0
        }

        stats['total_examples'] = (stats['training_examples'] +
                                  stats['evaluation_examples'])

        return stats


def main():
    """Demo: Load and display puzzle statistics"""
    print("ARC-AGI Dataset Loader Demo")
    print("=" * 60)

    # Load dataset
    loader = ARCDatasetLoader('../arc_dataset/data')

    # Get statistics
    stats = loader.get_dataset_stats()
    print("\nDataset Statistics:")
    print(f"  Training puzzles: {stats['num_training_puzzles']}")
    print(f"  Evaluation puzzles: {stats['num_evaluation_puzzles']}")
    print(f"  Total puzzles: {stats['total_puzzles']}")
    print(f"  Training examples: {stats['training_examples']}")
    print(f"  Evaluation examples: {stats['evaluation_examples']}")
    print(f"  Total examples: {stats['total_examples']}")

    # Load and visualize a specific puzzle
    print("\n" + "=" * 60)
    print("Loading example puzzle: 007bbfb7")
    print("=" * 60)

    puzzle = loader.load_puzzle('007bbfb7', 'training')
    print(f"\n{puzzle}")

    # Analyze transformation
    analysis = puzzle.analyze_transformation()
    print(f"\nTransformation Analysis:")
    print(f"  Training examples: {analysis['num_train_examples']}")
    print(f"  Test examples: {analysis['num_test_examples']}")

    if analysis['dimension_changes']:
        print(f"\n  Dimension changes detected:")
        for i, change in enumerate(analysis['dimension_changes']):
            print(f"    Example {i+1}: {change['from']} → {change['to']} "
                  f"(scale: {change['scale_h']:.1f}x{change['scale_w']:.1f})")

    # Visualize first example
    puzzle.visualize_example(0, use_colors=False)

    # Load first 5 training puzzles
    print("\n" + "=" * 60)
    print("Loading first 5 training puzzles:")
    print("=" * 60)

    puzzles = loader.load_training_set()[:5]
    for puzzle in puzzles:
        analysis = puzzle.analyze_transformation()
        print(f"\n{puzzle.puzzle_id}:")
        print(f"  Train examples: {len(puzzle.train_examples)}")
        print(f"  Test examples: {len(puzzle.test_examples)}")

        # Show dimension patterns
        input_dims = analysis['input_dimensions']
        output_dims = analysis['output_dimensions']
        if input_dims and output_dims:
            print(f"  Input dims: {input_dims[0]}")
            print(f"  Output dims: {output_dims[0]}")


if __name__ == '__main__':
    main()
