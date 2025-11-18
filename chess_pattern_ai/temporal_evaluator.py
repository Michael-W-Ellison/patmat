#!/usr/bin/env python3
"""
Temporal Evaluator - Adapts Evaluation Weights by Game Phase
Adjusts evaluation component weights based on opening/middlegame/endgame.
"""

import sqlite3
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalEvaluator:
    """Adapts evaluation based on game phase"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def evaluate_with_phase_adaptation(self, scores: Dict[str, float],
                                       fen: str, move_number: int) -> float:
        """
        Combine evaluation scores with phase-based weighting

        Args:
            scores: Dictionary of evaluation scores by component
            fen: FEN string of position
            move_number: Current move number

        Returns:
            Combined weighted score
        """
        # Determine phase
        phase = self._detect_simple_phase(fen, move_number)

        # Get phase-specific weights
        weights = self._get_phase_weights(phase)

        # Combine scores with weights
        total = 0.0
        for component, score in scores.items():
            weight = weights.get(component, 1.0)
            total += score * weight

        return total

    def _detect_simple_phase(self, fen: str, move_number: int) -> str:
        """
        Simple phase detection

        Args:
            fen: FEN string
            move_number: Move number

        Returns:
            'opening', 'middlegame', or 'endgame'
        """
        board_part = fen.split()[0]

        # Count pieces (excluding kings)
        total_pieces = sum(1 for c in board_part if c.isalpha() and c.upper() != 'K')

        # Count queens
        queens = board_part.count('Q') + board_part.count('q')

        # Detect phase
        if move_number <= 15 and total_pieces >= 20:
            return 'opening'
        elif total_pieces <= 10 or (queens == 0 and total_pieces <= 14):
            return 'endgame'
        else:
            return 'middlegame'

    def _get_phase_weights(self, phase: str) -> Dict[str, float]:
        """
        Get evaluation weights for phase

        Args:
            phase: 'opening', 'middlegame', or 'endgame'

        Returns:
            Dictionary of weights
        """
        if phase == 'opening':
            return {
                'material': 0.8,
                'mobility': 1.2,
                'safety': 1.5,
                'pawn_structure': 0.5,
                'positional': 1.0,
                'tactical': 1.0
            }
        elif phase == 'endgame':
            return {
                'material': 1.5,
                'mobility': 0.8,
                'safety': 0.5,
                'pawn_structure': 1.5,
                'positional': 1.2,
                'tactical': 0.8
            }
        else:  # middlegame
            return {
                'material': 1.0,
                'mobility': 1.0,
                'safety': 1.0,
                'pawn_structure': 1.0,
                'positional': 1.0,
                'tactical': 1.5
            }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_temporal_evaluator():
    """Test the temporal evaluator"""
    print("=" * 70)
    print("TEMPORAL EVALUATOR TEST")
    print("=" * 70)

    evaluator = TemporalEvaluator()

    # Test scores
    test_scores = {
        'material': 100.0,
        'mobility': 50.0,
        'safety': 30.0,
        'pawn_structure': 20.0,
        'positional': 40.0,
        'tactical': 60.0
    }

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, "Opening"),
        ("r1bqr1k1/pp3pbp/2np1np1/4p3/2B1P3/2NP1N2/PPPQ1PPP/R1B1R1K1 w - - 0 15", 15, "Middlegame"),
        ("6k1/5ppp/8/3p4/3P4/8/5PPP/6K1 w - - 0 40", 40, "Endgame"),
    ]

    print("\nPhase-Adapted Evaluation:")
    print("=" * 70)
    print(f"\nBase scores: {test_scores}")

    for fen, move_num, description in test_positions:
        total = evaluator.evaluate_with_phase_adaptation(test_scores, fen, move_num)
        phase = evaluator._detect_simple_phase(fen, move_num)
        weights = evaluator._get_phase_weights(phase)

        print(f"\n{description} (move {move_num}):")
        print(f"  Phase: {phase}")
        print(f"  Weights: {weights}")
        print(f"  Total score: {total:.1f}")

    evaluator.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_temporal_evaluator()
