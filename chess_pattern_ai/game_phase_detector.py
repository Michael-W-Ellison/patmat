#!/usr/bin/env python3
"""
Game Phase Detector - Detects Opening/Middlegame/Endgame
Detects game phase using piece count and move number.
Uses simple heuristics based on material on board.
"""

import sqlite3
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GamePhaseDetector:
    """Detects what phase of the game we're in"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.phase_thresholds = {}

        self._init_connection()

    def _init_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _load_phase_thresholds(self):
        """
        Load discovered phase transition thresholds
        These thresholds learned from analyzing game transitions
        """
        try:
            self.cursor.execute('''
                SELECT phase_name, threshold_value, confidence
                FROM discovered_phase_weights
                ORDER BY phase_name
            ''')

            for phase_name, threshold, confidence in self.cursor.fetchall():
                self.phase_thresholds[phase_name] = {
                    'threshold': threshold,
                    'confidence': confidence
                }

            if self.phase_thresholds:
                logger.info(f"✓ Loaded {len(self.phase_thresholds)} phase thresholds")

        except sqlite3.Error:
            # Use simple fallback thresholds
            pass

    def detect_game_phase(self, fen: str, move_number: int) -> str:
        """
        Detect current game phase

        Args:
            fen: FEN string of position
            move_number: Current move number

        Returns:
            'opening', 'middlegame', or 'endgame'
        """
        # Parse FEN to count material
        board_part = fen.split()[0]

        # Count total pieces (excluding kings)
        total_pieces = 0
        for char in board_part:
            if char.isalpha() and char.upper() != 'K':
                total_pieces += 1

        # Count queens
        queens = board_part.count('Q') + board_part.count('q')

        # Detect phase using heuristics
        # Opening: early moves OR lots of pieces
        if move_number <= 15 and total_pieces >= 20:
            return 'opening'

        # Endgame: few pieces left OR no queens
        elif total_pieces <= 10 or (queens == 0 and total_pieces <= 14):
            return 'endgame'

        # Middlegame: everything else
        else:
            return 'middlegame'

    def get_phase_weights(self, phase: str) -> Dict[str, float]:
        """
        Get evaluation weights for a specific phase

        Args:
            phase: 'opening', 'middlegame', or 'endgame'

        Returns:
            Dictionary of evaluation component weights
        """
        # Default weights by phase
        # These represent typical importance of different factors
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


def test_game_phase_detector():
    """Test the game phase detector"""
    print("=" * 70)
    print("GAME PHASE DETECTOR TEST")
    print("=" * 70)

    detector = GamePhaseDetector()

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, "Starting position"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", 2, "Early opening"),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6", 6, "Mid opening"),
        ("r1bq1rk1/pp1pppbp/2n2np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 10", 10, "Late opening"),
        ("r1bqr1k1/pp3pbp/2np1np1/4p3/2B1P3/2NP1N2/PPPQ1PPP/R1B1R1K1 w - - 0 15", 15, "Middlegame"),
        ("6k1/5ppp/8/3p4/3P4/8/5PPP/6K1 w - - 0 40", 40, "Endgame - few pieces"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 30", 30, "Endgame - no queens"),
    ]

    print("\nPhase Detection:")
    print("=" * 70)

    for fen, move_num, description in test_positions:
        phase = detector.detect_game_phase(fen, move_num)
        weights = detector.get_phase_weights(phase)

        print(f"\n{description} (move {move_num})")
        print(f"  Phase: {phase.upper()}")
        print(f"  Weights: material={weights['material']:.1f}, "
              f"mobility={weights['mobility']:.1f}, "
              f"safety={weights['safety']:.1f}")

    detector.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_game_phase_detector()
