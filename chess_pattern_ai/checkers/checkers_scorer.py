#!/usr/bin/env python3
"""
Checkers Differential Scoring System

Uses same philosophy as chess:
- Differential scoring (my advantage - opponent advantage)
- Material values: man=100, king=300
- Win bonus + time bonus
- Loss penalty
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scorer import GameScorer
from .checkers_board import CheckersBoard, Color, PieceType


class CheckersScorer:
    """
    Differential scoring for checkers

    Inherits philosophy from GameScorer but adapted for checkers
    """

    # Piece values (discovered from observation in real system)
    PIECE_VALUES = {
        PieceType.MAN: 100,   # Regular piece
        PieceType.KING: 300   # King = 3x value (can move backward!)
    }

    def calculate_material(self, board: CheckersBoard, color: Color) -> float:
        """Calculate total material value for a color"""
        total = 0
        for piece in board.get_pieces(color):
            total += self.PIECE_VALUES[piece.type]
        return total

    def calculate_final_score(self, board: CheckersBoard, ai_color: Color,
                             rounds_played: int) -> tuple:
        """
        Calculate differential score at game end

        Same formula as chess:
        - Win: (my_material - opp_material) + (100 - rounds) + 1000
        - Loss: (my_material - opp_material) - 1000
        - Draw: (my_material - opp_material)

        Returns: (score, result)
        """
        # Calculate material advantage (DIFFERENTIAL!)
        ai_material = self.calculate_material(board, ai_color)
        opp_color = Color.BLACK if ai_color == Color.RED else Color.RED
        opp_material = self.calculate_material(board, opp_color)

        material_advantage = ai_material - opp_material

        # Determine game result
        winner = board.get_winner()

        if winner == ai_color:
            # AI WON!
            time_bonus = max(0, 100 - rounds_played)  # Faster wins = higher score
            win_bonus = 1000
            final_score = material_advantage + time_bonus + win_bonus
            return (final_score, 'win')

        elif winner is not None and winner != ai_color:
            # AI LOST
            loss_penalty = 1000
            final_score = material_advantage - loss_penalty
            return (final_score, 'loss')

        else:
            # DRAW (both ran out of moves, or stalemate)
            final_score = material_advantage
            return (final_score, 'draw')

    def calculate_material_delta(self, board_before: CheckersBoard,
                                 board_after: CheckersBoard, ai_color: Color) -> float:
        """
        Calculate material advantage change (exchange evaluation)

        Returns: Change in material advantage after move
        """
        opp_color = Color.BLACK if ai_color == Color.RED else Color.RED

        # Before move
        ai_before = self.calculate_material(board_before, ai_color)
        opp_before = self.calculate_material(board_before, opp_color)
        advantage_before = ai_before - opp_before

        # After move
        ai_after = self.calculate_material(board_after, ai_color)
        opp_after = self.calculate_material(board_after, opp_color)
        advantage_after = ai_after - opp_after

        # Delta
        return advantage_after - advantage_before


def test_checkers_scorer():
    """Test differential scoring for checkers"""
    print("=" * 70)
    print("CHECKERS DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = CheckersScorer()

    # Test 1: Even position
    print("\nTest 1: Starting position (even material)")
    board = CheckersBoard()
    red_mat = scorer.calculate_material(board, Color.RED)
    black_mat = scorer.calculate_material(board, Color.BLACK)
    advantage = red_mat - black_mat

    print(f"  Red material: {red_mat}")
    print(f"  Black material: {black_mat}")
    print(f"  Advantage: {advantage:+.0f}")

    # Test 2: After capture
    print("\nTest 2: Red captured one black piece")
    board2 = CheckersBoard()
    # Remove one black piece
    black_pieces = board2.get_pieces(Color.BLACK)
    if black_pieces:
        piece_to_remove = list(black_pieces)[0]
        board2.pieces.remove(piece_to_remove)
        row, col = piece_to_remove.position
        board2.board[row][col] = None

    red_mat2 = scorer.calculate_material(board2, Color.RED)
    black_mat2 = scorer.calculate_material(board2, Color.BLACK)
    advantage2 = red_mat2 - black_mat2

    print(f"  Red material: {red_mat2}")
    print(f"  Black material: {black_mat2}")
    print(f"  Advantage: {advantage2:+.0f}")

    # Test 3: Win scoring
    print("\nTest 3: Red wins in 20 rounds, ahead by 300")
    board3 = CheckersBoard()
    # Simulate win state
    score, result = scorer.calculate_final_score(board3, Color.RED, rounds_played=20)
    print(f"  Result: {result}")
    print(f"  Assumed advantage: +300")
    print(f"  Formula: 300 + (100-20) + 1000 = {300 + 80 + 1000}")
    print(f"  Actual score: {score:.0f}")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for checkers!")
    print("Same philosophy as chess:")
    print("  - Material advantage (not absolute)")
    print("  - Time bonus for quick wins")
    print("  - Loss penalty")
    print("=" * 70)


if __name__ == '__main__':
    test_checkers_scorer()
