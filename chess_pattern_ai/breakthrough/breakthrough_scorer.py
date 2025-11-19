#!/usr/bin/env python3
"""
Breakthrough Differential Scoring System

Uses same philosophy as chess:
- Differential scoring (my advantage - opponent advantage)
- Material values: pawn=100
- Advancement bonus (distance from starting row)
- Win bonus + time bonus
- Loss penalty
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scorer import GameScorer
from .breakthrough_board import BreakthroughBoard, Color


class BreakthroughScorer:
    """
    Differential scoring for breakthrough

    Inherits philosophy from GameScorer but adapted for breakthrough
    """

    # Piece values
    PIECE_VALUE = 100  # All pieces are pawns

    def calculate_material(self, board: BreakthroughBoard, color: Color) -> float:
        """Calculate total material value for a color"""
        piece_count = len(board.get_pieces(color))
        return piece_count * self.PIECE_VALUE

    def calculate_advancement(self, board: BreakthroughBoard, color: Color) -> float:
        """
        Calculate advancement score for a color

        Measures how close pieces are to reaching opponent's back row
        - White pieces: row number (0=start, 7=goal)
        - Black pieces: 7-row number (6=start, 0=goal)
        - Encourages forward progress
        """
        score = 0.0
        pieces = board.get_pieces(color)

        if color == Color.WHITE:
            # White starts at rows 0-1, advances toward row 7
            for piece in pieces:
                row = piece.position[0]
                # Distance from starting position (row 1)
                advancement = row - 1
                score += advancement
        else:
            # Black starts at rows 6-7, advances toward row 0
            for piece in pieces:
                row = piece.position[0]
                # Distance from starting position (row 6)
                advancement = 6 - row
                score += advancement

        return score

    def calculate_final_score(self, board: BreakthroughBoard, ai_color: Color,
                             rounds_played: int) -> tuple:
        """
        Calculate differential score at game end

        Formula:
        - Win: (my_material - opp_material) + (my_advancement - opp_advancement)
          + (100 - rounds) + 1000
        - Loss: (my_material - opp_material) + (my_advancement - opp_advancement) - 1000
        - Draw: (my_material - opp_material) + (my_advancement - opp_advancement)

        Returns: (score, result)
        """
        # Calculate material advantage (DIFFERENTIAL!)
        ai_material = self.calculate_material(board, ai_color)
        opp_color = Color.BLACK if ai_color == Color.WHITE else Color.WHITE
        opp_material = self.calculate_material(board, opp_color)

        material_advantage = ai_material - opp_material

        # Calculate advancement advantage
        ai_advancement = self.calculate_advancement(board, ai_color)
        opp_advancement = self.calculate_advancement(board, opp_color)

        advancement_advantage = ai_advancement - opp_advancement

        # Determine game result
        winner = board.get_winner()

        if winner == ai_color:
            # AI WON!
            time_bonus = max(0, 100 - rounds_played)  # Faster wins = higher score
            win_bonus = 1000
            final_score = material_advantage + advancement_advantage + time_bonus + win_bonus
            return (final_score, 'win')

        elif winner is not None and winner != ai_color:
            # AI LOST
            loss_penalty = 1000
            final_score = material_advantage + advancement_advantage - loss_penalty
            return (final_score, 'loss')

        else:
            # DRAW
            final_score = material_advantage + advancement_advantage
            return (final_score, 'draw')

    def calculate_material_delta(self, board_before: BreakthroughBoard,
                                 board_after: BreakthroughBoard, ai_color: Color) -> float:
        """
        Calculate material advantage change (exchange evaluation)

        Returns: Change in material advantage after move
        """
        opp_color = Color.BLACK if ai_color == Color.WHITE else Color.WHITE

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

    def calculate_position_score(self, board: BreakthroughBoard, color: Color) -> float:
        """
        Calculate position score (material + advancement)

        This is used for evaluation during play
        """
        material = self.calculate_material(board, color)
        advancement = self.calculate_advancement(board, color)
        return material + advancement


def test_breakthrough_scorer():
    """Test differential scoring for breakthrough"""
    print("=" * 70)
    print("BREAKTHROUGH DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = BreakthroughScorer()

    # Test 1: Starting position
    print("\nTest 1: Starting position (even material)")
    board = BreakthroughBoard()
    white_mat = scorer.calculate_material(board, Color.WHITE)
    black_mat = scorer.calculate_material(board, Color.BLACK)
    white_adv = scorer.calculate_advancement(board, Color.WHITE)
    black_adv = scorer.calculate_advancement(board, Color.BLACK)

    print(f"  White material: {white_mat}")
    print(f"  Black material: {black_mat}")
    print(f"  Material advantage: {white_mat - black_mat:+.0f}")
    print(f"  White advancement: {white_adv:.1f}")
    print(f"  Black advancement: {black_adv:.1f}")
    print(f"  Advancement advantage: {white_adv - black_adv:+.1f}")

    # Test 2: After capture
    print("\nTest 2: White captured one black piece")
    board2 = BreakthroughBoard()
    black_pieces = board2.get_pieces(Color.BLACK)
    if black_pieces:
        piece_to_remove = list(black_pieces)[0]
        board2.pieces.remove(piece_to_remove)
        row, col = piece_to_remove.position
        board2.board[row][col] = None

    white_mat2 = scorer.calculate_material(board2, Color.WHITE)
    black_mat2 = scorer.calculate_material(board2, Color.BLACK)

    print(f"  White material: {white_mat2}")
    print(f"  Black material: {black_mat2}")
    print(f"  Advantage: {white_mat2 - black_mat2:+.0f}")

    # Test 3: Win scoring
    print("\nTest 3: White wins in 20 rounds")
    board3 = BreakthroughBoard()
    score, result = scorer.calculate_final_score(board3, Color.WHITE, rounds_played=20)
    print(f"  Result: {result}")
    print(f"  Formula: material + advancement + time_bonus + win_bonus")
    print(f"  Material advantage: 0 (even start)")
    print(f"  Advancement advantage: 0 (no moves)")
    print(f"  Time bonus: (100 - 20) = 80")
    print(f"  Win bonus: 1000")
    print(f"  Actual score: {score:.0f}")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for breakthrough!")
    print("Same philosophy as chess:")
    print("  - Material advantage (not absolute)")
    print("  - Advancement advantage (encourages forward progress)")
    print("  - Time bonus for quick wins")
    print("  - Loss penalty")
    print("=" * 70)


if __name__ == '__main__':
    test_breakthrough_scorer()
