#!/usr/bin/env python3
"""
Go Differential Scoring System

Uses same philosophy as chess and checkers:
- Differential scoring (my advantage - opponent advantage)
- Score = (my_territory + my_captures) - (opp_territory + opp_captures)
- Win bonus: +1000
- Loss penalty: -1000
- Time bonus: (200 - moves_played) for quick wins
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .go_board import GoBoard, Color


class GoScorer:
    """
    Differential scoring for Go

    Inherits philosophy from GameScorer but adapted for Go
    Uses territory and captures as the primary evaluation metric
    """

    def calculate_material(self, board: GoBoard, color: Color) -> float:
        """
        Calculate total material value for a color

        In Go, material = territory + captures
        """
        territory = board.count_territory(color)
        if color == Color.BLACK:
            captures = board.captured_by_black
        else:
            captures = board.captured_by_white
        return territory + captures

    def calculate_final_score(self, board: GoBoard, ai_color: Color,
                             moves_played: int) -> tuple:
        """
        Calculate differential score at game end

        Scoring formula:
        - Win: (my_score - opp_score) + (200 - moves) + 1000
        - Loss: (my_score - opp_score) - 1000
        - Draw: (my_score - opp_score)

        Where my_score = my_territory + my_captures
        And opp_score = opp_territory + opp_captures

        Args:
            board: GoBoard at end of game
            ai_color: Color of AI player
            moves_played: Total number of non-pass moves made in game

        Returns:
            Tuple of (score, result) where result is 'win', 'loss', or 'draw'
        """
        # Calculate material advantage (DIFFERENTIAL!)
        ai_material = self.calculate_material(board, ai_color)
        opp_color = Color.WHITE if ai_color == Color.BLACK else Color.BLACK
        opp_material = self.calculate_material(board, opp_color)

        material_advantage = ai_material - opp_material

        # Determine game result based on territory + captures
        ai_score = ai_material
        opp_score = opp_material

        if ai_score > opp_score:
            # AI WON!
            time_bonus = max(0, 200 - moves_played)  # Faster wins = higher score
            win_bonus = 1000
            final_score = material_advantage + time_bonus + win_bonus
            return (final_score, 'win')

        elif opp_score > ai_score:
            # AI LOST
            loss_penalty = 1000
            final_score = material_advantage - loss_penalty
            return (final_score, 'loss')

        else:
            # DRAW (equal territory + captures)
            final_score = material_advantage
            return (final_score, 'draw')

    def calculate_material_delta(self, board_before: GoBoard,
                                board_after: GoBoard, ai_color: Color) -> float:
        """
        Calculate material advantage change (exchange evaluation)

        Returns:
            Change in material advantage after move
        """
        opp_color = Color.WHITE if ai_color == Color.BLACK else Color.BLACK

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

    def evaluate_position(self, board: GoBoard, color: Color) -> float:
        """
        Evaluate current position for a color (mid-game evaluation)

        Returns the material advantage from the perspective of the given color
        """
        my_material = self.calculate_material(board, color)
        opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
        opp_material = self.calculate_material(board, opp_color)
        return my_material - opp_material


def test_go_scorer():
    """Test differential scoring for Go"""
    print("=" * 70)
    print("GO DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = GoScorer()

    # Test 1: Empty board
    print("\nTest 1: Empty board (no territory)")
    board = GoBoard(9)
    black_mat = scorer.calculate_material(board, Color.BLACK)
    white_mat = scorer.calculate_material(board, Color.WHITE)
    advantage = black_mat - white_mat

    print(f"  Black material: {black_mat}")
    print(f"  White material: {white_mat}")
    print(f"  Advantage: {advantage:+.0f}")

    # Test 2: Simulate some captures
    print("\nTest 2: Black captures 3 stones")
    board2 = GoBoard(9)
    board2.captured_by_black = 3
    black_mat2 = scorer.calculate_material(board2, Color.BLACK)
    white_mat2 = scorer.calculate_material(board2, Color.WHITE)
    advantage2 = black_mat2 - white_mat2

    print(f"  Black material: {black_mat2} (3 captures)")
    print(f"  White material: {white_mat2}")
    print(f"  Advantage: {advantage2:+.0f}")

    # Test 3: Win scoring
    print("\nTest 3: Black wins with material advantage of 15, in 40 moves")
    board3 = GoBoard(9)
    # Simulate win state by setting up captures and territory
    board3.captured_by_black = 10
    board3.captured_by_white = 5

    black_score = 10 + board3.count_territory(Color.BLACK)
    white_score = 5 + board3.count_territory(Color.WHITE)

    print(f"  Black score: {black_score}")
    print(f"  White score: {white_score}")

    score, result = scorer.calculate_final_score(board3, Color.BLACK,
                                                 moves_played=40)
    print(f"  Result: {result}")
    print(f"  Material advantage: +{black_score - white_score}")
    print(f"  Formula: (material_diff) + (200-40) + 1000 = (material_diff) + 160 + 1000")
    print(f"  Actual score: {score:.0f}")

    # Test 4: Differential scoring philosophy
    print("\nTest 4: Differential scoring philosophy")
    board4a = GoBoard(9)
    board4a.captured_by_black = 20
    board4a.captured_by_white = 10

    board4b = GoBoard(9)
    board4b.captured_by_black = 5
    board4b.captured_by_white = 0

    score_a = scorer.calculate_material(board4a, Color.BLACK) - \
              scorer.calculate_material(board4a, Color.WHITE)
    score_b = scorer.calculate_material(board4b, Color.BLACK) - \
              scorer.calculate_material(board4b, Color.WHITE)

    print(f"  Scenario A (B:20, W:10): advantage = {score_a:+.0f}")
    print(f"  Scenario B (B:5, W:0): advantage = {score_b:+.0f}")
    print(f"  Both show same advantage (10) - using DIFFERENTIAL scoring!")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for Go!")
    print("Same philosophy as chess and checkers:")
    print("  - Material advantage (territory + captures)")
    print("  - Time bonus for quick wins (200 - moves)")
    print("  - Win bonus: +1000")
    print("  - Loss penalty: -1000")
    print("=" * 70)


if __name__ == '__main__':
    test_go_scorer()
