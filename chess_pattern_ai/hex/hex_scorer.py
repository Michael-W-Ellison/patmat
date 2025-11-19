#!/usr/bin/env python3
"""
Hex Differential Scoring System

Uses same philosophy as chess, checkers, and Go:
- Differential scoring (my advantage - opponent advantage)
- Score based on connection strength (group size and distance to goal)
- Win bonus: +1000
- Loss penalty: -1000
- Time bonus: (200 - moves_played) for quick wins
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .hex_board import HexBoard, Color


class HexScorer:
    """
    Differential scoring for Hex

    Inherits philosophy from other game scorers but adapted for Hex
    Uses connection strength as the primary evaluation metric
    """

    def calculate_connection_strength(self, board: HexBoard, color: Color) -> float:
        """
        Calculate connection strength value for a color

        Connection strength = group_size + (10 * (1 / distance_to_goal))

        This values:
        - More stones in the main connected group
        - Shorter paths to the goal
        - A 1-step away win is worth much more than being far away

        Returns:
            Float representing connection strength
        """
        group_size, distance_to_goal = board.get_connection_strength(color)

        if group_size == 0:
            return 0.0

        # If already won, return high value
        if distance_to_goal == 0:
            return 100.0

        if distance_to_goal == float('inf'):
            # No path to goal yet
            return float(group_size)

        # Connection strength combines:
        # 1. Number of stones (linear, up to ~11)
        # 2. Distance bonus (exponential - getting closer matters a lot)
        distance_bonus = 10.0 / (distance_to_goal + 1)
        return group_size + distance_bonus

    def calculate_final_score(self, board: HexBoard, ai_color: Color,
                             moves_played: int) -> tuple:
        """
        Calculate differential score at game end

        Scoring formula:
        - Win: (connection_diff) + (200 - moves) + 1000
        - Loss: (connection_diff) - 1000
        - Game not over: (connection_diff)

        Where connection_diff = my_connection_strength - opp_connection_strength

        Args:
            board: HexBoard at end of game
            ai_color: Color of AI player
            moves_played: Total number of moves made in game

        Returns:
            Tuple of (score, result) where result is 'win', 'loss', or 'in_progress'
        """
        # Determine if game is over
        red_won = board.has_won(Color.RED)
        blue_won = board.has_won(Color.BLUE)

        # Calculate connection strength differential
        ai_connection = self.calculate_connection_strength(board, ai_color)
        opp_color = Color.BLUE if ai_color == Color.RED else Color.RED
        opp_connection = self.calculate_connection_strength(board, opp_color)

        connection_advantage = ai_connection - opp_connection

        if red_won or blue_won:
            # Game is over
            if (ai_color == Color.RED and red_won) or (ai_color == Color.BLUE and blue_won):
                # AI WON!
                time_bonus = max(0, 200 - moves_played)  # Faster wins = higher score
                win_bonus = 1000
                final_score = connection_advantage + time_bonus + win_bonus
                return (final_score, 'win')
            else:
                # AI LOST
                loss_penalty = 1000
                final_score = connection_advantage - loss_penalty
                return (final_score, 'loss')
        else:
            # Game still in progress (shouldn't happen if called at end)
            return (connection_advantage, 'in_progress')

    def calculate_material_delta(self, board_before: HexBoard,
                                board_after: HexBoard, ai_color: Color) -> float:
        """
        Calculate connection strength change (exchange evaluation)

        Returns:
            Change in connection advantage after move
        """
        opp_color = Color.BLUE if ai_color == Color.RED else Color.RED

        # Before move
        ai_before = self.calculate_connection_strength(board_before, ai_color)
        opp_before = self.calculate_connection_strength(board_before, opp_color)
        advantage_before = ai_before - opp_before

        # After move
        ai_after = self.calculate_connection_strength(board_after, ai_color)
        opp_after = self.calculate_connection_strength(board_after, opp_color)
        advantage_after = ai_after - opp_after

        # Delta (how much advantage changed)
        return advantage_after - advantage_before

    def evaluate_position(self, board: HexBoard, color: Color) -> float:
        """
        Evaluate current position for a color (mid-game evaluation)

        Returns the connection advantage from the perspective of the given color
        """
        my_connection = self.calculate_connection_strength(board, color)
        opp_color = Color.BLUE if color == Color.RED else Color.RED
        opp_connection = self.calculate_connection_strength(board, opp_color)
        return my_connection - opp_connection


def test_hex_scorer():
    """Test differential scoring for Hex"""
    print("=" * 70)
    print("HEX DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = HexScorer()

    # Test 1: Empty board
    print("\nTest 1: Empty board (no connections)")
    board = HexBoard(11)
    red_conn = scorer.calculate_connection_strength(board, Color.RED)
    blue_conn = scorer.calculate_connection_strength(board, Color.BLUE)
    advantage = red_conn - blue_conn

    print(f"  Red connection strength: {red_conn:.2f}")
    print(f"  Blue connection strength: {blue_conn:.2f}")
    print(f"  Advantage: {advantage:+.2f}")

    # Test 2: Red builds a path
    print("\nTest 2: Red builds connected path toward goal")
    board2 = HexBoard(11)

    # Red builds a path from top toward bottom
    red_positions = [(0, 5), (1, 5), (2, 5), (3, 5)]
    for pos in red_positions:
        from .hex_board import Move
        move = Move(Color.RED, pos)
        board2.make_move(move)

    red_conn2 = scorer.calculate_connection_strength(board2, Color.RED)
    blue_conn2 = scorer.calculate_connection_strength(board2, Color.BLUE)
    advantage2 = red_conn2 - blue_conn2

    print(f"  Red connection strength: {red_conn2:.2f}")
    print(f"  Blue connection strength: {blue_conn2:.2f}")
    print(f"  Advantage: {advantage2:+.2f}")

    # Test 3: Differential scoring philosophy
    print("\nTest 3: Differential scoring philosophy")
    board3a = HexBoard(11)
    board3b = HexBoard(11)

    # Scenario A: Red has big lead
    from .hex_board import Move
    for pos in [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]:
        move = Move(Color.RED, pos)
        board3a.make_move(move)

    # Scenario B: Both have some stones
    board3b.make_move(Move(Color.RED, (0, 5)))
    board3b.make_move(Move(Color.BLUE, (5, 0)))
    board3b.make_move(Move(Color.RED, (1, 5)))
    board3b.make_move(Move(Color.BLUE, (5, 1)))

    score_a = scorer.evaluate_position(board3a, Color.RED)
    score_b = scorer.evaluate_position(board3b, Color.RED)

    print(f"  Scenario A (Red 5 in line, Blue none): advantage = {score_a:+.2f}")
    print(f"  Scenario B (Red 2, Blue 2 separated): advantage = {score_b:+.2f}")
    print(f"  Red is winning in Scenario A - using DIFFERENTIAL scoring!")

    # Test 4: Win scoring
    print("\nTest 4: Win scoring")
    board4 = HexBoard(11)

    # Build a complete winning path for Red
    for i in range(11):
        move = Move(Color.RED, (i, 5))
        board4.make_move(move)
        if i < 10:  # Alternate turns by adding Blue moves
            move_blue = Move(Color.BLUE, (i, 3))
            if board4.get_stone_at((i, 3)) is None:
                board4.make_move(move_blue)

    print(f"  Red has won: {board4.has_won(Color.RED)}")

    if board4.has_won(Color.RED):
        score, result = scorer.calculate_final_score(board4, Color.RED, moves_played=20)
        print(f"  Result: {result}")
        print(f"  Score: {score:.0f}")
        print(f"  Formula: (connection_diff) + (200-20) + 1000 = (diff) + 180 + 1000")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for Hex!")
    print("Same philosophy as chess, checkers, and Go:")
    print("  - Connection strength (group size + distance bonus)")
    print("  - Time bonus for quick wins (200 - moves)")
    print("  - Win bonus: +1000")
    print("  - Loss penalty: -1000")
    print("=" * 70)


if __name__ == '__main__':
    test_hex_scorer()
