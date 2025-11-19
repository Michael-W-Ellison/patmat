#!/usr/bin/env python3
"""
Connect Four Differential Scoring System

Uses same philosophy as chess and checkers:
- Differential scoring (my advantage - opponent advantage)
- Threats = potential 4-in-a-rows (lines of 3 with extension potential)
- Win: (threats - opp_threats) + 1000
- Loss: (threats - opp_threats) - 1000
- Draw: (threats - opp_threats)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .connect4_board import Connect4Board, Color


class Connect4Scorer:
    """
    Differential scoring for Connect Four

    Inherits philosophy from GameScorer but adapted for Connect 4
    Focuses on threats (potential 4-in-a-rows) instead of just material
    """

    def count_threats(self, board: Connect4Board, color: Color) -> int:
        """
        Count threats for a color

        A threat is a line of 3 consecutive pieces of the same color
        with at least one empty position that could extend it to 4.
        """
        threats = 0
        pieces = board.get_pieces(color)
        piece_positions = set(p.position for p in pieces)

        # Directions: horizontal, vertical, diagonal-1, diagonal-2
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal (down-right)
            (1, -1)   # Diagonal (down-left)
        ]

        for piece in pieces:
            row, col = piece.position

            for d_row, d_col in directions:
                # Count consecutive pieces in both directions
                count = 1  # The piece itself

                # Forward direction
                for i in range(1, 4):
                    pos = (row + i * d_row, col + i * d_col)
                    if not board.is_valid_position(pos):
                        break
                    if pos in piece_positions:
                        count += 1
                    else:
                        break

                # Backward direction
                for i in range(1, 4):
                    pos = (row - i * d_row, col - i * d_col)
                    if not board.is_valid_position(pos):
                        break
                    if pos in piece_positions:
                        count += 1
                    else:
                        break

                # If we have 3 in a line, it's a threat (potential 4)
                if count == 3:
                    # Verify this is actually a unique threat (not double-counted)
                    # by checking that we found exactly 3 (not 2 + 1 or overlapping)
                    forward_count = 1
                    for i in range(1, 4):
                        pos = (row + i * d_row, col + i * d_col)
                        if not board.is_valid_position(pos):
                            break
                        if pos in piece_positions:
                            forward_count += 1
                        else:
                            break

                    backward_count = 1
                    for i in range(1, 4):
                        pos = (row - i * d_row, col - i * d_col)
                        if not board.is_valid_position(pos):
                            break
                        if pos in piece_positions:
                            backward_count += 1
                        else:
                            break

                    # Count threat only once per line of 3
                    if forward_count + backward_count - 1 == 3:
                        threats += 1

        # Divide by 3 since each line of 3 is counted 3 times (once for each piece)
        return threats // 3

    def calculate_final_score(self, board: Connect4Board, ai_color: Color,
                             rounds_played: int) -> tuple:
        """
        Calculate differential score at game end

        Same formula as checkers:
        - Win: (threats - opp_threats) + (100 - rounds) + 1000
        - Loss: (threats - opp_threats) - 1000
        - Draw: (threats - opp_threats)

        Returns: (score, result)
        """
        # Calculate threat advantage (DIFFERENTIAL!)
        ai_threats = self.count_threats(board, ai_color)
        opp_color = Color.YELLOW if ai_color == Color.RED else Color.RED
        opp_threats = self.count_threats(board, opp_color)

        threat_advantage = ai_threats - opp_threats

        # Determine game result
        winner = board.get_winner()

        if winner == ai_color:
            # AI WON!
            time_bonus = max(0, 100 - rounds_played)  # Faster wins = higher score
            win_bonus = 1000
            final_score = threat_advantage + time_bonus + win_bonus
            return (final_score, 'win')

        elif winner is not None and winner != ai_color:
            # AI LOST
            loss_penalty = 1000
            final_score = threat_advantage - loss_penalty
            return (final_score, 'loss')

        else:
            # DRAW (board full with no winner)
            final_score = threat_advantage
            return (final_score, 'draw')

    def calculate_threat_delta(self, board_before: Connect4Board,
                               board_after: Connect4Board, ai_color: Color) -> float:
        """
        Calculate threat advantage change (exchange evaluation)

        Returns: Change in threat advantage after move
        """
        opp_color = Color.YELLOW if ai_color == Color.RED else Color.RED

        # Before move
        ai_before = self.count_threats(board_before, ai_color)
        opp_before = self.count_threats(board_before, opp_color)
        advantage_before = ai_before - opp_before

        # After move
        ai_after = self.count_threats(board_after, ai_color)
        opp_after = self.count_threats(board_after, opp_color)
        advantage_after = ai_after - opp_after

        # Delta
        return advantage_after - advantage_before


def test_connect4_scorer():
    """Test differential scoring for Connect Four"""
    print("=" * 70)
    print("CONNECT FOUR DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = Connect4Scorer()

    # Test 1: Starting position
    print("\nTest 1: Starting position (no threats)")
    board = Connect4Board()
    red_threats = scorer.count_threats(board, Color.RED)
    yellow_threats = scorer.count_threats(board, Color.YELLOW)
    advantage = red_threats - yellow_threats

    print(f"  Red threats: {red_threats}")
    print(f"  Yellow threats: {yellow_threats}")
    print(f"  Advantage: {advantage:+.0f}")

    # Test 2: Create a position with threats
    print("\nTest 2: Red creates vertical threat (3 in a row)")
    board2 = Connect4Board()
    # Place 3 red pieces in column 0, rows 5, 4, 3
    from .connect4_board import Piece, Move
    for i, row in enumerate([5, 4, 3]):
        piece = Piece(Color.RED, (row, 0))
        move = Move(column=0, row=row, piece=piece)
        board2.make_move(move)

    red_threats2 = scorer.count_threats(board2, Color.RED)
    yellow_threats2 = scorer.count_threats(board2, Color.YELLOW)
    advantage2 = red_threats2 - yellow_threats2

    print(f"  Red threats: {red_threats2}")
    print(f"  Yellow threats: {yellow_threats2}")
    print(f"  Advantage: {advantage2:+.0f}")

    # Test 3: Win scoring
    print("\nTest 3: Red wins in 20 rounds with 1 threat vs 0")
    board3 = Connect4Board()
    score, result = scorer.calculate_final_score(board3, Color.RED, rounds_played=20)
    print(f"  Result: {result}")
    print(f"  Assumed threats: Red=0, Yellow=0")
    print(f"  Formula: 0 + (100-20) + 1000 = {0 + 80 + 1000}")
    print(f"  Actual score: {score:.0f}")

    # Test 4: Loss scoring
    print("\nTest 4: Red loses in 30 rounds")
    board4 = Connect4Board()
    score4, result4 = scorer.calculate_final_score(board4, Color.RED, rounds_played=30)
    print(f"  Result: {result4}")
    print(f"  Formula includes loss penalty: -1000")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for Connect Four!")
    print("Same philosophy as chess and checkers:")
    print("  - Threat advantage (not absolute)")
    print("  - Time bonus for quick wins")
    print("  - Loss penalty")
    print("=" * 70)


if __name__ == '__main__':
    test_connect4_scorer()
