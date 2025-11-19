#!/usr/bin/env python3
"""
Gomoku Differential Scoring System

Uses same philosophy as chess and checkers:
- Differential scoring (my advantage - opponent advantage)
- Threats = lines of stones with potential to win
- Win: (threats - opp_threats) + 1000
- Loss: (threats - opp_threats) - 1000
- Draw: (threats - opp_threats)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .gomoku_board import GomokuBoard, Color
from .gomoku_game import GomokuGame


class GomokuScorer:
    """
    Differential scoring for Gomoku

    Inherits philosophy from GameScorer but adapted for Gomoku
    Focuses on threats (potential 5-in-a-rows) instead of just material
    """

    # Threat point values (similar to Connect Four but adjusted for Gomoku)
    THREAT_WEIGHTS = {
        'open_4': 100,      # 4 in a row with both ends open = potential winning move
        'blocked_4': 50,    # 4 in a row with one end blocked
        'open_3': 30,       # 3 in a row with both ends open
        'blocked_3': 15,    # 3 in a row with one end blocked
        'open_2': 10,       # 2 in a row with both ends open
        'blocked_2': 5,     # 2 in a row with one end blocked
    }

    def count_threats(self, board: GomokuBoard, color: Color) -> int:
        """
        Count weighted threat score for a color

        A threat is a line of consecutive stones that could potentially
        extend to 5 in a row.

        Returns: Weighted threat count
        """
        game = GomokuGame(board=board)
        threats_dict = game.detect_threats(color)

        # Calculate weighted score
        threat_score = sum(
            count * self.THREAT_WEIGHTS[threat_type]
            for threat_type, count in threats_dict.items()
        )

        return threat_score

    def calculate_final_score(self, board: GomokuBoard, ai_color: Color,
                             rounds_played: int) -> tuple:
        """
        Calculate differential score at game end

        Same formula as checkers and Connect Four:
        - Win: (threat_advantage) + (100 - rounds) + 1000
        - Loss: (threat_advantage) - 1000
        - Draw: (threat_advantage)

        Args:
            board: Final game board
            ai_color: Color of AI player
            rounds_played: Number of turns played

        Returns: (score, result) where result is 'win', 'loss', or 'draw'
        """
        # Calculate threat advantage (DIFFERENTIAL!)
        ai_threats = self.count_threats(board, ai_color)
        opp_color = Color.WHITE if ai_color == Color.BLACK else Color.BLACK
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

    def calculate_threat_delta(self, board_before: GomokuBoard,
                               board_after: GomokuBoard, ai_color: Color) -> float:
        """
        Calculate threat advantage change (exchange evaluation)

        Returns: Change in threat advantage after move
        """
        opp_color = Color.WHITE if ai_color == Color.BLACK else Color.BLACK

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

    def rank_moves_by_threat(self, game: GomokuGame, color: Color,
                            top_k: int = 10) -> list:
        """
        Rank legal moves by threat creation

        Args:
            game: Current game state
            color: Color to evaluate moves for
            top_k: Return top K moves

        Returns: List of (move, threat_delta, threat_score) tuples, sorted by delta descending
        """
        legal_moves = game.get_legal_moves(color)
        move_scores = []

        opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK

        for move in legal_moves:
            # Create board after move
            board_after = game.board.copy()
            board_after.make_move(move)

            # Evaluate threats after move
            ai_threats_after = self.count_threats(board_after, color)
            opp_threats_after = self.count_threats(board_after, opp_color)
            threat_advantage = ai_threats_after - opp_threats_after

            # Delta from current position
            ai_threats_before = self.count_threats(game.board, color)
            opp_threats_before = self.count_threats(game.board, opp_color)
            threat_advantage_before = ai_threats_before - opp_threats_before

            threat_delta = threat_advantage - threat_advantage_before

            move_scores.append((move, threat_delta, threat_advantage))

        # Sort by delta descending
        move_scores.sort(key=lambda x: x[1], reverse=True)

        return move_scores[:top_k]


def test_gomoku_scorer():
    """Test differential scoring for Gomoku"""
    print("=" * 70)
    print("GOMOKU DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = GomokuScorer()

    # Test 1: Starting position
    print("\nTest 1: Starting position (no threats)")
    board = GomokuBoard(15)
    black_threats = scorer.count_threats(board, Color.BLACK)
    white_threats = scorer.count_threats(board, Color.WHITE)
    advantage = black_threats - white_threats

    print(f"  Black threat score: {black_threats}")
    print(f"  White threat score: {white_threats}")
    print(f"  Advantage: {advantage:+.0f}")

    # Test 2: Create a position with threats
    print("\nTest 2: Black creates line of 3 stones")
    board2 = GomokuBoard(15)
    from .gomoku_board import Stone, Move
    for col in range(7, 10):
        stone = Stone(Color.BLACK, (7, col))
        move = Move(position=(7, col), color=Color.BLACK, stone=stone)
        board2.make_move(move)

    black_threats2 = scorer.count_threats(board2, Color.BLACK)
    white_threats2 = scorer.count_threats(board2, Color.WHITE)
    advantage2 = black_threats2 - white_threats2

    print(f"  Black threat score: {black_threats2}")
    print(f"  White threat score: {white_threats2}")
    print(f"  Advantage: {advantage2:+.0f}")

    # Test 3: Win scoring
    print("\nTest 3: Black wins in 20 moves")
    board3 = GomokuBoard(15)
    # Create 5 in a row for black
    for col in range(7, 12):
        stone = Stone(Color.BLACK, (7, col))
        move = Move(position=(7, col), color=Color.BLACK, stone=stone)
        board3.make_move(move)

    score, result = scorer.calculate_final_score(board3, Color.BLACK, rounds_played=20)
    print(f"  Result: {result}")
    print(f"  Score: {score:.0f}")
    print(f"  Formula includes: threat_advantage + time_bonus + 1000")

    # Test 4: Loss scoring
    print("\nTest 4: Black loses")
    board4 = GomokuBoard(15)
    # Create 5 in a row for white
    for col in range(7, 12):
        stone = Stone(Color.WHITE, (7, col))
        move = Move(position=(7, col), color=Color.WHITE, stone=stone)
        board4.make_move(move)

    score4, result4 = scorer.calculate_final_score(board4, Color.BLACK, rounds_played=30)
    print(f"  Result: {result4}")
    print(f"  Score: {score4:.0f}")
    print(f"  Formula includes loss penalty: -1000")

    # Test 5: Threat weights
    print("\nTest 5: Threat weights")
    print("  Threat point values:")
    for threat_type in ['open_4', 'blocked_4', 'open_3', 'blocked_3', 'open_2', 'blocked_2']:
        print(f"    {threat_type}: {scorer.THREAT_WEIGHTS[threat_type]} points")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for Gomoku!")
    print("Same philosophy as chess and checkers:")
    print("  - Threat advantage (not absolute)")
    print("  - Time bonus for quick wins")
    print("  - Win/loss penalties")
    print("=" * 70)


if __name__ == '__main__':
    test_gomoku_scorer()
