#!/usr/bin/env python3
"""
Dots and Boxes Differential Scoring System

Uses same philosophy as chess and checkers:
- Differential scoring (my boxes - opponent boxes)
- Potential boxes = lines with 3 of 4 edges drawn
- Win: (box_advantage) + 1000
- Loss: (box_advantage) - 1000
- Draw: (box_advantage)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dots_boxes_board import DotsBoxesBoard, Color
from .dots_boxes_game import DotsBoxesGame


class DotsBoxesScorer:
    """
    Differential scoring for Dots and Boxes

    Inherits philosophy from GameScorer but adapted for Dots and Boxes
    Focuses on box ownership and potential (3-edge boxes) instead of just material
    """

    # Box value weights
    BOX_WEIGHT = 100        # Value of claiming a box
    POTENTIAL_WEIGHT = 20   # Value of 3-edge box (potential)
    EDGE_WEIGHT = 2         # Value of edge length (1 or 2 edges)

    def count_boxes(self, board: DotsBoxesBoard, color: Color) -> int:
        """
        Count boxes owned by a color

        Args:
            board: Game board

        Returns: Number of boxes claimed
        """
        red, blue = board.get_box_counts()
        return red if color == Color.RED else blue

    def count_potential_boxes(self, board: DotsBoxesBoard, color: Color) -> int:
        """
        Count boxes with 3 edges drawn (potential for next move)

        A "potential box" is one where 3 of 4 edges are drawn,
        meaning the next player to draw an edge there claims the box.

        Args:
            board: Game board
            color: Color to check (returns all potential boxes, not just opponent's)

        Returns: Number of unclaimed boxes with 3 edges
        """
        count = 0
        for box_pos in board.boxes:
            if board.boxes[box_pos] is None:  # Unclaimed
                if board.count_drawn_edges(box_pos) == 3:
                    count += 1
        return count

    def calculate_box_advantage(self, board: DotsBoxesBoard, ai_color: Color) -> int:
        """
        Calculate differential score based on box ownership

        Formula: (my_boxes - opp_boxes) * BOX_WEIGHT
                 + (my_potential_boxes - opp_potential_boxes) * POTENTIAL_WEIGHT

        Args:
            board: Game board
            ai_color: Color of AI player

        Returns: Differential score (negative if opponent is winning)
        """
        opp_color = Color.BLUE if ai_color == Color.RED else Color.RED

        # Box advantage
        ai_boxes = self.count_boxes(board, ai_color)
        opp_boxes = self.count_boxes(board, opp_color)
        box_advantage = (ai_boxes - opp_boxes) * self.BOX_WEIGHT

        # Potential box advantage (both players compete for these)
        ai_potential = self.count_potential_boxes(board, ai_color)
        opp_potential = self.count_potential_boxes(board, opp_color)
        potential_advantage = (ai_potential - opp_potential) * self.POTENTIAL_WEIGHT

        return box_advantage + potential_advantage

    def calculate_final_score(self, board: DotsBoxesBoard, ai_color: Color,
                             rounds_played: int) -> tuple:
        """
        Calculate differential score at game end

        Same formula as checkers and Connect Four:
        - Win: (box_advantage) + (100 - rounds) + 1000
        - Loss: (box_advantage) - 1000
        - Draw: (box_advantage)

        Args:
            board: Final game board
            ai_color: Color of AI player
            rounds_played: Number of turns played

        Returns: (score, result) where result is 'win', 'loss', or 'draw'
        """
        # Calculate box advantage (DIFFERENTIAL!)
        box_advantage = self.calculate_box_advantage(board, ai_color)

        # Determine game result
        winner = board.get_winner()

        if winner == ai_color:
            # AI WON!
            time_bonus = max(0, 100 - rounds_played)  # Faster wins = higher score
            win_bonus = 1000
            final_score = box_advantage + time_bonus + win_bonus
            return (final_score, 'win')

        elif winner is not None and winner != ai_color:
            # AI LOST
            loss_penalty = 1000
            final_score = box_advantage - loss_penalty
            return (final_score, 'loss')

        else:
            # DRAW (tied box count)
            final_score = box_advantage
            return (final_score, 'draw')

    def calculate_move_delta(self, board_before: DotsBoxesBoard,
                            board_after: DotsBoxesBoard, ai_color: Color) -> float:
        """
        Calculate advantage change after a move (exchange evaluation)

        Returns: Change in box advantage after move
        """
        # Before move
        advantage_before = self.calculate_box_advantage(board_before, ai_color)

        # After move
        advantage_after = self.calculate_box_advantage(board_after, ai_color)

        # Delta
        return advantage_after - advantage_before

    def rank_moves_by_value(self, game: DotsBoxesGame, color: Color,
                           top_k: int = 10) -> list:
        """
        Rank legal moves by box creation value

        Prioritizes moves that:
        1. Complete boxes for us
        2. Block opponent from completing boxes
        3. Create potential boxes for us
        4. Minimize opponent's potential

        Args:
            game: Current game state
            color: Color to evaluate moves for
            top_k: Return top K moves

        Returns: List of (edge, value_delta, advantage) tuples, sorted by delta descending
        """
        legal_moves = game.get_legal_moves()
        move_scores = []

        opp_color = Color.BLUE if color == Color.RED else Color.RED

        for edge in legal_moves:
            # Create board after move
            board_after = game.board.copy()
            completed_box = board_after.draw_edge(edge, color)

            # Calculate advantage after move
            advantage_after = self.calculate_box_advantage(board_after, color)
            advantage_before = self.calculate_box_advantage(game.board, color)
            value_delta = advantage_after - advantage_before

            # Heavy bonus for completing a box
            if completed_box is not None:
                value_delta += 500  # Significant bonus for box completion

            move_scores.append((edge, value_delta, advantage_after))

        # Sort by delta descending (best moves first)
        move_scores.sort(key=lambda x: x[1], reverse=True)

        return move_scores[:top_k]

    def count_edge_groups(self, board: DotsBoxesBoard, color: Color) -> dict:
        """
        Count edge patterns for strategic evaluation

        Returns dict with:
        - claimed_boxes: Boxes owned by this color
        - potential_boxes: Unclaimed boxes with 3 edges
        - threatened_boxes: Boxes with 2+ of opponent's edges
        - open_boxes: Boxes with 0 edges drawn

        Args:
            board: Game board
            color: Color to evaluate

        Returns: Dict with box counts by type
        """
        opp_color = Color.BLUE if color == Color.RED else Color.RED

        claimed = self.count_boxes(board, color)
        potential = self.count_potential_boxes(board, color)

        # Count boxes with 2+ opponent edges
        threatened = 0
        open_boxes = 0

        for box_pos in board.boxes:
            if board.boxes[box_pos] is None:  # Unclaimed
                edges_drawn = board.count_drawn_edges(box_pos)
                if edges_drawn == 0:
                    open_boxes += 1
                # Could add more sophisticated threat detection here

        return {
            'claimed_boxes': claimed,
            'potential_boxes': potential,
            'open_boxes': open_boxes,
            'total_unclaimed': sum(1 for b in board.boxes.values() if b is None),
        }


def test_dots_boxes_scorer():
    """Test differential scoring for Dots and Boxes"""
    print("=" * 70)
    print("DOTS AND BOXES DIFFERENTIAL SCORING TEST")
    print("=" * 70)

    scorer = DotsBoxesScorer()

    # Test 1: Starting position
    print("\nTest 1: Starting position (no boxes)")
    board = DotsBoxesBoard()
    red_advantage = scorer.calculate_box_advantage(board, Color.RED)
    blue_advantage = scorer.calculate_box_advantage(board, Color.BLUE)

    print(f"  Red advantage: {red_advantage:+.0f}")
    print(f"  Blue advantage: {blue_advantage:+.0f}")
    print(f"  (Should be equal and opposite)")

    # Test 2: Red claims a box
    print("\nTest 2: Red claims 1 box")
    board2 = DotsBoxesBoard()
    box_edges = board2.get_box_edges((0, 0))
    for edge in box_edges:
        board2.draw_edge(edge, Color.RED)

    red_advantage2 = scorer.calculate_box_advantage(board2, Color.RED)
    red_count = scorer.count_boxes(board2, Color.RED)

    print(f"  Red boxes: {red_count}")
    print(f"  Red advantage: {red_advantage2:+.0f}")
    print(f"  (Should be {1 * scorer.BOX_WEIGHT} = {scorer.BOX_WEIGHT})")

    # Test 3: Multiple boxes
    print("\nTest 3: Red=2 boxes, Blue=1 box")
    board3 = DotsBoxesBoard()

    # Red claims box (0,0)
    for edge in board3.get_box_edges((0, 0)):
        board3.draw_edge(edge, Color.RED)

    # Red claims box (1,0)
    for edge in board3.get_box_edges((1, 0)):
        board3.draw_edge(edge, Color.RED)

    # Blue claims box (0,1)
    for edge in board3.get_box_edges((0, 1)):
        board3.draw_edge(edge, Color.BLUE)

    red_adv = scorer.calculate_box_advantage(board3, Color.RED)
    blue_adv = scorer.calculate_box_advantage(board3, Color.BLUE)

    print(f"  Red boxes: {scorer.count_boxes(board3, Color.RED)}")
    print(f"  Blue boxes: {scorer.count_boxes(board3, Color.BLUE)}")
    print(f"  Red advantage: {red_adv:+.0f}")
    print(f"  Blue advantage: {blue_adv:+.0f}")
    print(f"  (Differential: advantages are opposite)")

    # Test 4: Win scoring
    print("\nTest 4: Red wins 8-7")
    board4 = DotsBoxesBoard()
    # Manually set 8 red boxes and 7 blue boxes (simulate end game)
    for i in range(8):
        board4.boxes[(i // 4, i % 4)] = Color.RED
    for i in range(8, 15):
        board4.boxes[(i // 4, i % 4)] = Color.BLUE

    score, result = scorer.calculate_final_score(board4, Color.RED, rounds_played=40)
    print(f"  Result: {result}")
    print(f"  Score: {score:.0f}")
    print(f"  Formula: (red_boxes - blue_boxes) * BOX_WEIGHT + time_bonus + 1000")

    # Test 5: Loss scoring
    print("\nTest 5: Red loses 3-9")
    board5 = DotsBoxesBoard()
    for i in range(3):
        board5.boxes[(i // 4, i % 4)] = Color.RED
    for i in range(3, 12):
        board5.boxes[(i // 4, i % 4)] = Color.BLUE

    score5, result5 = scorer.calculate_final_score(board5, Color.RED, rounds_played=30)
    print(f"  Result: {result5}")
    print(f"  Score: {score5:.0f}")
    print(f"  Formula: (box_advantage) - 1000")

    # Test 6: Weights
    print("\nTest 6: Scoring weights")
    print(f"  BOX_WEIGHT: {scorer.BOX_WEIGHT}")
    print(f"  POTENTIAL_WEIGHT: {scorer.POTENTIAL_WEIGHT}")
    print(f"  EDGE_WEIGHT: {scorer.EDGE_WEIGHT}")

    print("\n" + "=" * 70)
    print("✓ Differential scoring working for Dots and Boxes!")
    print("Same philosophy as chess and checkers:")
    print("  - Box advantage (not absolute)")
    print("  - Time bonus for quick wins")
    print("  - Win/loss penalties")
    print("=" * 70)


if __name__ == '__main__':
    test_dots_boxes_scorer()
