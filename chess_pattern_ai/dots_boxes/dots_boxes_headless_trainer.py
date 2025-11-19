#!/usr/bin/env python3
"""
Dots and Boxes Headless Trainer - Fast Training Without GUI

Philosophy: Same as chess trainer
- Differential scoring
- Pattern learning
- Observation-based (future)
- No GUI overhead
- Complete box detection crucial (gets another turn!)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from typing import List, Tuple
from collections import defaultdict

from dots_boxes.dots_boxes_board import DotsBoxesBoard, Color, Edge
from dots_boxes.dots_boxes_game import DotsBoxesGame
from dots_boxes.dots_boxes_scorer import DotsBoxesScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class DotsBoxesHeadlessTrainer:
    """
    Fast terminal-based Dots and Boxes trainer

    Uses differential scoring and pattern learning
    Same philosophy as chess trainer
    """

    def __init__(self, db_path='dots_boxes_training.db'):
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = DotsBoxesScorer()
        self.board_size = 5  # Standard 5x5 dots (4x4 boxes)

        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []

        # Learning stats
        self.patterns_by_category = defaultdict(int)

    def categorize_move(self, board: DotsBoxesBoard, edge: Edge, ai_color: Color) -> str:
        """
        Categorize a move for pattern learning

        Dots & Boxes-specific categories:
        - 'complete_box' - Completing a box (crucial! gets another turn)
        - 'setup' - Setting up to complete box on next turn
        - 'safe' - Move that doesn't let opponent complete box
        - 'sacrifice' - Giving opponent boxes to get advantage
        - 'quiet' - Normal positional move
        """
        # Check if this move completes a box
        board_copy = board.copy()

        # Get affected boxes for this edge
        affected_boxes = board_copy._get_affected_boxes(edge)

        # Check if any affected box will be completed by this edge
        completes_box = False
        for box_pos in affected_boxes:
            if board_copy.count_drawn_edges(box_pos) == 3:
                completes_box = True
                break

        if completes_box:
            return 'complete_box'

        # Check if this creates a setup (makes a box have 3 edges)
        is_setup = False
        for box_pos in affected_boxes:
            if board_copy.count_drawn_edges(box_pos) == 2:
                is_setup = True
                break

        if is_setup:
            return 'setup'

        # For simplicity, mark the rest as safe or quiet
        # Safe if it doesn't create vulnerability
        if affected_boxes and all(board_copy.count_drawn_edges(box_pos) <= 1 for box_pos in affected_boxes):
            return 'safe'
        else:
            return 'sacrifice'

        return 'quiet'

    def distance_from_center(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> int:
        """
        Calculate distance from center for an edge
        Dots & Boxes board is 5x5 (center at 2,2)
        """
        center = self.board_size // 2
        avg_row = (dot1[0] + dot2[0]) / 2.0
        avg_col = (dot1[1] + dot2[1]) / 2.0

        return max(abs(int(avg_row) - center), abs(int(avg_col) - center))

    def get_game_phase(self, boxes_completed: int, legal_edges_count: int) -> str:
        """
        Determine game phase based on boxes completed

        Args:
            boxes_completed: Total boxes completed so far
            legal_edges_count: Remaining legal edges

        Returns: 'opening', 'middlegame', or 'endgame'
        """
        total_boxes = (self.board_size - 1) * (self.board_size - 1)  # 4x4 = 16 boxes

        if boxes_completed < total_boxes * 0.2:
            return 'opening'
        elif boxes_completed < total_boxes * 0.8:
            return 'middlegame'
        else:
            return 'endgame'

    def select_move(self, game: DotsBoxesGame, ai_color: Color,
                   use_learning: bool = True) -> Edge:
        """
        Select move using learned patterns

        Args:
            game: Current game state
            ai_color: AI's color
            use_learning: If True, use pattern prioritizer; else random
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return None

        if not use_learning:
            return random.choice(legal_moves)

        # Score each move using learned patterns
        move_scores = []

        red_boxes, blue_boxes = game.board.get_box_counts()
        boxes_completed = red_boxes + blue_boxes
        legal_edges = game.board.get_legal_edges()

        for edge in legal_moves:
            # Extract move features for pattern matching
            piece_type = 'edge'

            # Categorize move
            category = self.categorize_move(game.board, edge, ai_color)

            # Calculate distance from center
            distance = self.distance_from_center(edge.dot1, edge.dot2)

            # Determine game phase
            phase = self.get_game_phase(boxes_completed, len(legal_edges))

            # Get priority from learned patterns (query cache directly)
            key = (piece_type, category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns (Dots & Boxes-specific)
                if category == 'complete_box':
                    priority = 100.0  # Completing a box is best (gets another turn!)
                elif category == 'setup':
                    priority = 70.0   # Setup for completing box
                elif category == 'safe':
                    priority = 60.0   # Safe moves
                elif category == 'sacrifice':
                    priority = 30.0   # Sacrifice moves
                else:  # quiet
                    priority = 25.0

            move_scores.append((edge, priority))

        # Select move based on priorities (with exploration)
        if random.random() < 0.1:  # 10% exploration
            return random.choice(legal_moves)
        else:
            # Choose move with highest priority
            move_scores.sort(key=lambda x: x[1], reverse=True)
            return move_scores[0][0]

    def play_game(self, ai_color: Color, verbose: bool = False) -> Tuple[str, float, int]:
        """
        Play one Dots and Boxes game

        Returns:
            (result, score, rounds) where result is 'win', 'loss', or 'draw'
        """
        game = DotsBoxesGame()
        rounds = 0
        max_rounds = 500  # Prevent infinite games

        # Track moves for learning
        game_moves = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting game - AI plays {ai_color.name}")
            print(f"{'='*70}")
            print(game.board)

        while not game.is_game_over() and rounds < max_rounds:
            current_player = game.current_player

            # Select move
            if current_player == ai_color:
                move = self.select_move(game, ai_color, use_learning=True)
            else:
                # Opponent uses random moves (bootstrapping)
                legal_moves = game.get_legal_moves()
                move = random.choice(legal_moves) if legal_moves else None

            if move is None:
                break  # No legal moves

            # Record move features (for learning)
            if current_player == ai_color:
                piece_type = 'edge'

                category = self.categorize_move(game.board, move, ai_color)

                distance = self.distance_from_center(move.dot1, move.dot2)

                red_boxes, blue_boxes = game.board.get_box_counts()
                boxes_completed = red_boxes + blue_boxes
                legal_edges = game.board.get_legal_edges()
                phase = self.get_game_phase(boxes_completed, len(legal_edges))

                game_moves.append({
                    'piece_type': piece_type,
                    'category': category,
                    'distance': distance,
                    'phase': phase
                })

            # Make move
            success, completed_box = game.make_move(move)
            rounds += 1

            if verbose:
                print(f"\nRound {rounds}: {current_player.name.upper()} - {move}")
                if completed_box:
                    print(f"  COMPLETED BOX! Player gets another turn.")
                print(game.board)

        # Calculate final score (DIFFERENTIAL!)
        score, result = self.scorer.calculate_final_score(
            game.board, ai_color, rounds
        )

        # Determine result from AI perspective
        winner = game.get_winner()
        if winner == ai_color:
            result = 'win'
        elif winner is not None:
            result = 'loss'
        else:
            result = 'draw'

        if verbose:
            red_boxes, blue_boxes = game.board.get_box_counts()
            print(f"\n{'='*70}")
            print(f"Game Over - {result.upper()}")
            print(f"Final Score: Red={red_boxes}, Blue={blue_boxes}")
            print(f"Rounds: {rounds}")
            print(f"Differential Score: {score:+.0f}")
            print(f"{'='*70}\n")

        # Record game for learning (directly update statistics)
        for move_data in game_moves:
            self.prioritizer._update_move_statistics(
                piece_type=move_data['piece_type'],
                move_category=move_data['category'],
                distance=move_data['distance'],
                phase=move_data['phase'],
                result=result,
                final_score=score
            )
        self.prioritizer.conn.commit()

        return result, score, rounds

    def train(self, num_games: int, verbose: bool = False,
             progress_interval: int = 10):
        """
        Train for a number of games

        Args:
            num_games: Number of games to play
            verbose: Print detailed game info
            progress_interval: Print progress every N games
        """
        print(f"{'='*70}")
        print(f"DOTS AND BOXES HEADLESS TRAINING")
        print(f"{'='*70}")
        print(f"Board: 5x5 dots (4x4 boxes)")
        print(f"Games to play: {num_games}")
        print(f"Database: {self.db_path}")
        print(f"Progress updates every {progress_interval} games")
        print(f"{'='*70}\n")

        start_time = time.time()

        for i in range(num_games):
            # Alternate colors
            ai_color = Color.RED if i % 2 == 0 else Color.BLUE

            # Play game
            result, score, rounds = self.play_game(ai_color, verbose)

            # Update statistics
            self.games_played += 1
            self.score_history.append(score)

            if result == 'win':
                self.wins += 1
            elif result == 'loss':
                self.losses += 1
            else:
                self.draws += 1

            # Progress update
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed

                avg_score = sum(self.score_history) / len(self.score_history)
                recent_avg = sum(self.score_history[-10:]) / min(10, len(self.score_history))

                win_rate = (self.wins / self.games_played) * 100

                print(f"\nProgress: {i+1}/{num_games} games")
                print(f"  W-L-D: {self.wins}-{self.losses}-{self.draws} ({win_rate:.1f}% wins)")
                print(f"  Avg Score: {avg_score:+.0f} (recent 10: {recent_avg:+.0f})")
                print(f"  Speed: {games_per_sec:.2f} games/sec")
                print(f"  Elapsed: {elapsed:.1f}s")

        # Final summary
        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total Games: {self.games_played}")
        print(f"Results: {self.wins}W - {self.losses}L - {self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played)*100:.1f}%")
        print(f"Average Score: {sum(self.score_history)/len(self.score_history):+.0f}")
        print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Speed: {self.games_played/total_time:.2f} games/sec")
        print(f"{'='*70}\n")

    def show_learned_patterns(self, limit: int = 10):
        """Display top learned patterns"""
        print(f"\n{'='*70}")
        print(f"TOP {limit} LEARNED DOTS & BOXES PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<14} {'Distance':<10} {'Phase':<12} {'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*80}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<14} {distance:<10} {phase:<12} {times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Dots and Boxes Headless Trainer')
    parser.add_argument('num_games', type=int, nargs='?', default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--progress', '-p', type=int, default=10,
                       help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='dots_boxes_training.db',
                       help='Database path (default: dots_boxes_training.db)')
    parser.add_argument('--show-patterns', '-s', type=int, nargs='?',
                       const=10, default=None,
                       help='Show top N learned patterns')

    args = parser.parse_args()

    trainer = DotsBoxesHeadlessTrainer(args.db)

    if args.show_patterns is not None:
        # Just show patterns and exit
        trainer.show_learned_patterns(args.show_patterns)
    else:
        # Run training
        trainer.train(args.num_games, args.verbose, args.progress)

        # Show learned patterns
        trainer.show_learned_patterns(10)


if __name__ == '__main__':
    main()
