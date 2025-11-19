#!/usr/bin/env python3
"""
Othello Headless Trainer - Fast Training Without GUI

Philosophy: Same as chess trainer
- Differential scoring
- Pattern learning
- Observation-based (future)
- No GUI overhead
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from typing import List, Tuple
from collections import defaultdict

from othello.othello_board import OthelloBoard, Color, Move
from othello.othello_game import OthelloGame
from othello.othello_scorer import OthelloScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class OthelloHeadlessTrainer:
    """
    Fast terminal-based othello trainer

    Uses differential scoring and pattern learning
    Same philosophy as chess trainer
    """

    def __init__(self, db_path='othello_training.db'):
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = OthelloScorer()

        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []

        # Learning stats
        self.patterns_by_category = defaultdict(int)

    def is_corner(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a corner"""
        row, col = pos
        return (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]

    def is_edge(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on edge (but not corner)"""
        row, col = pos
        is_boundary = row in [0, 7] or col in [0, 7]
        return is_boundary and not self.is_corner(pos)

    def is_center(self, pos: Tuple[int, int]) -> bool:
        """Check if position is in center 4x4 area"""
        row, col = pos
        return 2 <= row <= 5 and 2 <= col <= 5

    def categorize_move(self, move: Move) -> str:
        """
        Categorize a move for pattern learning

        Categories: corner, edge, center, capture
        """
        # Capture is special - takes priority
        if move.flipped_discs:
            return 'capture'

        # Otherwise, categorize by position
        if not move.is_pass:
            if self.is_corner(move.position):
                return 'corner'
            elif self.is_edge(move.position):
                return 'edge'
            elif self.is_center(move.position):
                return 'center'

        return 'quiet'

    def distance_from_center(self, pos: Tuple[int, int]) -> float:
        """
        Calculate distance from board center (3.5, 3.5)
        Used for pattern tracking
        """
        row, col = pos
        center_row, center_col = 3.5, 3.5
        distance = ((row - center_row) ** 2 + (col - center_col) ** 2) ** 0.5
        return int(distance * 2)  # Scale to 0-10 range

    def select_move(self, game: OthelloGame, ai_color: Color,
                   use_learning: bool = True) -> Move:
        """
        Select move using learned patterns

        Args:
            game: Current game state
            ai_color: AI's color
            use_learning: If True, use pattern prioritizer; else random
        """
        legal_moves = game.get_legal_moves(ai_color)

        if not legal_moves:
            return None

        if not use_learning:
            return random.choice(legal_moves)

        # Score each move using learned patterns
        move_scores = []

        for move in legal_moves:
            if move.is_pass:
                # Pass moves get low priority
                move_scores.append((move, 20.0))
                continue

            # Extract move features for pattern matching
            piece_type = 'disc'

            # Categorize move
            category = self.categorize_move(move)

            # Calculate distance from center
            distance = self.distance_from_center(move.position)

            # Determine game phase
            total_discs = game.board.count_discs(Color.BLACK) + \
                         game.board.count_discs(Color.WHITE)
            if total_discs < 20:
                phase = 'opening'
            elif total_discs < 50:
                phase = 'middlegame'
            else:
                phase = 'endgame'

            # Get priority from learned patterns (query cache directly)
            key = (piece_type, category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns
                if category == 'corner':
                    priority = 90.0  # Corners are extremely valuable in Othello
                elif category == 'capture':
                    priority = 70.0
                elif category == 'edge':
                    priority = 40.0
                elif category == 'center':
                    priority = 50.0
                else:  # quiet
                    priority = 25.0

            move_scores.append((move, priority))

        # Select move based on priorities (with exploration)
        if random.random() < 0.1:  # 10% exploration
            return random.choice(legal_moves)
        else:
            # Choose move with highest priority
            move_scores.sort(key=lambda x: x[1], reverse=True)
            return move_scores[0][0]

    def play_game(self, ai_color: Color, verbose: bool = False) -> Tuple[str, float, int]:
        """
        Play one othello game

        Returns:
            (result, score, rounds) where result is 'win', 'loss', or 'draw'
        """
        game = OthelloGame()
        rounds = 0
        max_rounds = 200  # Prevent infinite games

        # Track moves for learning
        game_moves = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting game - AI plays {ai_color.name}")
            print(f"{'='*70}")
            print(game.board)

        while not game.is_game_over() and rounds < max_rounds:
            current_color = game.board.turn

            # Select move
            if current_color == ai_color:
                move = self.select_move(game, ai_color, use_learning=True)
            else:
                # Opponent uses random moves (bootstrapping)
                legal_moves = game.get_legal_moves(current_color)
                move = random.choice(legal_moves) if legal_moves else None

            if move is None:
                break  # No legal moves

            # Record move features (for learning)
            if current_color == ai_color:
                piece_type = 'disc'

                category = self.categorize_move(move)

                if not move.is_pass:
                    distance = self.distance_from_center(move.position)
                else:
                    distance = 0

                total_discs = game.board.count_discs(Color.BLACK) + \
                             game.board.count_discs(Color.WHITE)
                if total_discs < 20:
                    phase = 'opening'
                elif total_discs < 50:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

                game_moves.append({
                    'piece_type': piece_type,
                    'category': category,
                    'distance': distance,
                    'phase': phase
                })

            # Make move
            game.make_move(move)
            rounds += 1

            if verbose:
                print(f"\nRound {rounds}: {current_color.name} - {move.notation()}")
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
            print(f"\n{'='*70}")
            print(f"Game Over - {result.upper()}")
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
        print(f"OTHELLO HEADLESS TRAINING")
        print(f"{'='*70}")
        print(f"Games to play: {num_games}")
        print(f"Database: {self.db_path}")
        print(f"Progress updates every {progress_interval} games")
        print(f"{'='*70}\n")

        start_time = time.time()

        for i in range(num_games):
            # Alternate colors
            ai_color = Color.BLACK if i % 2 == 0 else Color.WHITE

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
        print(f"TOP {limit} LEARNED OTHELLO PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<12} {'Distance':<10} {'Phase':<12} {'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*80}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<12} {distance:<10} {phase:<12} {times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Othello Headless Trainer')
    parser.add_argument('num_games', type=int, nargs='?', default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--progress', '-p', type=int, default=10,
                       help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='othello_training.db',
                       help='Database path (default: othello_training.db)')
    parser.add_argument('--show-patterns', '-s', type=int, nargs='?',
                       const=10, default=None,
                       help='Show top N learned patterns')

    args = parser.parse_args()

    trainer = OthelloHeadlessTrainer(args.db)

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
