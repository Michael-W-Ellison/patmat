#!/usr/bin/env python3
"""
Lines of Action headless trainer for pattern learning.

Trains AI to play Lines of Action through self-play, learning patterns like:
- Grouping strategies (reducing number of separate groups)
- Connection patterns (adjacent piece placement)
- Capture opportunities
- Centralization tactics
- Mobility management

Traditional AI struggles with LOA due to connectivity evaluation complexity.
This observation-based learner discovers grouping patterns naturally.
"""

import sys
import os
import random
import argparse
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loa.loa_board import LOABoard, Color
from loa.loa_game import LOAGame, LOAMove
from loa.loa_scorer import LOAScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class LOAHeadlessTrainer:
    """Trains Lines of Action AI through self-play with pattern learning"""

    def __init__(self, db_path: str = 'loa_training.db'):
        """Initialize trainer"""
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = LOAScorer()

        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_score = 0

    def select_move(self, game: LOAGame, legal_moves: List[LOAMove],
                    exploration_rate: float = 0.15) -> LOAMove:
        """
        Select a move using learned patterns + exploration.

        Uses pattern priorities from database to guide move selection.
        """
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Exploration: random move
        if random.random() < exploration_rate:
            return random.choice(legal_moves)

        # Exploitation: use learned patterns
        move_scores = []

        for move in legal_moves:
            # Simulate move
            new_game = game.make_move(move)

            # Get move category
            category = self.scorer.get_move_category(game, new_game, move, game.current_player)

            # Get distance metric
            distance = self.scorer.get_distance_metric(move, game.board)

            # Determine game phase based on group count
            groups = game.board.count_groups(game.current_player)
            if groups > 6:
                phase = 'opening'
            elif groups > 3:
                phase = 'middlegame'
            else:
                phase = 'endgame'

            # Get priority from learned patterns
            key = ('piece', category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns
                if category == 'winning':
                    priority = 100.0
                elif category == 'grouping':
                    priority = 90.0
                elif category == 'connecting':
                    priority = 85.0
                elif category == 'capture':
                    priority = 70.0
                elif category == 'centralizing':
                    priority = 60.0
                else:  # quiet
                    priority = 30.0

            move_scores.append((move, priority))

        # Select move with highest priority
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]

    def play_game(self, verbose: bool = False) -> Tuple[str, float, int]:
        """
        Play one game of Lines of Action.

        Returns: (result, differential_score, rounds)
        """
        game = LOAGame()
        game_moves = []  # Track moves for learning
        rounds = 0

        if verbose:
            print("\n" + "="*70)
            print("Starting new Lines of Action game")
            print("="*70)
            print(game)

        while not game.is_game_over():
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                # No legal moves → opponent wins
                game.winner = game.current_player.opposite()
                break

            # Select move (Black uses learning, White random for training)
            if game.current_player == Color.BLACK:
                move = self.select_move(game, legal_moves, exploration_rate=0.15)
            else:
                # White plays with some strategy (not purely random)
                move = self.select_move(game, legal_moves, exploration_rate=0.5)

            # Record move data for learning (Black's moves only)
            game_before = game
            new_game = game.make_move(move)

            if game.current_player == Color.BLACK:
                category = self.scorer.get_move_category(game_before, new_game, move, Color.BLACK)
                distance = self.scorer.get_distance_metric(move, game_before.board)

                groups = game_before.board.count_groups(Color.BLACK)
                if groups > 6:
                    phase = 'opening'
                elif groups > 3:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

                game_moves.append({
                    'piece_type': 'piece',
                    'category': category,
                    'distance': distance,
                    'phase': phase
                })

            game = new_game
            rounds += 1

            if verbose and rounds % 10 == 0:
                print(f"\nRound {rounds}:")
                print(game)

            # Safety: prevent infinite games
            if rounds > 200:
                if verbose:
                    print("Game exceeded 200 rounds, ending as draw")
                game.is_draw = True
                break

        # Get result from Black's perspective
        result = game.get_result(Color.BLACK)

        # Calculate final differential score
        score = self.scorer.score_game(game, Color.BLACK)

        # Add win/loss bonus
        if result == 'win':
            score += 1000
        elif result == 'loss':
            score -= 1000

        # Add time bonus (faster wins are better)
        if result == 'win':
            time_bonus = max(0, 100 - rounds)
            score += time_bonus

        if verbose:
            print(f"\n{'='*70}")
            print(f"Game Over - {result.upper()}")
            print(f"Rounds: {rounds}")
            print(f"Final position:")
            print(game)
            print(f"Differential Score: {score:+.0f}")
            print(f"{'='*70}\n")

        # Record game for learning
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
              progress_interval: int = 10) -> dict:
        """
        Train for a number of games.

        Returns dictionary with training statistics.
        """
        print(f"\n{'='*70}")
        print(f"LINES OF ACTION TRAINING - {num_games} games")
        print(f"{'='*70}\n")

        for i in range(num_games):
            result, score, rounds = self.play_game(verbose=verbose)

            # Update statistics
            self.games_played += 1
            self.total_score += score

            if result == 'win':
                self.black_wins += 1
            elif result == 'loss':
                self.white_wins += 1
            else:
                self.draws += 1

            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                win_rate = (self.black_wins / self.games_played) * 100
                avg_score = self.total_score / self.games_played

                print(f"Game {i+1}/{num_games}: {result.upper()} "
                      f"(Rounds: {rounds}, Score: {score:+.0f})")
                print(f"  Overall: {self.black_wins}W-{self.white_wins}L-{self.draws}D "
                      f"({win_rate:.1f}% win rate, avg score: {avg_score:+.0f})")

        # Final statistics
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Games Played: {self.games_played}")
        print(f"Results: {self.black_wins}W-{self.white_wins}L-{self.draws}D")
        print(f"Win Rate: {(self.black_wins/self.games_played)*100:.1f}%")
        print(f"Average Score: {self.total_score/self.games_played:+.0f}")
        print(f"{'='*70}\n")

        return {
            'games_played': self.games_played,
            'wins': self.black_wins,
            'losses': self.white_wins,
            'draws': self.draws,
            'win_rate': self.black_wins / self.games_played,
            'avg_score': self.total_score / self.games_played
        }

    def show_learned_patterns(self, limit: int = 10):
        """Display top learned patterns"""
        print(f"\n{'='*70}")
        print(f"TOP {limit} LEARNED LINES OF ACTION PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<15} {'Dist':<6} {'Phase':<12} "
              f"{'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*90}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<15} {distance:<6} {phase:<12} "
                  f"{times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train Lines of Action AI')
    parser.add_argument('num_games', nargs='?', type=int, default=None,
                        help='Number of games to play')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=10,
                        help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='loa_training.db',
                        help='Database path (default: loa_training.db)')
    parser.add_argument('--show-patterns', nargs='?', const=10, type=int,
                        help='Show top N learned patterns and exit')

    args = parser.parse_args()

    trainer = LOAHeadlessTrainer(db_path=args.db)

    # Show patterns mode
    if args.show_patterns is not None:
        trainer.show_learned_patterns(limit=args.show_patterns)
        return

    # Training mode
    if args.num_games is None:
        print("Error: num_games required (or use --show-patterns)")
        parser.print_help()
        sys.exit(1)

    trainer.train(
        num_games=args.num_games,
        verbose=args.verbose,
        progress_interval=args.progress
    )

    # Show top patterns after training
    print("\nTop 5 patterns learned:")
    trainer.show_learned_patterns(limit=5)


if __name__ == '__main__':
    main()
