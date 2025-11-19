#!/usr/bin/env python3
"""
Nine Men's Morris headless trainer for pattern learning.

Trains AI to play Nine Men's Morris through self-play, learning patterns like:
- Mill formation strategies
- 2-piece mill setups (threats)
- Blocking opponent mills
- Piece removal priorities
- Phase-specific strategies (placement, movement, flying)

Traditional AI often misses subtle mill patterns.
This observation-based learner discovers winning formations naturally.
"""

import sys
import os
import random
import argparse
from typing import List, Tuple, Optional, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morris.morris_board import MorrisBoard, Color
from morris.morris_game import MorrisGame, MorrisMove, GamePhase
from morris.morris_scorer import MorrisScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class MorrisHeadlessTrainer:
    """Trains Nine Men's Morris AI through self-play with pattern learning"""

    def __init__(self, db_path: str = 'morris_training.db'):
        """Initialize trainer"""
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = MorrisScorer()

        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_score = 0

    def select_move(self, game: MorrisGame, legal_moves: List[Union[MorrisMove, int]],
                    exploration_rate: float = 0.15) -> Union[MorrisMove, int]:
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
            distance = self.scorer.get_distance_metric(move, game)

            # Determine game phase
            phase_str = game.phase.value if game.phase else 'unknown'

            # Get priority from learned patterns
            key = ('piece', category, distance, phase_str)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns
                if category == 'form_mill':
                    priority = 95.0
                elif category == 'remove_piece':
                    priority = 90.0
                elif category == 'create_2mill':
                    priority = 75.0
                elif category == 'block_mill':
                    priority = 70.0
                elif category == 'flying':
                    priority = 60.0
                elif category == 'movement':
                    priority = 45.0
                elif category == 'placement':
                    priority = 40.0
                else:  # quiet
                    priority = 30.0

            move_scores.append((move, priority))

        # Select move with highest priority
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]

    def play_game(self, verbose: bool = False) -> Tuple[str, float, int]:
        """
        Play one game of Nine Men's Morris.

        Returns: (result, differential_score, rounds)
        """
        game = MorrisGame()
        game_moves = []  # Track moves for learning
        rounds = 0

        if verbose:
            print("\n" + "="*70)
            print("Starting new Nine Men's Morris game")
            print("="*70)

        while not game.is_game_over():
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                # No legal moves → opponent wins
                break

            # Select move (White uses learning, Black random for now)
            if game.current_player == Color.WHITE:
                move = self.select_move(game, legal_moves, exploration_rate=0.15)
            else:
                # Black plays with some strategy (not purely random)
                move = self.select_move(game, legal_moves, exploration_rate=0.5)

            # Record move data for learning
            game_before = game
            new_game = game.make_move(move)

            if game.current_player == Color.WHITE:
                category = self.scorer.get_move_category(game_before, new_game, move, Color.WHITE)
                distance = self.scorer.get_distance_metric(move, game_before)
                phase_str = game_before.phase.value if game_before.phase else 'unknown'

                game_moves.append({
                    'piece_type': 'piece',
                    'category': category,
                    'distance': distance,
                    'phase': phase_str
                })

            game = new_game
            rounds += 1

            if verbose and rounds % 5 == 0:
                print(f"\nRound {rounds} ({game.phase.value}):")
                print(game.board)
                print(f"White: {game.board.count_pieces(Color.WHITE)} pieces, "
                      f"Black: {game.board.count_pieces(Color.BLACK)} pieces")

            # Safety: prevent infinite games
            if rounds > 200:
                if verbose:
                    print("Game exceeded 200 rounds, ending as draw")
                game.is_draw = True
                break

        # Get result from White's perspective
        result = game.get_result(Color.WHITE)

        # Calculate final differential score
        score = self.scorer.score_game(game, Color.WHITE)

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
            print(game.board)
            print(f"White: {game.board.count_pieces(Color.WHITE)} pieces, "
                  f"Black: {game.board.count_pieces(Color.BLACK)} pieces")
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
        print(f"NINE MEN'S MORRIS TRAINING - {num_games} games")
        print(f"{'='*70}\n")

        for i in range(num_games):
            result, score, rounds = self.play_game(verbose=verbose)

            # Update statistics
            self.games_played += 1
            self.total_score += score

            if result == 'win':
                self.white_wins += 1
            elif result == 'loss':
                self.black_wins += 1
            else:
                self.draws += 1

            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                win_rate = (self.white_wins / self.games_played) * 100
                avg_score = self.total_score / self.games_played

                print(f"Game {i+1}/{num_games}: {result.upper()} "
                      f"(Rounds: {rounds}, Score: {score:+.0f})")
                print(f"  Overall: {self.white_wins}W-{self.black_wins}L-{self.draws}D "
                      f"({win_rate:.1f}% win rate, avg score: {avg_score:+.0f})")

        # Final statistics
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Games Played: {self.games_played}")
        print(f"Results: {self.white_wins}W-{self.black_wins}L-{self.draws}D")
        print(f"Win Rate: {(self.white_wins/self.games_played)*100:.1f}%")
        print(f"Average Score: {self.total_score/self.games_played:+.0f}")
        print(f"{'='*70}\n")

        return {
            'games_played': self.games_played,
            'wins': self.white_wins,
            'losses': self.black_wins,
            'draws': self.draws,
            'win_rate': self.white_wins / self.games_played,
            'avg_score': self.total_score / self.games_played
        }

    def show_learned_patterns(self, limit: int = 10):
        """Display top learned patterns"""
        print(f"\n{'='*70}")
        print(f"TOP {limit} LEARNED NINE MEN'S MORRIS PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<15} {'Ring':<6} {'Phase':<12} "
              f"{'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*90}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            ring_name = ['Outer', 'Middle', 'Inner'][distance] if distance < 3 else f"Ring{distance}"
            print(f"{piece_type:<8} {category:<15} {ring_name:<6} {phase:<12} "
                  f"{times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train Nine Men\'s Morris AI')
    parser.add_argument('num_games', nargs='?', type=int, default=None,
                        help='Number of games to play')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=10,
                        help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='morris_training.db',
                        help='Database path (default: morris_training.db)')
    parser.add_argument('--show-patterns', nargs='?', const=10, type=int,
                        help='Show top N learned patterns and exit')

    args = parser.parse_args()

    trainer = MorrisHeadlessTrainer(db_path=args.db)

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
