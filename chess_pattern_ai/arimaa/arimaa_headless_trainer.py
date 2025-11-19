#!/usr/bin/env python3
"""
Arimaa headless trainer for pattern learning.

Trains AI to play Arimaa through self-play, learning patterns like:
- Rabbit advancement strategies
- Push/pull tactics (using piece strength hierarchy)
- Trap control
- Material preservation
- Forward advancement

Arimaa was specifically designed to be difficult for computers due to:
- Huge branching factor (16^4)
- Complex push/pull mechanics
- Trap control requiring strategic understanding

This observation-based learner discovers tactical patterns naturally.
"""

import sys
import os
import random
import argparse
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arimaa.arimaa_board import ArimaaBoard, Color
from arimaa.arimaa_game import ArimaaGame, ArimaaStep
from arimaa.arimaa_scorer import ArimaaScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class ArimaaHeadlessTrainer:
    """Trains Arimaa AI through self-play with pattern learning"""

    def __init__(self, db_path: str = 'arimaa_training.db'):
        """Initialize trainer"""
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = ArimaaScorer()

        # Statistics
        self.games_played = 0
        self.gold_wins = 0
        self.silver_wins = 0
        self.draws = 0
        self.total_score = 0

    def select_step(self, game: ArimaaGame, legal_steps: List[ArimaaStep],
                    exploration_rate: float = 0.2) -> ArimaaStep:
        """
        Select a step using learned patterns + exploration.

        Uses pattern priorities from database to guide step selection.
        """
        if not legal_steps:
            raise ValueError("No legal steps available")

        # Exploration: random step
        if random.random() < exploration_rate:
            return random.choice(legal_steps)

        # Exploitation: use learned patterns
        step_scores = []

        for step in legal_steps:
            # Simulate step
            new_game = game.make_step(step)

            # Get step category
            category = self.scorer.get_move_category(game, new_game, step, game.current_player)

            # Get distance metric
            distance = self.scorer.get_distance_metric(step, game.board)

            # Determine game phase
            total_pieces = len(game.board.pieces_gold) + len(game.board.pieces_silver)
            if total_pieces > 24:
                phase = 'opening'
            elif total_pieces > 16:
                phase = 'middlegame'
            else:
                phase = 'endgame'

            # Get priority from learned patterns
            key = ('step', category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns
                if category == 'rabbit_goal':
                    priority = 100.0
                elif category == 'rabbit_advance':
                    priority = 90.0
                elif category == 'capture':
                    priority = 85.0
                elif category == 'push':
                    priority = 70.0
                elif category == 'pull':
                    priority = 70.0
                elif category == 'trap_control':
                    priority = 65.0
                elif category == 'advance':
                    priority = 55.0
                else:  # quiet
                    priority = 30.0

            step_scores.append((step, priority))

        # Select step with highest priority
        step_scores.sort(key=lambda x: x[1], reverse=True)
        return step_scores[0][0]

    def execute_turn(self, game: ArimaaGame, max_steps: int = 4, verbose: bool = False) -> Tuple[ArimaaGame, List[dict]]:
        """
        Execute a complete turn (up to 4 steps).

        Returns: (new_game, step_data_list)
        """
        turn_steps = []
        current_game = game

        for step_num in range(max_steps):
            if current_game.is_game_over():
                break

            legal_steps = current_game.get_legal_steps()
            if not legal_steps:
                break

            # Select step
            step = self.select_step(current_game, legal_steps,
                                   exploration_rate=0.2 if game.current_player == Color.GOLD else 0.5)

            # Record step data
            game_before = current_game
            new_game = current_game.make_step(step)

            if game.current_player == Color.GOLD:
                category = self.scorer.get_move_category(game_before, new_game, step, Color.GOLD)
                distance = self.scorer.get_distance_metric(step, game_before.board)

                total_pieces = len(game_before.board.pieces_gold) + len(game_before.board.pieces_silver)
                if total_pieces > 24:
                    phase = 'opening'
                elif total_pieces > 16:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

                turn_steps.append({
                    'piece_type': 'step',
                    'category': category,
                    'distance': distance,
                    'phase': phase
                })

            current_game = new_game

            if verbose:
                print(f"  Step {step_num + 1}: {step} ({category if game.current_player == Color.GOLD else 'opponent'})")

        return current_game, turn_steps

    def play_game(self, verbose: bool = False) -> Tuple[str, float, int]:
        """
        Play one game of Arimaa.

        Returns: (result, differential_score, turns)
        """
        game = ArimaaGame()
        all_steps = []  # Track all steps for learning
        turns = 0

        if verbose:
            print("\n" + "="*70)
            print("Starting new Arimaa game")
            print("="*70)

        while not game.is_game_over() and turns < 100:
            # Execute turn (up to 4 steps)
            game, turn_steps = self.execute_turn(game, verbose=verbose)
            all_steps.extend(turn_steps)
            turns += 1

            if verbose and turns % 5 == 0:
                print(f"\nTurn {turns}:")
                print(game)

        # Safety: prevent extremely long games
        if turns >= 100:
            if verbose:
                print("Game exceeded 100 turns, ending as draw")
            game.is_draw = True

        # Get result from Gold's perspective
        result = game.get_result(Color.GOLD)

        # Calculate final differential score
        score = self.scorer.score_game(game, Color.GOLD)

        # Add win/loss bonus
        if result == 'win':
            score += 2000
        elif result == 'loss':
            score -= 2000

        # Add efficiency bonus (shorter wins are better)
        if result == 'win':
            efficiency = max(0, 100 - turns)
            score += efficiency * 10

        if verbose:
            print(f"\n{'='*70}")
            print(f"Game Over - {result.upper()}")
            print(f"Turns: {turns}")
            print(f"Final position:")
            print(game)
            print(f"Differential Score: {score:+.0f}")
            print(f"{'='*70}\n")

        # Record game for learning
        for step_data in all_steps:
            self.prioritizer._update_move_statistics(
                piece_type=step_data['piece_type'],
                move_category=step_data['category'],
                distance=step_data['distance'],
                phase=step_data['phase'],
                result=result,
                final_score=score
            )
        self.prioritizer.conn.commit()

        return result, score, turns

    def train(self, num_games: int, verbose: bool = False,
              progress_interval: int = 5) -> dict:
        """
        Train for a number of games.

        Returns dictionary with training statistics.
        """
        print(f"\n{'='*70}")
        print(f"ARIMAA TRAINING - {num_games} games")
        print(f"{'='*70}\n")

        for i in range(num_games):
            result, score, turns = self.play_game(verbose=verbose)

            # Update statistics
            self.games_played += 1
            self.total_score += score

            if result == 'win':
                self.gold_wins += 1
            elif result == 'loss':
                self.silver_wins += 1
            else:
                self.draws += 1

            # Progress update
            if (i + 1) % progress_interval == 0 or (i + 1) == num_games:
                win_rate = (self.gold_wins / self.games_played) * 100
                avg_score = self.total_score / self.games_played

                print(f"Game {i+1}/{num_games}: {result.upper()} "
                      f"(Turns: {turns}, Score: {score:+.0f})")
                print(f"  Overall: {self.gold_wins}W-{self.silver_wins}L-{self.draws}D "
                      f"({win_rate:.1f}% win rate, avg score: {avg_score:+.0f})")

        # Final statistics
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Games Played: {self.games_played}")
        print(f"Results: {self.gold_wins}W-{self.silver_wins}L-{self.draws}D")
        print(f"Win Rate: {(self.gold_wins/self.games_played)*100:.1f}%")
        print(f"Average Score: {self.total_score/self.games_played:+.0f}")
        print(f"{'='*70}\n")

        return {
            'games_played': self.games_played,
            'wins': self.gold_wins,
            'losses': self.silver_wins,
            'draws': self.draws,
            'win_rate': self.gold_wins / self.games_played,
            'avg_score': self.total_score / self.games_played
        }

    def show_learned_patterns(self, limit: int = 10):
        """Display top learned patterns"""
        print(f"\n{'='*70}")
        print(f"TOP {limit} LEARNED ARIMAA PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Type':<8} {'Category':<18} {'Dist':<6} {'Phase':<12} "
              f"{'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*90}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<18} {distance:<6} {phase:<12} "
                  f"{times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train Arimaa AI')
    parser.add_argument('num_games', nargs='?', type=int, default=None,
                        help='Number of games to play')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=5,
                        help='Progress update interval (default: 5)')
    parser.add_argument('--db', type=str, default='arimaa_training.db',
                        help='Database path (default: arimaa_training.db)')
    parser.add_argument('--show-patterns', nargs='?', const=10, type=int,
                        help='Show top N learned patterns and exit')

    args = parser.parse_args()

    trainer = ArimaaHeadlessTrainer(db_path=args.db)

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
