#!/usr/bin/env python3
"""
Go Headless Trainer - Fast Training Without GUI

Philosophy: Same as chess and checkers trainer
- Differential scoring
- Pattern learning
- Observation-based (future)
- No GUI overhead
- Supports 9x9 and 19x19 boards
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from typing import List, Tuple, Optional
from collections import defaultdict

from go.go_board import GoBoard, Color, Move
from go.go_game import GoGame
from go.go_scorer import GoScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class GoHeadlessTrainer:
    """
    Fast terminal-based Go trainer

    Uses differential scoring and pattern learning
    Same philosophy as chess and checkers trainers
    Supports 9x9 and 19x19 boards
    """

    def __init__(self, db_path='go_training.db', board_size: int = 9):
        """
        Initialize Go trainer

        Args:
            db_path: Path to training database
            board_size: Board size (9 or 19)
        """
        self.db_path = db_path
        self.board_size = board_size
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = GoScorer()

        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []

        # Learning stats
        self.patterns_by_category = defaultdict(int)

    def _distance_from_center(self, pos: Tuple[int, int]) -> int:
        """
        Calculate distance from board center

        In Go, the center is important for influence and territory.
        Returns distance as Manhattan distance from board center.
        """
        if pos is None:  # Pass move
            return -1
        row, col = pos
        center = self.board_size // 2
        return abs(row - center) + abs(col - center)

    def select_move(self, game: GoGame, ai_color: Color,
                   use_learning: bool = True) -> Optional[Move]:
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
            # All pieces in Go are stones
            piece_type = 'stone'

            # Categorize move
            if move.position is None:
                # Pass move - categorize based on game state
                category = 'quiet'
                distance = -1  # Pass doesn't have a position
            else:
                # Placement move - check if it captures
                if move.captured and len(move.captured) > 0:
                    category = 'capture'
                else:
                    # Check if this move gains territory or defends
                    # For now, categorize non-capture moves as territory/quiet
                    # This could be enhanced by checking if move is near opponent stones
                    category = 'territory'

                # Distance from board center
                distance = self._distance_from_center(move.position)

            # Determine game phase (based on move count)
            # Go games typically last 150-200 moves on 19x19, 40-80 on 9x9
            move_count = len(game.board.move_history)
            total_intersections = self.board_size * self.board_size

            if self.board_size == 9:
                # 9x9 games are shorter
                if move_count < 20:
                    phase = 'opening'
                elif move_count < 50:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'
            else:  # 19x19
                # 19x19 games are longer
                if move_count < 60:
                    phase = 'opening'
                elif move_count < 150:
                    phase = 'middlegame'
                else:
                    phase = 'endgame'

            # Get priority from learned patterns (query cache directly)
            key = (piece_type, category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns
                if category == 'capture':
                    priority = 85.0  # Captures are valuable
                elif category == 'territory':
                    priority = 60.0  # Territory is good
                else:  # quiet/pass
                    priority = 20.0  # Pass is lower priority

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
        Play one Go game

        Returns:
            (result, score, moves) where result is 'win', 'loss', or 'draw'
        """
        game = GoGame(board_size=self.board_size)
        moves_played = 0
        max_moves = 1000  # Prevent infinite games

        # Track moves for learning
        game_moves = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting game - AI plays {ai_color.name} on {self.board_size}x{self.board_size} board")
            print(f"{'='*70}")
            print(game.board)

        while not game.is_game_over() and moves_played < max_moves:
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

            # Record move features (for learning) - only for non-pass moves
            if current_color == ai_color:
                piece_type = 'stone'

                if move.position is None:
                    # Pass move
                    category = 'quiet'
                    distance = -1
                else:
                    # Placement move
                    if move.captured and len(move.captured) > 0:
                        category = 'capture'
                    else:
                        category = 'territory'

                    distance = self._distance_from_center(move.position)

                # Phase calculation
                move_count = len(game.board.move_history)

                if self.board_size == 9:
                    if move_count < 20:
                        phase = 'opening'
                    elif move_count < 50:
                        phase = 'middlegame'
                    else:
                        phase = 'endgame'
                else:  # 19x19
                    if move_count < 60:
                        phase = 'opening'
                    elif move_count < 150:
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
            if move.position is not None:
                moves_played += 1

            if verbose and move.position is not None:
                print(f"\nMove {moves_played}: {current_color.name} - {move.notation()}")
                print(game.board)

        # Calculate final score (DIFFERENTIAL!)
        score, result = self.scorer.calculate_final_score(
            game.board, ai_color, moves_played
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
            print(f"Moves: {moves_played}")
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

        return result, score, moves_played

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
        print(f"GO HEADLESS TRAINING ({self.board_size}x{self.board_size})")
        print(f"{'='*70}")
        print(f"Games to play: {num_games}")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Database: {self.db_path}")
        print(f"Progress updates every {progress_interval} games")
        print(f"{'='*70}\n")

        start_time = time.time()

        for i in range(num_games):
            # Alternate colors
            ai_color = Color.BLACK if i % 2 == 0 else Color.WHITE

            # Play game
            result, score, moves = self.play_game(ai_color, verbose)

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
        print(f"TOP {limit} LEARNED GO PATTERNS ({self.board_size}x{self.board_size})")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<15} {'Distance':<10} {'Phase':<12} {'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*80}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<15} {distance:<10} {phase:<12} {times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Go Headless Trainer')
    parser.add_argument('num_games', type=int, nargs='?', default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--progress', '-p', type=int, default=10,
                       help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='go_training.db',
                       help='Database path (default: go_training.db)')
    parser.add_argument('--size', '-s', type=int, choices=[9, 19], default=9,
                       help='Board size: 9 or 19 (default: 9)')
    parser.add_argument('--show-patterns', type=int, nargs='?',
                       const=10, default=None,
                       help='Show top N learned patterns')

    args = parser.parse_args()

    # Validate board size
    if args.size not in [9, 19]:
        print(f"Error: Board size must be 9 or 19, got {args.size}")
        sys.exit(1)

    trainer = GoHeadlessTrainer(args.db, board_size=args.size)

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
