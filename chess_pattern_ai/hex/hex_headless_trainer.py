#!/usr/bin/env python3
"""
Hex Headless Trainer - Fast Training Without GUI

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

from hex.hex_board import HexBoard, Color, Move
from hex.hex_game import HexGame
from hex.hex_scorer import HexScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class HexHeadlessTrainer:
    """
    Fast terminal-based Hex trainer

    Uses differential scoring and pattern learning
    Same philosophy as chess trainer
    """

    def __init__(self, db_path='hex_training.db', board_size: int = 11):
        self.db_path = db_path
        self.board_size = board_size
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = HexScorer()

        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []

        # Learning stats
        self.patterns_by_category = defaultdict(int)

    def categorize_move(self, board: HexBoard, move: Move, ai_color: Color) -> str:
        """
        Categorize a move for pattern learning

        Hex-specific categories:
        - 'bridge' - Creating or extending connection (bridge patterns)
        - 'block' - Blocking opponent's path to connection
        - 'center' - Playing near center (strong strategic position)
        - 'edge' - Playing near board edges
        - 'quiet' - Normal positional move
        """
        row, col = move.position
        center = self.board_size // 2

        # Check if this move creates a connection (bridge pattern)
        board_copy = board.copy()
        board_copy.make_move(move)

        # Get connection strength before and after
        own_positions_before = {s.position for s in board.stones if s.color == ai_color}
        own_positions_after = {s.position for s in board_copy.stones if s.color == ai_color}

        opp_color = Color.BLUE if ai_color == Color.RED else Color.RED
        opp_positions_before = {s.position for s in board.stones if s.color == opp_color}
        opp_positions_after = {s.position for s in board_copy.stones if s.color == opp_color}

        # Check if move is near existing stones (bridge)
        is_bridge = False
        if own_positions_before:
            for own_pos in own_positions_before:
                own_row, own_col = own_pos
                dist = max(abs(row - own_row), abs(col - own_col))
                if dist <= 2:  # Bridge distance
                    is_bridge = True
                    break

        # Check if move blocks opponent
        is_block = False
        if opp_positions_before:
            for opp_pos in opp_positions_before:
                opp_row, opp_col = opp_pos
                dist = max(abs(row - opp_row), abs(col - opp_col))
                if dist <= 2:  # Defensive distance
                    is_block = True
                    break

        # Check distances from center
        center_distance = max(abs(row - center), abs(col - center))

        # Categorize
        if is_bridge:
            return 'bridge'
        elif is_block:
            return 'block'
        elif center_distance <= 2:
            return 'center'
        elif center_distance >= self.board_size - 2:
            return 'edge'
        else:
            return 'quiet'

    def distance_from_center(self, row: int, col: int) -> int:
        """
        Calculate distance from center (Chebyshev distance)
        Center control is key in Hex
        """
        center = self.board_size // 2
        return max(abs(row - center), abs(col - center))

    def get_game_phase(self, total_stones: int, board_fill_percent: float) -> str:
        """
        Determine game phase based on move count and board fill

        Args:
            total_stones: Total stones on board
            board_fill_percent: Percentage of board filled (0-100)

        Returns: 'opening', 'middlegame', or 'endgame'
        """
        total_capacity = self.board_size * self.board_size

        if total_stones < 10 or board_fill_percent < 5:
            return 'opening'
        elif board_fill_percent < 70:
            return 'middlegame'
        else:
            return 'endgame'

    def select_move(self, game: HexGame, ai_color: Color,
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
            # Extract move features for pattern matching
            piece_type = 'stone'

            # Categorize move
            category = self.categorize_move(game.board, move, ai_color)

            # Calculate distance from center
            distance = self.distance_from_center(move.position[0], move.position[1])

            # Determine game phase
            total_stones = sum(1 for stone in game.board.stones)
            total_capacity = self.board_size * self.board_size
            board_fill_percent = (total_stones / total_capacity) * 100
            phase = self.get_game_phase(total_stones, board_fill_percent)

            # Get priority from learned patterns (query cache directly)
            key = (piece_type, category, distance, phase)

            if key in self.prioritizer.move_priorities:
                priority = self.prioritizer.move_priorities[key]['priority']
            else:
                # Default priorities for unseen patterns (Hex-specific)
                if category == 'bridge':
                    priority = 90.0   # Connection building is crucial
                elif category == 'block':
                    priority = 85.0   # Blocking opponent is important
                elif category == 'center':
                    priority = 70.0   # Center control
                elif category == 'edge':
                    priority = 30.0   # Edge moves
                else:  # quiet
                    priority = 40.0

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
        Play one Hex game

        Returns:
            (result, score, rounds) where result is 'win', 'loss', or 'draw'
        """
        game = HexGame(board_size=self.board_size)
        rounds = 0
        max_rounds = 500  # Prevent infinite games

        # Track moves for learning
        game_moves = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting game - AI plays {ai_color.name}")
            print(f"Board size: {self.board_size}x{self.board_size}")
            print(f"{'='*70}")
            print(game.board)

        while not game.is_game_over() and rounds < max_rounds:
            # Alternate colors in turn
            current_color = Color.RED if rounds % 2 == 0 else Color.BLUE

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
                piece_type = 'stone'

                category = self.categorize_move(game.board, move, ai_color)

                distance = self.distance_from_center(move.position[0], move.position[1])

                total_stones = sum(1 for stone in game.board.stones)
                total_capacity = self.board_size * self.board_size
                board_fill_percent = (total_stones / total_capacity) * 100
                phase = self.get_game_phase(total_stones, board_fill_percent)

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
                row, col = move.position
                notation = move.notation()
                print(f"\nRound {rounds}: {current_color.name.upper()} - {notation}")
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
        print(f"HEX HEADLESS TRAINING")
        print(f"{'='*70}")
        print(f"Board size: {self.board_size}x{self.board_size}")
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
        print(f"TOP {limit} LEARNED HEX PATTERNS")
        print(f"{'='*70}\n")

        patterns = self.prioritizer.get_top_patterns(limit)

        print(f"{'Piece':<8} {'Category':<10} {'Distance':<10} {'Phase':<12} {'Games':<8} {'Win%':<8} {'Priority':<10}")
        print(f"{'-'*76}")

        for pattern in patterns:
            piece_type, category, distance, phase, times_seen, win_rate, confidence, priority = pattern
            win_pct = win_rate * 100
            print(f"{piece_type:<8} {category:<10} {distance:<10} {phase:<12} {times_seen:<8} {win_pct:<8.1f} {priority:<10.1f}")

        print(f"{'='*70}\n")


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Hex Headless Trainer')
    parser.add_argument('num_games', type=int, nargs='?', default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--progress', '-p', type=int, default=10,
                       help='Progress update interval (default: 10)')
    parser.add_argument('--db', type=str, default='hex_training.db',
                       help='Database path (default: hex_training.db)')
    parser.add_argument('--show-patterns', '-s', type=int, nargs='?',
                       const=10, default=None,
                       help='Show top N learned patterns')
    parser.add_argument('--size', type=int, default=11, choices=[11, 13],
                       help='Board size: 11 or 13 (default: 11)')

    args = parser.parse_args()

    trainer = HexHeadlessTrainer(args.db, args.size)

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
