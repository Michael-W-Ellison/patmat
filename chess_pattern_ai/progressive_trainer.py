#!/usr/bin/env python3
"""
Progressive Training System - Adaptive Difficulty

Trains the AI against Stockfish with automatic difficulty scaling:
- Starts at Stockfish skill level 0
- After 10 consecutive wins, promotes to next level
- Tracks progress across all difficulty levels
- Uses differential scoring for learning
"""

import chess
import chess.engine
import json
import time
from datetime import datetime
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class ProgressiveTrainer:
    """Adaptive training against progressively harder opponents"""

    def __init__(self, stockfish_path='/usr/games/stockfish', db_path='progressive_training.db'):
        self.stockfish_path = stockfish_path
        self.scorer = GameScorer()
        self.prioritizer = LearnableMovePrioritizer(db_path)

        # Progressive training state
        self.current_level = 0
        self.max_level = 20  # Stockfish skill levels 0-20
        self.consecutive_wins = 0
        self.wins_needed = 10  # Consecutive wins to advance

        # Statistics per level
        self.level_stats = {}
        self.reset_level_stats()

        # Overall statistics
        self.total_games = 0
        self.games_by_level = {i: 0 for i in range(21)}
        self.score_history = []
        self.level_history = []

    def reset_level_stats(self):
        """Reset statistics for current level"""
        if self.current_level not in self.level_stats:
            self.level_stats[self.current_level] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'games': 0,
                'scores': [],
                'best_score': float('-inf'),
                'avg_score': 0
            }

    def play_ai_move(self, board, ai_color, time_limit=0.5):
        """AI makes a move using learned priorities and search"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        legal_moves = self.prioritizer.sort_moves_by_priority(board, legal_moves)

        # Depth-limited search with move ordering
        best_move = None
        best_score = -999999

        for move in legal_moves[:20]:  # Top 20 by learned priority
            board.push(move)

            # Simple 2-ply search
            score = self.evaluate_position(board, ai_color)

            # Look ahead one opponent move
            if not board.is_game_over():
                opp_moves = list(board.legal_moves)[:10]
                worst_response = 999999

                for opp_move in opp_moves:
                    board.push(opp_move)
                    response_score = self.evaluate_position(board, ai_color)
                    board.pop()

                    if response_score < worst_response:
                        worst_response = response_score

                score = worst_response

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_position(self, board, ai_color):
        """Evaluate position using differential scoring"""
        if board.is_checkmate():
            return 10000 if board.turn != ai_color else -10000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        ai_mat = self.scorer._calculate_material(board, ai_color)
        opp_mat = self.scorer._calculate_material(board, not ai_color)
        return ai_mat - opp_mat

    def play_game_vs_stockfish(self, ai_color, stockfish_level, verbose=False):
        """Play one game against Stockfish at specified level"""
        board = chess.Board()
        game_moves = []
        move_count = 0

        # Initialize Stockfish
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            engine.configure({"Skill Level": stockfish_level})
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            print(f"Make sure Stockfish is installed at: {self.stockfish_path}")
            print("You can specify a different path when creating ProgressiveTrainer")
            return None, None, None

        try:
            while not board.is_game_over() and move_count < 100:
                fen_before = board.fen()

                if board.turn == ai_color:
                    # AI move
                    move = self.play_ai_move(board, ai_color)
                    if move:
                        move_san = board.san(move)
                        board.push(move)
                        game_moves.append((fen_before, move.uci(), move_san))

                        if verbose:
                            advantage = self.evaluate_position(board, ai_color)
                            print(f"  AI:  {move_san:8s} | Advantage: {advantage:+6.0f}")
                else:
                    # Stockfish move
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                    if move:
                        board.push(move)
                        move_count += 1

                        if verbose:
                            print(f"  SF{stockfish_level}: {board.san(move):8s}")

            # Calculate final score
            rounds_played = board.fullmove_number
            final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)

            # Record for learning
            self.prioritizer.record_game_moves(game_moves, ai_color, result, final_score)

            return result, final_score, rounds_played

        finally:
            engine.quit()

    def handle_game_result(self, result, score):
        """Update statistics and check for level promotion"""
        stats = self.level_stats[self.current_level]

        stats['games'] += 1
        stats['scores'].append(score)
        stats['avg_score'] = sum(stats['scores']) / len(stats['scores'])

        if score > stats['best_score']:
            stats['best_score'] = score

        if result == 'win':
            stats['wins'] += 1
            self.consecutive_wins += 1
        elif result == 'loss':
            stats['losses'] += 1
            self.consecutive_wins = 0  # Reset streak
        else:
            stats['draws'] += 1
            self.consecutive_wins = 0  # Reset streak (draws don't count)

        # Check for promotion
        promoted = False
        if self.consecutive_wins >= self.wins_needed and self.current_level < self.max_level:
            self.current_level += 1
            self.consecutive_wins = 0
            self.reset_level_stats()
            promoted = True

        return promoted

    def train(self, max_games=1000, verbose=False):
        """Train with progressive difficulty"""
        print("=" * 80)
        print("PROGRESSIVE TRAINING - Adaptive Difficulty vs Stockfish")
        print("=" * 80)
        print(f"Starting at level {self.current_level}")
        print(f"Promotion requirement: {self.wins_needed} consecutive wins")
        print(f"Max games: {max_games}")
        print("=" * 80)

        start_time = time.time()
        games_played = 0

        try:
            while games_played < max_games:
                ai_color = chess.WHITE if games_played % 2 == 0 else chess.BLACK
                color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"

                if verbose:
                    print(f"\n--- Game {games_played + 1} vs Stockfish Level {self.current_level} (AI={color_str}) ---")

                result, score, rounds = self.play_game_vs_stockfish(
                    ai_color, self.current_level, verbose
                )

                if result is None:  # Stockfish error
                    break

                # Update statistics
                games_played += 1
                self.total_games += 1
                self.games_by_level[self.current_level] += 1
                self.score_history.append(score)
                self.level_history.append(self.current_level)

                promoted = self.handle_game_result(result, score)

                # Progress report
                stats = self.level_stats[self.current_level]
                win_rate = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0

                if verbose or promoted or games_played % 10 == 0:
                    print(f"\n  Game {games_played}: {result.upper():6s} | Score: {score:+7.0f} | Rounds: {rounds:2d}")
                    print(f"  Level {self.current_level}: {stats['wins']}W-{stats['losses']}L-{stats['draws']}D "
                          f"| Win Rate: {win_rate:.1f}% | Streak: {self.consecutive_wins}/{self.wins_needed}")

                    if promoted:
                        print(f"\n  🎉 PROMOTED TO LEVEL {self.current_level}! 🎉")
                        print(f"  Conquered level {self.current_level - 1} with {self.wins_needed} consecutive wins!")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")

        elapsed = time.time() - start_time

        # Final report
        self.print_summary(elapsed)

    def print_summary(self, elapsed_time):
        """Print comprehensive training summary"""
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)

        print(f"\nTime: {elapsed_time:.1f}s ({elapsed_time/self.total_games:.2f}s per game)")
        print(f"Total Games: {self.total_games}")
        print(f"Current Level: {self.current_level} / {self.max_level}")
        print(f"Current Streak: {self.consecutive_wins}/{self.wins_needed}")

        print("\n" + "-" * 80)
        print("PERFORMANCE BY LEVEL")
        print("-" * 80)
        print(f"{'Level':<8} {'Games':<8} {'W-L-D':<12} {'Win%':<8} {'Avg Score':<12} {'Best':<8}")
        print("-" * 80)

        for level in sorted(self.level_stats.keys()):
            stats = self.level_stats[level]
            if stats['games'] == 0:
                continue

            win_rate = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            wld = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"

            print(f"{level:<8} {stats['games']:<8} {wld:<12} {win_rate:<7.1f}% "
                  f"{stats['avg_score']:<+11.0f} {stats['best_score']:<+8.0f}")

        # Learning statistics
        print("\n" + "-" * 80)
        print("LEARNING PROGRESS")
        print("-" * 80)

        learn_stats = self.prioritizer.get_statistics()
        print(f"Patterns Learned: {learn_stats['patterns_learned']}")
        print(f"Average Confidence: {learn_stats['avg_confidence']:.3f}")
        print(f"Average Win Rate: {learn_stats['avg_win_rate']:.1%}")

        print("\n" + "=" * 80)

    def export_results(self, filename=None):
        """Export training results to JSON"""
        if filename is None:
            filename = f"progressive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'summary': {
                'total_games': self.total_games,
                'current_level': self.current_level,
                'max_level': self.max_level,
                'consecutive_wins': self.consecutive_wins
            },
            'level_stats': self.level_stats,
            'score_history': self.score_history,
            'level_history': self.level_history,
            'games_by_level': {k: v for k, v in self.games_by_level.items() if v > 0}
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results exported to {filename}")

    def show_top_patterns(self, limit=10):
        """Display top learned patterns"""
        print("\n" + "=" * 80)
        print(f"TOP {limit} LEARNED PATTERNS")
        print("=" * 80)

        patterns = self.prioritizer.get_top_patterns(limit)

        if not patterns:
            print("No patterns learned yet!")
            return

        print(f"{'Piece':<10} {'Category':<14} {'Phase':<12} {'Seen':<6} {'Win%':<8} {'Conf':<6} {'Priority':<8}")
        print("-" * 80)

        for row in patterns:
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"{piece:<10} {category:<14} {phase:<12} {seen:<6} {win_rate:<7.1%} {conf:<6.2f} {priority:<8.1f}")

        print("=" * 80)

    def close(self):
        """Clean up resources"""
        self.prioritizer.close()


def main():
    """Run progressive trainer"""
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Chess AI Trainer')
    parser.add_argument('max_games', type=int, nargs='?', default=1000,
                       help='Maximum number of games (default: 1000)')
    parser.add_argument('--stockfish', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish binary')
    parser.add_argument('--start-level', type=int, default=0,
                       help='Starting Stockfish level (default: 0)')
    parser.add_argument('--wins-needed', type=int, default=10,
                       help='Consecutive wins needed to advance (default: 10)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed game output')
    parser.add_argument('--export', action='store_true',
                       help='Export results to JSON')
    parser.add_argument('--show-patterns', type=int, default=0, metavar='N',
                       help='Show top N patterns after training')

    args = parser.parse_args()

    trainer = ProgressiveTrainer(stockfish_path=args.stockfish)
    trainer.current_level = args.start_level
    trainer.wins_needed = args.wins_needed

    try:
        trainer.train(max_games=args.max_games, verbose=args.verbose)

        if args.export:
            trainer.export_results()

        if args.show_patterns > 0:
            trainer.show_top_patterns(args.show_patterns)

    finally:
        trainer.close()


if __name__ == '__main__':
    main()
