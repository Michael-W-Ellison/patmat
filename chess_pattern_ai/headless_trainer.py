#!/usr/bin/env python3
"""
Headless Chess AI Trainer - No GUI Required

Provides similar functionality to the GUI but runs in terminal:
- Train multiple games
- Real-time metrics display
- Progress tracking
- Export results to JSON/CSV
"""

import chess
import json
import csv
import time
import random
from datetime import datetime
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer


class HeadlessTrainer:
    """Command-line chess AI trainer"""

    def __init__(self, db_path='headless_training.db'):
        self.scorer = GameScorer()
        self.prioritizer = LearnableMovePrioritizer(db_path)

        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []
        self.game_results = []

    def play_ai_move(self, board, ai_color):
        """Simple AI move selection"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        legal_moves = self.prioritizer.sort_moves_by_priority(board, legal_moves)

        # Simple evaluation
        best_move = None
        best_score = -999999

        for move in legal_moves[:15]:
            board.push(move)
            ai_mat = self.scorer._calculate_material(board, ai_color)
            opp_mat = self.scorer._calculate_material(board, not ai_color)
            score = ai_mat - opp_mat
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def play_opponent_move(self, board):
        """Random opponent with 30% capture bias"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]

        if captures and random.random() < 0.3:
            return random.choice(captures)

        return random.choice(legal_moves)

    def play_game(self, ai_color, verbose=False):
        """Play a complete game"""
        board = chess.Board()
        game_moves = []
        move_count = 0

        while not board.is_game_over() and move_count < 60:
            fen_before = board.fen()

            if board.turn == ai_color:
                move = self.play_ai_move(board, ai_color)
                if move:
                    move_san = board.san(move)
                    board.push(move)
                    game_moves.append((fen_before, move.uci(), move_san))

                    if verbose:
                        ai_mat = self.scorer._calculate_material(board, ai_color)
                        opp_mat = self.scorer._calculate_material(board, not ai_color)
                        advantage = ai_mat - opp_mat
                        print(f"  AI: {move_san:8s} | Advantage: {advantage:+6.0f}")
            else:
                move = self.play_opponent_move(board)
                if move:
                    board.push(move)
                    move_count += 1
                    if verbose:
                        print(f"  Opp: {board.san(move):8s}")

        # Calculate final score
        rounds_played = board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)

        # Update statistics
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        self.score_history.append(final_score)

        # Record for learning
        self.prioritizer.record_game_moves(game_moves, ai_color, result, final_score)

        # Store result details
        self.game_results.append({
            'game_num': self.games_played,
            'result': result,
            'score': final_score,
            'rounds': rounds_played,
            'ai_color': 'white' if ai_color == chess.WHITE else 'black'
        })

        return result, final_score, rounds_played

    def train(self, num_games, verbose=False, progress_interval=10):
        """Train for multiple games"""
        print("=" * 70)
        print(f"HEADLESS TRAINING - {num_games} GAMES")
        print("=" * 70)

        start_time = time.time()

        for i in range(num_games):
            ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"

            if verbose or (i + 1) % progress_interval == 0:
                print(f"\n--- Game {i+1}/{num_games} (AI plays {color_str}) ---")

            result, score, rounds = self.play_game(ai_color, verbose=verbose)

            if (i + 1) % progress_interval == 0 or verbose:
                win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0
                avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
                print(f"  Result: {result.upper():6s} | Score: {score:+7.0f} | Rounds: {rounds:2d}")
                print(f"  Progress: {self.wins}W-{self.losses}L-{self.draws}D | Win Rate: {win_rate:.1f}% | Avg Score: {avg_score:+.0f}")

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
        print(f"Results: {self.wins}W-{self.losses}L-{self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played*100):.1f}%")
        print(f"Average Score: {sum(self.score_history)/len(self.score_history):+.0f}")
        print(f"Best Score: {max(self.score_history):+.0f}")
        print(f"Worst Score: {min(self.score_history):+.0f}")

        # Learning statistics
        stats = self.prioritizer.get_statistics()
        print(f"\nLearning Statistics:")
        print(f"  Patterns Learned: {stats['patterns_learned']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg Win Rate: {stats['avg_win_rate']:.1%}")

        print("\n" + "=" * 70)

    def export_results(self, filename=None):
        """Export results to JSON"""
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'summary': {
                'games_played': self.games_played,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'win_rate': self.wins / self.games_played if self.games_played > 0 else 0,
                'avg_score': sum(self.score_history) / len(self.score_history) if self.score_history else 0
            },
            'score_history': self.score_history,
            'games': self.game_results
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results exported to {filename}")

    def export_csv(self, filename=None):
        """Export results to CSV"""
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['game_num', 'result', 'score', 'rounds', 'ai_color'])
            writer.writeheader()
            writer.writerows(self.game_results)

        print(f"✓ Results exported to {filename}")

    def show_top_patterns(self, limit=10):
        """Display top learned patterns"""
        print("\n" + "=" * 70)
        print(f"TOP {limit} LEARNED PATTERNS (by priority)")
        print("=" * 70)

        patterns = self.prioritizer.get_top_patterns(limit)

        if not patterns:
            print("No patterns learned yet!")
            return

        print(f"{'Piece':<10} {'Category':<14} {'Phase':<12} {'Seen':<6} {'Win%':<8} {'Conf':<6} {'Priority':<8}")
        print("-" * 70)

        for row in patterns:
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"{piece:<10} {category:<14} {phase:<12} {seen:<6} {win_rate:<7.1%} {conf:<6.2f} {priority:<8.1f}")

        print("=" * 70)

    def close(self):
        """Clean up resources"""
        self.prioritizer.close()


def main():
    """Run headless trainer"""
    import argparse

    parser = argparse.ArgumentParser(description='Chess AI Headless Trainer')
    parser.add_argument('num_games', type=int, help='Number of games to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=10, help='Progress update interval (default: 10)')
    parser.add_argument('--export-json', action='store_true', help='Export results to JSON')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--show-patterns', type=int, default=0, metavar='N', help='Show top N patterns after training')

    args = parser.parse_args()

    trainer = HeadlessTrainer()

    try:
        trainer.train(args.num_games, verbose=args.verbose, progress_interval=args.progress)

        if args.export_json:
            trainer.export_results()

        if args.export_csv:
            trainer.export_csv()

        if args.show_patterns > 0:
            trainer.show_top_patterns(args.show_patterns)

    finally:
        trainer.close()


if __name__ == '__main__':
    main()
