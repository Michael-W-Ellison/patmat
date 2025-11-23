#!/usr/bin/env python3
"""
Enhanced Headless Chess AI Trainer - Quick Integration Version

Maintains full backward compatibility while adding enhanced pattern learning.
"""

import chess
import json
import time
import random
from datetime import datetime
from game_scorer import GameScorer

# Try to import enhanced system, fall back gracefully
try:
    from enhanced_learnable_move_prioritizer import EnhancedLearnableMovePrioritizer
    ENHANCED_AVAILABLE = True
except ImportError:
    try:
        from learnable_move_prioritizer import LearnableMovePrioritizer
        ENHANCED_AVAILABLE = False
    except ImportError:
        print("❌ Could not import move prioritizer!")
        exit(1)

class CompatibleHeadlessTrainer:
    """Enhanced trainer with backward compatibility"""

    def __init__(self, db_path='enhanced_training.db', enhanced_mode=False):
        self.scorer = GameScorer()
        self.enhanced_mode = enhanced_mode and ENHANCED_AVAILABLE
        
        # Initialize prioritizer
        if ENHANCED_AVAILABLE:
            self.prioritizer = EnhancedLearnableMovePrioritizer(db_path)
            if not self.enhanced_mode:
                self.prioritizer.enhanced_mode = False
            print(f"🧠 Enhanced Pattern Learning: {'ENABLED' if self.enhanced_mode else 'DISABLED'}")
        else:
            self.prioritizer = LearnableMovePrioritizer(db_path)
            print("📊 Using Original Pattern Learning")

        # Statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []
        self.insights_learned = []

        # Draw analysis
        self.draw_types = {
            'stalemate': 0, 'insufficient_material': 0, 'fifty_moves': 0, 
            'repetition': 0, 'avoidable': 0, 'unavoidable': 0
        }

    def play_ai_move(self, board, ai_color):
        """AI move selection"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        legal_moves = self.prioritizer.sort_moves_by_priority(board, legal_moves)

        best_move = None
        best_score = -999999

        for move in legal_moves[:15]:
            # Material evaluation
            board.push(move)
            ai_mat = self.scorer._calculate_material(board, ai_color)
            opp_mat = self.scorer._calculate_material(board, not ai_color)
            material_score = ai_mat - opp_mat
            board.pop()

            # Pattern priority
            priority = self.prioritizer.get_move_priority(board, move)
            score = material_score + (priority * 20)

            # Enhanced safety checks if available
            if self.enhanced_mode and hasattr(self.prioritizer, '_extract_enhanced_context'):
                try:
                    context = self.prioritizer._extract_enhanced_context(board, move)
                    # Simple safety penalty
                    if not context.piece_defended and context.target_square_attacks > 0:
                        score -= 50  # Penalty for undefended moves to attacked squares
                    if context.king_safety_level == 'exposed' and context.forcing_move:
                        score -= 25  # Penalty for aggressive moves with exposed king
                except:
                    pass  # Fall back to basic evaluation

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def play_opponent_move(self, board):
        """Random opponent move"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]

        if captures and random.random() < 0.3:
            return random.choice(captures)
        return random.choice(legal_moves)

    def play_game(self, ai_color, verbose=False):
        """Play a single game"""
        board = chess.Board()
        game_moves = []
        move_count = 0

        while not board.is_game_over():
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
                        mode_icon = "🧠" if self.enhanced_mode else "📊"
                        print(f"  {mode_icon} AI: {move_san:8s} | Advantage: {advantage:+6.0f}")
            else:
                move = self.play_opponent_move(board)
                if move:
                    move_san = board.san(move)
                    board.push(move)
                    move_count += 1
                    if verbose:
                        print(f"  Opp: {move_san:8s}")

        # Calculate result
        rounds_played = board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)

        # Simple learning enhancement
        if self.enhanced_mode and final_score < -500:
            self.insights_learned.append("Learned from poor performance")

        # Update statistics
        self._update_statistics(result, final_score, board)

        # Record for learning
        self.prioritizer.record_game_moves(game_moves, ai_color, result, final_score)

        return result, final_score, rounds_played

    def _update_statistics(self, result, final_score, board):
        """Update game statistics"""
        
        # Draw analysis
        if result == 'draw':
            ai_mat = self.scorer._calculate_material(board, True)
            opp_mat = self.scorer._calculate_material(board, False)
            material_advantage = ai_mat - opp_mat

            if board.is_stalemate():
                self.draw_types['stalemate'] += 1
                self.draw_types['avoidable'] += 1 if material_advantage > 10 else 0
                self.draw_types['unavoidable'] += 1 if material_advantage <= 10 else 0
            elif board.is_insufficient_material():
                self.draw_types['insufficient_material'] += 1
                self.draw_types['unavoidable'] += 1
            elif board.is_fifty_moves():
                self.draw_types['fifty_moves'] += 1
            elif board.is_repetition():
                self.draw_types['repetition'] += 1

        # Game counters
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        self.score_history.append(final_score)

    def train(self, num_games, verbose=False, progress_interval=10):
        """Train the AI"""
        
        mode_name = "ENHANCED" if self.enhanced_mode else "ORIGINAL"
        print("=" * 70)
        print(f"{mode_name} PATTERN LEARNING TRAINING - {num_games} GAMES")
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
        self._show_training_results(elapsed, num_games, mode_name)

    def _show_training_results(self, elapsed, num_games, mode_name):
        """Show training results"""
        
        print("\n" + "=" * 70)
        print(f"{mode_name} TRAINING COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
        print(f"Results: {self.wins}W-{self.losses}L-{self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played*100):.1f}%")
        print(f"Average Score: {sum(self.score_history)/len(self.score_history):+.0f}")
        print(f"Best Score: {max(self.score_history):+.0f}")
        print(f"Worst Score: {min(self.score_history):+.0f}")

        # Draw analysis
        if self.draws > 0:
            print(f"\nDraw Analysis ({self.draws} total draws):")
            for draw_type, count in self.draw_types.items():
                if count > 0:
                    pct = count/self.draws*100
                    print(f"  {draw_type.replace('_', ' ').title():20s}: {count:3d} ({pct:5.1f}%)")

        # Enhanced learning stats
        if self.enhanced_mode and self.insights_learned:
            print(f"\n🧠 Enhanced Learning:")
            print(f"  Insights learned: {len(self.insights_learned)}")

        # Pattern statistics
        stats = self.prioritizer.get_statistics()
        print(f"\nPattern Statistics:")
        print(f"  Patterns Learned: {stats['patterns_learned']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg Win Rate: {stats['avg_win_rate']:.1%}")

        print("\n" + "=" * 70)

    def close(self):
        """Close resources"""
        self.prioritizer.close()

def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Chess AI Trainer')
    parser.add_argument('num_games', type=int, help='Number of games to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--progress', type=int, default=10, help='Progress interval')
    parser.add_argument('--enhanced', action='store_true', help='Enable enhanced mode')
    parser.add_argument('--db', default='enhanced_training.db', help='Database file')

    args = parser.parse_args()

    trainer = CompatibleHeadlessTrainer(db_path=args.db, enhanced_mode=args.enhanced)

    try:
        trainer.train(args.num_games, verbose=args.verbose, progress_interval=args.progress)
    finally:
        trainer.close()

if __name__ == '__main__':
    main()
