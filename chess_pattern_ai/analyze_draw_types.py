#!/usr/bin/env python3
"""
Analyze what types of draws are occurring in recent training games.
This will help us understand why the draw rate remains so high.
"""

import chess
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_pattern_ai.learnable_move_prioritizer import LearnableMovePrioritizer
from chess_pattern_ai.game_scorer import GameScorer

def simulate_games(num_games=100):
    """Simulate games and track draw types."""
    prioritizer = LearnableMovePrioritizer('chess_pattern_ai/headless_training.db')
    prioritizer.load_patterns()

    draw_types = {
        'stalemate': 0,
        'insufficient_material': 0,
        'seventyfive_moves': 0,
        'fivefold_repetition': 0,
        'fifty_moves': 0,
        'repetition': 0
    }

    avoidable_draws = {
        'stalemate_while_ahead': 0,
        'threefold_repetition': 0,
        'fifty_move_rule_while_ahead': 0
    }

    total_draws = 0
    total_wins = 0
    total_losses = 0

    for game_num in range(num_games):
        board = chess.Board()
        ai_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
        move_count = 0

        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            if board.turn == ai_color:
                # AI move - simplified version
                best_move = None
                best_score = float('-inf')

                for move in legal_moves[:15]:
                    board.push(move)

                    # Calculate material
                    ai_mat = sum([
                        len(board.pieces(chess.PAWN, ai_color)) * 100,
                        len(board.pieces(chess.KNIGHT, ai_color)) * 320,
                        len(board.pieces(chess.BISHOP, ai_color)) * 330,
                        len(board.pieces(chess.ROOK, ai_color)) * 500,
                        len(board.pieces(chess.QUEEN, ai_color)) * 900
                    ])

                    opp_color = chess.BLACK if ai_color == chess.WHITE else chess.WHITE
                    opp_mat = sum([
                        len(board.pieces(chess.PAWN, opp_color)) * 100,
                        len(board.pieces(chess.KNIGHT, opp_color)) * 320,
                        len(board.pieces(chess.BISHOP, opp_color)) * 330,
                        len(board.pieces(chess.ROOK, opp_color)) * 500,
                        len(board.pieces(chess.QUEEN, opp_color)) * 900
                    ])

                    material_score = ai_mat - opp_mat
                    board.pop()

                    priority = prioritizer.get_move_priority(board, move)
                    score = material_score + (priority * 20)

                    if score > best_score:
                        best_score = score
                        best_move = move

                if best_move:
                    board.push(best_move)
            else:
                # Opponent - random with capture bias
                import random
                capture_moves = [m for m in legal_moves if board.is_capture(m)]
                if capture_moves and random.random() < 0.3:
                    board.push(random.choice(capture_moves))
                else:
                    board.push(random.choice(legal_moves))

            move_count += 1

            # Prevent infinite games
            if move_count > 200:
                break

        # Analyze game result
        result = board.result()

        if result == '1/2-1/2':
            total_draws += 1

            # Calculate material advantage for AI
            ai_mat = sum([
                len(board.pieces(chess.PAWN, ai_color)) * 100,
                len(board.pieces(chess.KNIGHT, ai_color)) * 320,
                len(board.pieces(chess.BISHOP, ai_color)) * 330,
                len(board.pieces(chess.ROOK, ai_color)) * 500,
                len(board.pieces(chess.QUEEN, ai_color)) * 900
            ])

            opp_color = chess.BLACK if ai_color == chess.WHITE else chess.WHITE
            opp_mat = sum([
                len(board.pieces(chess.PAWN, opp_color)) * 100,
                len(board.pieces(chess.KNIGHT, opp_color)) * 320,
                len(board.pieces(chess.BISHOP, opp_color)) * 330,
                len(board.pieces(chess.ROOK, opp_color)) * 500,
                len(board.pieces(chess.QUEEN, opp_color)) * 900
            ])

            material_advantage = ai_mat - opp_mat

            # Identify draw type
            if board.is_stalemate():
                draw_types['stalemate'] += 1
                if material_advantage > 100:
                    avoidable_draws['stalemate_while_ahead'] += 1

            elif board.is_insufficient_material():
                draw_types['insufficient_material'] += 1

            elif board.is_seventyfive_moves():
                draw_types['seventyfive_moves'] += 1

            elif board.is_fivefold_repetition():
                draw_types['fivefold_repetition'] += 1

            elif board.can_claim_fifty_moves():
                draw_types['fifty_moves'] += 1
                if material_advantage > 200:
                    avoidable_draws['fifty_move_rule_while_ahead'] += 1

            elif board.is_repetition():
                draw_types['repetition'] += 1
                if material_advantage > -200:
                    avoidable_draws['threefold_repetition'] += 1

        elif (result == '1-0' and ai_color == chess.WHITE) or (result == '0-1' and ai_color == chess.BLACK):
            total_wins += 1
        else:
            total_losses += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"DRAW TYPE ANALYSIS ({num_games} games)")
    print(f"{'='*60}\n")

    print(f"Overall Results:")
    print(f"  Wins:   {total_wins:3d} ({total_wins/num_games*100:5.1f}%)")
    print(f"  Losses: {total_losses:3d} ({total_losses/num_games*100:5.1f}%)")
    print(f"  Draws:  {total_draws:3d} ({total_draws/num_games*100:5.1f}%)")
    print()

    if total_draws > 0:
        print(f"Draw Types:")
        for draw_type, count in draw_types.items():
            if count > 0:
                pct = count / total_draws * 100
                print(f"  {draw_type:25s}: {count:3d} ({pct:5.1f}% of draws)")
        print()

        print(f"Avoidable Draws (should be heavily penalized):")
        total_avoidable = sum(avoidable_draws.values())
        for draw_type, count in avoidable_draws.items():
            if count > 0:
                pct = count / total_draws * 100
                print(f"  {draw_type:35s}: {count:3d} ({pct:5.1f}% of draws)")
        print()
        print(f"Total Avoidable: {total_avoidable} ({total_avoidable/total_draws*100:.1f}% of draws)")
        print(f"Total Unavoidable: {total_draws - total_avoidable} ({(total_draws - total_avoidable)/total_draws*100:.1f}% of draws)")

if __name__ == '__main__':
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    simulate_games(num_games)
