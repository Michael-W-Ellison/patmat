#!/usr/bin/env python3
"""
Simple Test Game - Demonstrates AI Making Moves

Shows the AI playing moves and using differential scoring to evaluate positions
"""

import chess
import random
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer

def simple_evaluate_position(board, ai_color, scorer):
    """Simple position evaluation using material advantage"""
    if board.is_checkmate():
        if board.turn != ai_color:
            return 10000  # AI wins
        else:
            return -10000  # AI loses

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Use differential scoring for evaluation
    ai_material = scorer._calculate_material(board, ai_color)
    opp_material = scorer._calculate_material(board, not ai_color)
    material_advantage = ai_material - opp_material

    return material_advantage


def get_best_move(board, ai_color, scorer, prioritizer, depth=2):
    """
    Simple minimax search using learned priorities and differential scoring
    """
    legal_moves = list(board.legal_moves)

    # Sort by learned priority
    legal_moves = prioritizer.sort_moves_by_priority(board, legal_moves)

    # Limit search to top moves based on learned priorities
    max_moves = min(15, len(legal_moves))
    legal_moves = legal_moves[:max_moves]

    best_move = None
    best_score = -999999

    for move in legal_moves:
        board.push(move)

        # Evaluate position after this move
        if depth <= 1 or board.is_game_over():
            score = simple_evaluate_position(board, ai_color, scorer)
        else:
            # Opponent's turn - they'll try to minimize our score
            opponent_moves = list(board.legal_moves)[:10]  # Limit opponent search
            worst_response = 999999

            for opp_move in opponent_moves:
                board.push(opp_move)
                response_score = simple_evaluate_position(board, ai_color, scorer)
                board.pop()

                if response_score < worst_response:
                    worst_response = response_score

            score = worst_response

        board.pop()

        # Track the best move
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score


def play_test_game():
    """Play a simple test game"""
    print("=" * 80)
    print("PATTERN RECOGNITION AI - TEST GAME")
    print("=" * 80)
    print("\nDemonstrating:")
    print("  ✓ Differential scoring (material advantage)")
    print("  ✓ Learnable move prioritization")
    print("  ✓ Pattern-based decision making")
    print()

    board = chess.Board()
    scorer = GameScorer()
    prioritizer = LearnableMovePrioritizer('test_game.db')

    ai_color = chess.WHITE
    game_moves = []
    move_number = 1

    print("Game Start - AI plays WHITE vs Random Opponent")
    print("=" * 80)
    print()

    while not board.is_game_over() and move_number <= 30:  # Limit to 30 moves

        if board.turn == ai_color:
            # AI's turn
            print(f"\n{'='*80}")
            print(f"Move {move_number} - AI (WHITE) to move")
            print(f"{'='*80}")

            # Show current material advantage
            ai_mat = scorer._calculate_material(board, ai_color)
            opp_mat = scorer._calculate_material(board, not ai_color)
            advantage = ai_mat - opp_mat
            print(f"Material advantage: {advantage:+.0f}")

            # Get best move using learned priorities
            fen_before = board.fen()
            best_move, expected_score = get_best_move(board, ai_color, scorer, prioritizer, depth=2)

            if best_move is None:
                print("No legal moves available!")
                break

            # Show move details
            move_chars = prioritizer.classify_move(board, best_move)
            priority = prioritizer.get_move_priority(board, best_move)

            # Get move notation before pushing
            move_san = board.san(best_move)

            print(f"\nAI chooses: {move_san}")
            print(f"  Piece:     {move_chars['piece_type']}")
            print(f"  Category:  {move_chars['move_category']}")
            print(f"  Priority:  {priority:.1f} (learned from past games)")
            print(f"  Expected:  {expected_score:+.0f} advantage")

            # Calculate exchange evaluation
            board.push(best_move)
            fen_after = board.fen()
            delta = scorer.calculate_material_delta(fen_before, fen_after, ai_color)

            if abs(delta) > 0:
                print(f"  Exchange:  {delta:+.0f} advantage change!")

            # Record move
            game_moves.append((fen_before, best_move.uci(), move_san))

            # Show board state
            print(f"\nPosition after move:")
            print(board)

        else:
            # Opponent's turn (random moves for demo)
            legal_moves = list(board.legal_moves)

            # 70% random move, 30% capture if available
            captures = [m for m in legal_moves if board.is_capture(m)]
            if captures and random.random() < 0.3:
                opp_move = random.choice(captures)
            else:
                opp_move = random.choice(legal_moves)

            print(f"\nMove {move_number} - Opponent (BLACK): {board.san(opp_move)}")
            board.push(opp_move)

            move_number += 1

    print("\n" + "=" * 80)
    print("GAME OVER")
    print("=" * 80)

    # Determine result
    if board.is_checkmate():
        if board.turn != ai_color:
            result = 'win'
            print("Result: AI WINS by checkmate! ✓")
        else:
            result = 'loss'
            print("Result: AI LOSES by checkmate")
    elif board.is_stalemate():
        result = 'draw'
        print("Result: DRAW by stalemate")
    elif board.is_insufficient_material():
        result = 'draw'
        print("Result: DRAW by insufficient material")
    else:
        result = 'unfinished'
        print("Result: Game ended (move limit or other)")

    # Calculate final score using differential scoring
    rounds_played = board.fullmove_number
    final_score, _ = scorer.calculate_final_score(board, ai_color, rounds_played)

    ai_mat = scorer._calculate_material(board, ai_color)
    opp_mat = scorer._calculate_material(board, not ai_color)
    advantage = ai_mat - opp_mat

    print(f"\nFinal Statistics:")
    print(f"  Moves played:      {move_number}")
    print(f"  Material advantage: {advantage:+.0f}")
    print(f"  DIFFERENTIAL SCORE: {final_score:.0f}")

    if result == 'win':
        time_bonus = max(0, 100 - rounds_played)
        print(f"\n  Score breakdown:")
        print(f"    Material advantage: {advantage:+.0f}")
        print(f"    Time bonus:         +{time_bonus}")
        print(f"    Win bonus:          +1000")
        print(f"    Total:              {final_score:.0f}")
    elif result == 'loss':
        print(f"\n  Score breakdown:")
        print(f"    Material advantage: {advantage:+.0f}")
        print(f"    Loss penalty:       -1000")
        print(f"    Total:              {final_score:.0f}")

    # Record game for learning
    print(f"\nRecording game for learning...")
    prioritizer.record_game_moves(game_moves, ai_color, result, final_score)
    print(f"✓ {len(game_moves)} moves recorded")
    print(f"✓ Patterns will be updated based on score: {final_score:.0f}")

    # Show what was learned
    stats = prioritizer.get_statistics()
    print(f"\nLearning Statistics:")
    print(f"  Patterns learned: {stats['patterns_learned']}")

    if stats['patterns_learned'] > 0:
        print(f"\n  Top 3 patterns from this session:")
        for row in prioritizer.get_top_patterns(3):
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"    {piece:8s} {category:12s}: priority={priority:.1f}, seen={seen}x")

    prioritizer.close()

    print("\n" + "=" * 80)
    print("The AI learns:")
    print("  ✓ Which move types led to this score")
    print("  ✓ Material advantage changes (exchanges)")
    print("  ✓ Patterns that improve position")
    print("  ✓ Adjusts priorities for future games")
    print("=" * 80)


if __name__ == '__main__':
    play_test_game()
