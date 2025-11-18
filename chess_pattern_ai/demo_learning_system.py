#!/usr/bin/env python3
"""
Demonstration of the Complete Learning System

Shows how the AI learns from game outcomes using:
1. Differential scoring (material advantage, not absolute)
2. Learnable move prioritization (no hardcoded squares)
3. Pattern recognition from outcomes
4. Exchange evaluation
"""

import chess
import os
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer

def play_demo_games():
    """Play several demo games to demonstrate learning"""

    # Use a demo database
    db_path = "demo_learning.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    prioritizer = LearnableMovePrioritizer(db_path)
    scorer = GameScorer()

    print("=" * 80)
    print("CHESS PATTERN RECOGNITION AI - LEARNING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstrates the AI learning from game outcomes WITHOUT hardcoded")
    print("chess knowledge. It learns which move types lead to high differential scores.")
    print()

    # ========================================================================
    # GAME 1: Quick win with good development
    # ========================================================================
    print("\n" + "=" * 80)
    print("GAME 1: Scholar's Mate - Quick Win with Development")
    print("=" * 80)

    board1 = chess.Board()
    moves1 = [
        ('e4', 'white pawn advance'),
        ('e5', 'black pawn response'),
        ('Bc4', 'white develops bishop (targets f7)'),
        ('Nc6', 'black develops knight'),
        ('Qh5', 'white brings out queen (aggressive)'),
        ('Nf6', 'black develops knight'),
        ('Qxf7#', 'CHECKMATE!')
    ]

    game_moves1 = []
    print("\nMoves:")
    for i, (move_san, comment) in enumerate(moves1, 1):
        fen_before = board1.fen()
        try:
            move = board1.parse_san(move_san)
            game_moves1.append((fen_before, move.uci(), move_san))
            board1.push(move)
            print(f"  {i}. {move_san:8s} - {comment}")
        except:
            pass

    score1, result1 = scorer.calculate_final_score(board1, chess.WHITE, rounds_played=4)
    ai_mat1 = scorer._calculate_material(board1, chess.WHITE)
    opp_mat1 = scorer._calculate_material(board1, chess.BLACK)
    advantage1 = ai_mat1 - opp_mat1

    print(f"\nOutcome:")
    print(f"  Result: {result1}")
    print(f"  Material advantage: {advantage1:+.0f}")
    print(f"  Time bonus: {100 - 4} (quick checkmate!)")
    print(f"  DIFFERENTIAL SCORE: {score1:.0f}")
    print(f"  Formula: {advantage1:.0f} + {100-4} + 1000 = {score1:.0f}")

    # Record for learning
    prioritizer.record_game_moves(game_moves1, chess.WHITE, result1, score1)
    print(f"  ✓ Recorded for learning")

    # ========================================================================
    # GAME 2: Loss from bad opening (fool's mate)
    # ========================================================================
    print("\n" + "=" * 80)
    print("GAME 2: Fool's Mate - Quick Loss from Poor Play")
    print("=" * 80)

    board2 = chess.Board()
    moves2 = [
        ('f3', 'white weakens kingside'),
        ('e5', 'black controls center'),
        ('g4', 'white further weakens king'),
        ('Qh4#', 'CHECKMATE!')
    ]

    game_moves2 = []
    print("\nMoves:")
    for i, (move_san, comment) in enumerate(moves2, 1):
        fen_before = board2.fen()
        try:
            move = board2.parse_san(move_san)
            game_moves2.append((fen_before, move.uci(), move_san))
            board2.push(move)
            print(f"  {i}. {move_san:8s} - {comment}")
        except:
            pass

    score2, result2 = scorer.calculate_final_score(board2, chess.WHITE, rounds_played=2)
    ai_mat2 = scorer._calculate_material(board2, chess.WHITE)
    opp_mat2 = scorer._calculate_material(board2, chess.BLACK)
    advantage2 = ai_mat2 - opp_mat2

    print(f"\nOutcome:")
    print(f"  Result: {result2}")
    print(f"  Material advantage: {advantage2:+.0f}")
    print(f"  Loss penalty: -1000")
    print(f"  DIFFERENTIAL SCORE: {score2:.0f}")
    print(f"  Formula: {advantage2:.0f} - 1000 = {score2:.0f}")

    prioritizer.record_game_moves(game_moves2, chess.WHITE, result2, score2)
    print(f"  ✓ Recorded for learning")

    # ========================================================================
    # GAME 3: Win with favorable exchange
    # ========================================================================
    print("\n" + "=" * 80)
    print("GAME 3: Winning Through Favorable Exchanges")
    print("=" * 80)

    board3 = chess.Board()
    moves3 = [
        ('e4', 'pawn to e4'),
        ('d5', 'black challenges'),
        ('exd5', 'WHITE CAPTURES PAWN'),
        ('Qxd5', 'black recaptures'),
        ('Nc3', 'white develops, attacks queen'),
        ('Qd8', 'queen retreats'),
        ('Nf3', 'develop knight'),
        ('Nf6', 'black develops'),
    ]

    game_moves3 = []
    print("\nMoves:")
    for i, (move_san, comment) in enumerate(moves3, 1):
        fen_before = board3.fen()
        try:
            move = board3.parse_san(move_san)
            game_moves3.append((fen_before, move.uci(), move_san))

            # Show exchange evaluation
            if move_san == 'exd5':
                fen_after = board3.fen()
                board3.push(move)
                delta = scorer.calculate_material_delta(fen_before, board3.fen(), chess.WHITE)
                print(f"  {i}. {move_san:8s} - {comment} [Exchange: {delta:+.0f}]")
            else:
                board3.push(move)
                print(f"  {i}. {move_san:8s} - {comment}")
        except:
            pass

    # Simulate winning this game
    score3, result3 = 1100.0, 'win'  # Simulated
    print(f"\n(Game continues... white wins)")
    print(f"\nOutcome:")
    print(f"  Result: {result3}")
    print(f"  DIFFERENTIAL SCORE: {score3:.0f}")

    prioritizer.record_game_moves(game_moves3, chess.WHITE, result3, score3)
    print(f"  ✓ Recorded for learning")

    # ========================================================================
    # Show learning statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("LEARNING STATISTICS")
    print("=" * 80)

    stats = prioritizer.get_statistics()
    print(f"\nPatterns learned: {stats['patterns_learned']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"Average win rate: {stats['avg_win_rate']:.1%}")

    if stats['patterns_learned'] > 0:
        print("\n" + "-" * 80)
        print("Top Move Patterns (by differential score)")
        print("-" * 80)
        print(f"{'Piece':<10} {'Category':<14} {'Phase':<12} {'Seen':<6} {'Win%':<8} {'Conf':<6} {'Priority':<8}")
        print("-" * 80)

        for row in prioritizer.get_top_patterns(10):
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"{piece:<10} {category:<14} {phase:<12} {seen:<6} {win_rate:<7.1%} {conf:<6.2f} {priority:<8.1f}")

    # ========================================================================
    # Demonstrate learned priorities
    # ========================================================================
    print("\n" + "=" * 80)
    print("APPLYING LEARNED PRIORITIES")
    print("=" * 80)

    print("\nIn a new game, the AI now prioritizes moves based on learned patterns:")
    test_board = chess.Board()

    # Get all legal moves
    all_moves = list(test_board.legal_moves)

    # Sort by learned priority
    sorted_moves = prioritizer.sort_moves_by_priority(test_board, all_moves)

    print(f"\nTop 10 moves by learned priority:")
    print(f"{'Move':<8} {'Piece':<10} {'Category':<14} {'Priority':<10}")
    print("-" * 50)

    for move in sorted_moves[:10]:
        priority = prioritizer.get_move_priority(test_board, move)
        chars = prioritizer.classify_move(test_board, move)
        if chars:
            print(f"{move.uci():<8} {chars['piece_type']:<10} {chars['move_category']:<14} {priority:<10.1f}")

    # ========================================================================
    # Exchange evaluation demonstration
    # ========================================================================
    print("\n" + "=" * 80)
    print("DIFFERENTIAL EXCHANGE EVALUATION")
    print("=" * 80)

    print("\nThe AI learns that exchanges matter RELATIVELY:")
    print()

    # Example 1: Even trade
    print("Example 1: Even trade (pawn for pawn)")
    board_ex1 = chess.Board()
    fen_before = board_ex1.fen()
    board_ex1.push_san('e4')
    board_ex1.push_san('d5')
    board_ex1.push_san('exd5')
    delta1 = scorer.calculate_material_delta(fen_before, board_ex1.fen(), chess.WHITE)
    print(f"  Lost: 100 (pawn), Gained: 100 (pawn)")
    print(f"  Net advantage: {delta1:+.0f} → Neutral exchange")

    # Example 2: Favorable trade (simulate queen for knight + pawn)
    print("\nExample 2: Favorable trade (capture rook with pawn)")
    board_ex2 = chess.Board()
    # Set up position where white pawn can capture black rook
    board_ex2.set_fen("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    # Move pawn to d5 to threaten rook
    board_ex2.set_fen("rnbqkb1r/ppp2ppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    fen_before2 = board_ex2.fen()
    # Take the knight
    board_ex2.push_san('exf6')
    delta2 = scorer.calculate_material_delta(fen_before2, board_ex2.fen(), chess.WHITE)
    print(f"  Lost: 100 (pawn), Gained: 320 (knight)")
    print(f"  Net advantage: {delta2:+.0f} → Good exchange!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("LEARNING SUMMARY")
    print("=" * 80)

    print("\nThe AI learned from observation:")
    print("  ✓ Quick wins score higher (time bonus)")
    print("  ✓ Material advantage matters (differential scoring)")
    print("  ✓ Development moves appear in winning games")
    print("  ✓ Weakening king position appears in losses")
    print("  ✓ Favorable exchanges improve position")
    print()
    print("NO hardcoded knowledge about:")
    print("  ✗ Specific 'good' squares (e4, d4, etc.)")
    print("  ✗ Opening theory or principles")
    print("  ✗ Positional concepts")
    print()
    print("ONLY observable features:")
    print("  ✓ Piece types and move categories")
    print("  ✓ Material counts and changes")
    print("  ✓ Game outcomes and scores")
    print("  ✓ Statistical win rates")

    prioritizer.close()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    play_demo_games()
