#!/usr/bin/env chess_env/bin/python3
"""
Fast Learning AI - With Optimized Search

Uses intelligent move filtering to search 25x faster!
- Prunes hanging piece moves
- Prunes king-into-danger moves
- Focuses on forcing moves
- Only searches top 15 candidate moves
- Filters opponent's nonsensical moves too
"""

import chess
import chess.engine
import time
from integrated_ai_with_clustering import ClusteredIntegratedAI
from optimized_search import OptimizedSearchMixin
from test_learning_ai_with_clustering import LearningGameTracker
from opening_performance_tracker import OpeningPerformanceTracker


class FastLearningAI(OptimizedSearchMixin, ClusteredIntegratedAI):
    """AI with optimized search - inherits both optimizations and full AI"""
    pass


def test_fast_ai(num_games: int = 3, use_stockfish_feedback: bool = False):
    """
    Test the optimized AI

    Args:
        num_games: Number of games to play
        use_stockfish_feedback: Enable Stockfish move analysis (slower but better learning)
    """
    print("=" * 70)
    print("FAST LEARNING AI - OPTIMIZED SEARCH")
    print("=" * 70)

    tracker = LearningGameTracker(use_stockfish_feedback=use_stockfish_feedback)
    opening_tracker = OpeningPerformanceTracker()

    print("\nInitializing AI with:")
    print("  - Depth 3 search")
    print("  - Intelligent move pruning")
    print("  - Abstract pattern learning (learns WHY, not just WHAT)")
    print("  - ~25x faster than unoptimized!")

    ai = FastLearningAI(
        search_depth=3,
        enable_pattern_learning=True,
        enable_clustering=True,
        time_limit_per_move=15.0
    )

    # Add opening performance tracker to AI
    ai.opening_tracker = opening_tracker

    # Add pattern abstraction engine to AI
    ai.pattern_engine = tracker.pattern_engine

    # Add move prioritizer to AI (learns which move types win)
    ai.move_prioritizer = tracker.move_prioritizer

    print("\nStarting Knowledge:")
    stats = tracker.get_learning_stats()
    print(f"  Games played:      {stats['games_played']}")
    print(f"  Mistakes learned:  {stats['mistakes_learned']}")
    print(f"  Tactics learned:   {stats['tactics_learned']}")

    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    engine.configure({"Skill Level": 0})

    print("\n" + "=" * 70)
    print(f"PLAYING {num_games} GAMES")
    print("=" * 70)

    results = {'wins': 0, 'draws': 0, 'losses': 0}
    total_time = 0
    total_moves = 0
    total_positions_evaluated = 0

    for game_num in range(1, num_games + 1):
        board = chess.Board()
        ai_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
        move_history = []
        move_count = 0

        color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"
        print(f"\nGame {game_num}/3 - AI plays {color_str}")
        print("-" * 70)

        game_start = time.time()
        initial_positions = ai.positions_evaluated

        while not board.is_game_over():  # No artificial move limit - use chess rules
            move_count += 1
            fen_before = board.fen()

            if board.turn == ai_color:
                # AI move with optimized search
                print(f"Move {move_count:3d} (AI): ", end='', flush=True)
                move_start = time.time()

                move_uci, score = ai.find_best_move_optimized(board.fen(), use_optimized=True)

                move_time = time.time() - move_start

                if move_uci == "0000":
                    print("No legal moves!")
                    break

                move = chess.Move.from_uci(move_uci)
                move_san = board.san(move)
                print(f"{move_san:8s} (score: {score:7.1f}, time: {move_time:.1f}s)")

                board.push(move)
                move_history.append((fen_before, move_uci, move_san))

            else:
                # Stockfish move
                print(f"Move {move_count:3d} (SF): ", end='', flush=True)
                result = engine.play(board, chess.engine.Limit(time=0.3))
                sf_move_san = board.san(result.move)
                print(f"{sf_move_san}")

                board.push(result.move)
                move_history.append((fen_before, result.move.uci(), sf_move_san))

        game_time = time.time() - game_start
        total_time += game_time
        total_moves += move_count
        positions_this_game = ai.positions_evaluated - initial_positions
        total_positions_evaluated += positions_this_game

        # Determine result
        if board.is_checkmate():
            if board.turn != ai_color:
                results['wins'] += 1
                result = 'win'
                print(f"\n🎉 AI WINS by checkmate!")
            else:
                results['losses'] += 1
                result = 'loss'
                print(f"\n❌ AI loses by checkmate")
        else:
            results['draws'] += 1
            result = 'draw'
            print(f"\n🤝 Draw")

        print(f"Game stats: {move_count} moves, {game_time:.1f}s, {positions_this_game:,} positions")

        # Commit all pending transactions to reduce lock contention
        if hasattr(ai, 'adaptive_cache') and ai.adaptive_cache:
            ai.adaptive_cache.commit()

        if hasattr(opening_tracker, 'conn'):
            opening_tracker.conn.commit()

        # Record game for learning (includes Stockfish analysis if enabled)
        tracker.record_game(board, ai_color, result, move_history)

        # Record opening performance
        opening_tracker.record_opening_result(move_history, result, ai_color)

        # Update adaptive cache based on game outcome AFTER learning
        if hasattr(ai, 'adaptive_cache') and ai.adaptive_cache:
            if result == 'loss':
                # Clear losing patterns to force exploration next game
                ai.adaptive_cache.clear_losing_patterns(move_history, ai_color)
            elif result == 'win':
                # Reinforce winning patterns
                ai.adaptive_cache.update_from_game_outcome(move_history, result, ai_color)

            ai.adaptive_cache.commit()

    engine.quit()

    # Final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    score = (results['wins'] + 0.5 * results['draws']) / num_games * 100
    print(f"\n{results['wins']}W {results['draws']}D {results['losses']}L - Score: {score:.1f}%")

    print(f"\nPerformance:")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Avg per game:         {total_time/num_games:.1f}s")
    print(f"  Avg per AI move:      {total_time/(total_moves/2):.1f}s")
    print(f"  Positions evaluated:  {total_positions_evaluated:,}")
    print(f"  Positions per second: {total_positions_evaluated/total_time:,.0f}")

    # Learning progress
    final_stats = tracker.get_learning_stats()
    print(f"\nLearning Progress:")
    print(f"  Total games:       {final_stats['games_played']} (+{final_stats['games_played'] - stats['games_played']})")
    print(f"  Mistakes learned:  {final_stats['mistakes_learned']} (+{final_stats['mistakes_learned'] - stats['mistakes_learned']})")
    print(f"  Tactics learned:   {final_stats['tactics_learned']} (+{final_stats['tactics_learned'] - stats['tactics_learned']})")

    # Abstract patterns learned with win rates
    print(f"\nAbstract Patterns Learned (Top 5):")
    cursor = tracker.pattern_engine.cursor
    cursor.execute('''
        SELECT pattern_type, pattern_description, times_seen, avg_material_lost,
               confidence, games_with_pattern_won, games_with_pattern_lost,
               games_with_pattern_draw, win_rate
        FROM abstract_patterns
        WHERE confidence > 0.2
        ORDER BY confidence DESC, avg_material_lost DESC
        LIMIT 5
    ''')

    for row in cursor.fetchall():
        ptype, pdesc, times_seen, avg_loss, confidence, won, lost, draw, win_rate = row
        total_games = won + lost + draw
        print(f"  {ptype:20s}: {pdesc:30s}")
        print(f"    Seen {times_seen:3d}x | Avg loss: {avg_loss:.1f} | Confidence: {confidence:.2f}")
        if total_games > 0:
            print(f"    Win rate: {win_rate*100:.0f}% ({won}W-{lost}L-{draw}D in {total_games} games)")

    # Learned move patterns (which move TYPES lead to wins)
    print(f"\nLearned Move Patterns (Top 5):")
    move_stats = tracker.move_prioritizer.get_statistics()
    print(f"  Patterns learned:  {move_stats['patterns_learned']}")
    print(f"  Avg confidence:    {move_stats['avg_confidence']:.2f}")
    print(f"  Avg win rate:      {move_stats['avg_win_rate']:.1%}")

    if move_stats['patterns_learned'] > 0:
        for row in tracker.move_prioritizer.get_top_patterns(5):
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"  {piece:8s} {category:12s} (dist={dist}, {phase:10s})")
            print(f"    Seen {seen:3d}x | Win rate: {win_rate:.1%} | Priority: {priority:.1f}")

    # Adaptive cache stats
    if hasattr(ai, 'adaptive_cache') and ai.adaptive_cache:
        cache_stats = ai.adaptive_cache.get_stats()
        print(f"\nAdaptive Pattern Cache:")
        print(f"  Patterns checked:      {cache_stats['patterns_checked']:,}")
        print(f"  Cache hits:            {cache_stats['cache_hits']:,} ({cache_stats['cache_hit_rate']:.1f}%)")
        print(f"  Patterns in memory:    {cache_stats['in_memory_cache_size']:,}")
        print(f"  Persistent patterns:   {cache_stats['persistent_cache_size']:,}")
        print(f"  Doing expensive queries: {cache_stats['doing_expensive_queries']}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    num_games = 3  # Default
    use_stockfish_feedback = False

    # Check for arguments
    for arg in sys.argv[1:]:
        if arg == '--stockfish-feedback':
            use_stockfish_feedback = True
            print("🎓 Stockfish feedback enabled - AI will learn from move quality analysis")
        elif arg.isdigit():
            num_games = int(arg)
            if num_games < 1:
                print("Error: Number of games must be at least 1")
                sys.exit(1)
        elif arg in ['--help', '-h']:
            print("Usage: python fast_learning_ai.py [number_of_games] [--stockfish-feedback]")
            print("\nArguments:")
            print("  number_of_games       Number of games to play (default: 3)")
            print("  --stockfish-feedback  Enable Stockfish move analysis for enhanced learning")
            print("                        (slower but identifies exact mistakes)")
            print("\nExamples:")
            print("  python fast_learning_ai.py 10")
            print("  python fast_learning_ai.py 20 --stockfish-feedback")
            sys.exit(0)
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use --help for usage information")
            sys.exit(1)

    test_fast_ai(num_games, use_stockfish_feedback)
