#!/usr/bin/env chess_env/bin/python3
"""
Integrated Chess AI + Indexed Database + Position Clustering
Combines the full pattern learning AI with the new clustering infrastructure
"""

import chess
import chess.engine
import sqlite3
import logging
import sys
import time
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# Import all the existing AI components
from integrated_chess_ai import IntegratedChessAI

# Import clustering
from integrate_clustering import PositionClusteringIntegrator

# Import enhanced pattern matching
from enhanced_pattern_matching import EnhancedPatternMatcher

# Import adaptive caching
from adaptive_pattern_cache import AdaptivePatternCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusteredIntegratedAI(IntegratedChessAI):
    """
    Enhanced Integrated AI with Position Clustering
    Extends the full AI with cluster-based pattern matching
    """

    def __init__(self, db_path: str = "rule_discovery.db",
                 search_depth: int = 1,
                 enable_opponent_prediction: bool = False,
                 enable_pattern_learning: bool = True,
                 enable_clustering: bool = True,
                 time_limit_per_move: float = 10.0):
        """
        Initialize with clustering support
        """
        # Initialize base AI
        super().__init__(
            db_path=db_path,
            search_depth=search_depth,
            enable_opponent_prediction=enable_opponent_prediction,
            enable_pattern_learning=enable_pattern_learning,
            time_limit_per_move=time_limit_per_move
        )

        # Add clustering
        self.enable_clustering = enable_clustering
        self.clustering = None
        self.cluster_stats = {
            'lookups': 0,
            'matches': 0,
            'avg_time': 0
        }

        if enable_clustering:
            logger.info("\n6. Loading Position Clustering...")
            try:
                self.clustering = PositionClusteringIntegrator(db_path)
                self.clustering._connect()
                self.clustering._load_clusters()
                logger.info("   ✅ Clustering enabled")
            except Exception as e:
                logger.warning(f"   ⚠️  Clustering failed: {e}")
                self.enable_clustering = False

        # Add enhanced pattern matcher
        logger.info("\n7. Loading Enhanced Pattern Matcher...")
        try:
            self.pattern_matcher = EnhancedPatternMatcher(db_path)
            logger.info("   ✅ Enhanced pattern matching enabled")
            logger.info("   📊 Can now leverage ALL 19.5M tactics in database!")
        except Exception as e:
            logger.warning(f"   ⚠️  Pattern matcher failed: {e}")
            self.pattern_matcher = None

        # Add adaptive cache
        logger.info("\n8. Loading Adaptive Pattern Cache...")
        try:
            self.adaptive_cache = AdaptivePatternCache(db_path)
            logger.info("   ✅ Adaptive caching enabled")
            logger.info("   ⏱️  Will start slow but speed up as it learns patterns")
        except Exception as e:
            logger.warning(f"   ⚠️  Adaptive cache failed: {e}")
            self.adaptive_cache = None

        logger.info("\n" + "=" * 70)
        logger.info("✅ CLUSTERED INTEGRATED AI READY")
        logger.info(f"   Clustering: {'ON' if self.enable_clustering else 'OFF'}")
        logger.info("=" * 70)

    def _query_learned_patterns(self, board: chess.Board, move: chess.Move) -> float:
        """
        Override parent's pattern query to use enhanced pattern matcher
        """
        if self.pattern_matcher:
            # Use enhanced matching with multiple strategies
            return self.pattern_matcher.get_move_bonus(board, move, board.turn)
        else:
            # Fall back to parent's exact matching
            return super()._query_learned_patterns(board, move)

    def evaluate_position(self, board: chess.Board, perspective: chess.Color) -> float:
        """
        Enhanced evaluation (clustering done at root, not here)

        NOTE: Clustering is now used in root move ordering (optimized_search.py)
              not during minimax recursion. This keeps evaluation fast.
        """
        # Just use base evaluation - clustering happens at root
        return super().evaluate_position(board, perspective)

    def _get_cluster_pattern_bonus(self, board: chess.Board, perspective: chess.Color) -> float:
        """
        Get pattern bonus from similar clustered positions

        Now uses adaptive cache:
        - First time: Slow (does full clustering + queries)
        - Subsequent times: Fast (cached result)
        - Over time: Builds cache of high-value cluster patterns
        """
        if not self.adaptive_cache or not self.clustering:
            return 0.0

        # Check if we should do expensive clustering queries
        if not self.adaptive_cache.should_do_expensive_queries():
            # Cache hit rate is good - skip expensive queries
            return 0.0

        start_time = time.time()
        self.cluster_stats['lookups'] += 1

        try:
            fen = board.fen()

            # Use cached cluster info if available
            cluster_id, similar = self.adaptive_cache.get_cluster_info(
                fen, self.clustering
            )

            if not similar:
                return 0.0

            self.cluster_stats['matches'] += 1

            # Query patterns from similar positions
            bonus = 0.0
            for sim_fen, distance, cluster_id in similar:
                # Weight by inverse distance (closer positions = higher weight)
                weight = 1.0 / (1.0 + distance)

                # Check learned tactics from this SPECIFIC similar position
                self.cursor.execute('''
                    SELECT AVG(success_rate) as avg_success,
                           AVG(material_gained) as avg_gain,
                           COUNT(*) as tactic_count
                    FROM learned_tactics
                    WHERE fen_before = ?
                    AND success_rate > 0.5
                ''', (sim_fen,))

                row = self.cursor.fetchone()
                if row and row[0] is not None:
                    avg_success, avg_gain, tactic_count = row
                    # Bonus weighted by similarity and number of tactics found
                    confidence = min(1.0, tactic_count / 5.0)
                    bonus += (avg_success * 50 + (avg_gain or 0) * 10) * weight * confidence

                # ALSO check for mistakes to avoid from similar positions
                self.cursor.execute('''
                    SELECT AVG(material_lost) as avg_loss,
                           COUNT(*) as mistake_count
                    FROM learned_mistakes
                    WHERE fen_before = ?
                ''', (sim_fen,))

                row = self.cursor.fetchone()
                if row and row[0] is not None:
                    avg_loss, mistake_count = row
                    # Penalty for positions similar to past mistakes
                    confidence = min(1.0, mistake_count / 3.0)
                    bonus -= avg_loss * 20 * weight * confidence  # Strong penalty

            elapsed = time.time() - start_time
            self.cluster_stats['avg_time'] = (
                (self.cluster_stats['avg_time'] * (self.cluster_stats['lookups'] - 1) + elapsed)
                / self.cluster_stats['lookups']
            )

            return bonus

        except Exception as e:
            logger.debug(f"Cluster bonus error: {e}")
            return 0.0

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            'positions_evaluated': self.positions_evaluated,
            'patterns_used': self.patterns_used,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100,
        }

        if self.enable_clustering:
            stats.update({
                'cluster_lookups': self.cluster_stats['lookups'],
                'cluster_matches': self.cluster_stats['matches'],
                'cluster_match_rate': (self.cluster_stats['matches'] /
                                      max(1, self.cluster_stats['lookups']) * 100),
                'avg_cluster_time_ms': self.cluster_stats['avg_time'] * 1000
            })

        if self.pattern_matcher:
            pattern_stats = self.pattern_matcher.get_stats()
            stats.update({
                'enhanced_pattern_queries': pattern_stats['total_queries'],
                'exact_pattern_matches': pattern_stats['exact_matches'],
                'tactical_pattern_matches': pattern_stats['pattern_matches'],
                'exact_match_rate': pattern_stats.get('exact_match_rate', 0.0),
                'tactical_match_rate': pattern_stats.get('pattern_match_rate', 0.0)
            })

        if self.adaptive_cache:
            cache_stats = self.adaptive_cache.get_stats()
            stats.update({
                'adaptive_cache_hits': cache_stats['cache_hits'],
                'adaptive_cache_hit_rate': cache_stats['cache_hit_rate'],
                'persistent_patterns_learned': cache_stats['persistent_cache_size'],
                'doing_expensive_queries': cache_stats['doing_expensive_queries']
            })

        return stats


def test_clustered_ai(num_games: int = 5, stockfish_level: int = 0):
    """Test the full AI with clustering"""
    print("=" * 70)
    print("FULL INTEGRATED AI + CLUSTERING TEST")
    print("=" * 70)

    # Initialize AI with clustering
    ai = ClusteredIntegratedAI(
        search_depth=1,
        enable_opponent_prediction=False,  # Disable for speed
        enable_pattern_learning=True,
        enable_clustering=True,
        time_limit_per_move=10.0
    )

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    engine.configure({"Skill Level": stockfish_level})

    results = {'wins': 0, 'draws': 0, 'losses': 0}
    total_moves = 0
    total_time = 0

    for game_num in range(1, num_games + 1):
        board = chess.Board()
        ai_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
        move_count = 0
        max_moves = 150
        game_start = time.time()

        color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"
        print(f"\nGame {game_num}/{num_games} - AI plays {color_str}")
        print("-" * 70)

        while not board.is_game_over() and move_count < max_moves:
            move_count += 1

            if board.turn == ai_color:
                # AI move
                sys.stdout.write(f"Move {move_count:3d} (AI): ")
                sys.stdout.flush()

                move_uci, score = ai.find_best_move(board.fen())
                if move_uci == "0000":
                    print("No legal moves!")
                    break

                move = chess.Move.from_uci(move_uci)
                print(f"{board.san(move)}")
                board.push(move)

            else:
                # Stockfish move
                sys.stdout.write(f"Move {move_count:3d} (SF): ")
                sys.stdout.flush()

                result = engine.play(board, chess.engine.Limit(time=0.5))
                print(f"{board.san(result.move)}")
                board.push(result.move)

            # Progress indicator
            if move_count % 20 == 0:
                elapsed = time.time() - game_start
                print(f"  [{move_count} moves, {elapsed:.1f}s elapsed]")

        game_time = time.time() - game_start
        total_time += game_time
        total_moves += move_count

        # Determine result
        if board.is_checkmate():
            if board.turn != ai_color:
                results['wins'] += 1
                outcome = "🎉 WIN"
            else:
                results['losses'] += 1
                outcome = "❌ LOSS"
        elif board.is_stalemate():
            results['draws'] += 1
            outcome = "🤝 DRAW (stalemate)"
        elif board.is_insufficient_material():
            results['draws'] += 1
            outcome = "🤝 DRAW (insufficient material)"
        else:
            results['draws'] += 1
            outcome = f"🤝 DRAW (max moves)"

        print(f"\nResult: {outcome} ({move_count} moves, {game_time:.1f}s)")

    engine.quit()

    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nGames:   {num_games}")
    print(f"Wins:    {results['wins']:2d} ({results['wins']/num_games*100:5.1f}%)")
    print(f"Draws:   {results['draws']:2d} ({results['draws']/num_games*100:5.1f}%)")
    print(f"Losses:  {results['losses']:2d} ({results['losses']/num_games*100:5.1f}%)")
    score = (results['wins'] + 0.5 * results['draws']) / num_games * 100
    print(f"Score:   {score:.1f}%")

    print(f"\nPerformance:")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Avg per game:     {total_time/num_games:.1f}s")
    print(f"  Avg per move:     {total_time/total_moves:.2f}s")
    print(f"  Moves per game:   {total_moves/num_games:.1f}")

    # AI statistics
    stats = ai.get_stats()
    print(f"\nAI Statistics:")
    print(f"  Positions evaluated: {stats['positions_evaluated']:,}")
    print(f"  Patterns used:       {stats['patterns_used']:,}")
    print(f"  Cache hit rate:      {stats['cache_hit_rate']:.1f}%")

    if ai.enable_clustering:
        print(f"\nClustering Statistics:")
        print(f"  Cluster lookups:     {stats['cluster_lookups']:,}")
        print(f"  Cluster matches:     {stats['cluster_matches']:,}")
        print(f"  Match rate:          {stats['cluster_match_rate']:.1f}%")
        print(f"  Avg lookup time:     {stats['avg_cluster_time_ms']:.2f}ms")

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    test_clustered_ai(num_games=num_games, stockfish_level=0)
