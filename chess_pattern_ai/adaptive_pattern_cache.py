#!/usr/bin/env python3
"""
Adaptive Pattern Cache - Learns Which Patterns Are Most Valuable

Strategy:
1. Start slow - do comprehensive pattern matching (clustering, tactical, etc.)
2. Track which patterns actually help make good moves
3. Build a cache of high-value patterns
4. Over time, rely more on cache and less on expensive queries
5. Result: Slow at first, but gets faster as it learns what matters
"""

import sqlite3
import json
import time
from typing import Dict, List, Tuple, Optional
import chess
import logging

logger = logging.getLogger(__name__)


class AdaptivePatternCache:
    """
    Intelligent cache that learns which patterns are worth the query cost
    """

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Runtime cache (in-memory, fast)
        self.pattern_cache = {}  # (fen, move) -> cached_bonus
        self.cluster_cache = {}  # fen -> (cluster_id, similar_positions)

        # Pattern value tracking
        self.pattern_usefulness = {}  # pattern_type -> running statistics

        # Adaptive thresholds
        self.cache_hit_threshold = 0.10  # Start doing expensive queries if cache hit < 10%
        self.patterns_checked = 0
        self.cache_hits = 0

        # Initialize persistent cache table
        self._init_persistent_cache()

        # Load high-value patterns from previous sessions
        self._load_persistent_cache()

    def _init_persistent_cache(self):
        """Create table for storing learned high-value patterns"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_value_cache (
                fen_pattern TEXT,
                move_pattern TEXT,
                cached_bonus REAL,
                times_used INTEGER DEFAULT 1,
                avg_search_time_ms REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fen_pattern, move_pattern)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pattern_cache_usage
            ON pattern_value_cache(times_used DESC, avg_search_time_ms)
        ''')

        self.conn.commit()

    def _load_persistent_cache(self):
        """Load frequently-used patterns from database into memory"""
        # Load top 5000 most-used patterns
        self.cursor.execute('''
            SELECT fen_pattern, move_pattern, cached_bonus, times_used
            FROM pattern_value_cache
            ORDER BY times_used DESC
            LIMIT 5000
        ''')

        count = 0
        for fen_pat, move_pat, bonus, uses in self.cursor.fetchall():
            cache_key = (fen_pat, move_pat)
            self.pattern_cache[cache_key] = {
                'bonus': bonus,
                'uses': uses,
                'from_persistent': True
            }
            count += 1

        if count > 0:
            logger.info(f"   📚 Loaded {count} high-value patterns from previous sessions")

    def get_pattern_bonus(self, fen: str, move_uci: str,
                          expensive_query_func) -> Tuple[float, bool]:
        """
        Get pattern bonus, using cache when available, expensive queries when needed

        Args:
            fen: Position FEN
            move_uci: Move in UCI format
            expensive_query_func: Function to call if not in cache (slow but thorough)

        Returns:
            (bonus, from_cache) tuple
        """
        self.patterns_checked += 1
        cache_key = (fen, move_uci)

        # Check in-memory cache first
        if cache_key in self.pattern_cache:
            self.cache_hits += 1
            cached = self.pattern_cache[cache_key]
            cached['uses'] = cached.get('uses', 0) + 1
            return cached['bonus'], True

        # Not in cache - do expensive query
        start_time = time.time()
        bonus = expensive_query_func()
        query_time_ms = (time.time() - start_time) * 1000

        # Store in cache for future use
        self.pattern_cache[cache_key] = {
            'bonus': bonus,
            'uses': 1,
            'query_time_ms': query_time_ms,
            'from_persistent': False
        }

        # If this pattern is valuable (non-zero bonus) and was expensive to compute,
        # save it to persistent storage
        if abs(bonus) > 1.0 and query_time_ms > 10.0:  # Meaningful bonus + slow query
            self._save_to_persistent_cache(fen, move_uci, bonus, query_time_ms)

        return bonus, False

    def _save_to_persistent_cache(self, fen: str, move_uci: str,
                                   bonus: float, query_time_ms: float):
        """Save a high-value pattern to persistent storage"""
        try:
            self.cursor.execute('''
                INSERT INTO pattern_value_cache
                (fen_pattern, move_pattern, cached_bonus, avg_search_time_ms)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fen_pattern, move_pattern) DO UPDATE SET
                    times_used = times_used + 1,
                    last_updated = CURRENT_TIMESTAMP
            ''', (fen, move_uci, bonus, query_time_ms))

            # Commit periodically (every 10 new patterns)
            if self.patterns_checked % 10 == 0:
                self.conn.commit()
        except Exception as e:
            logger.debug(f"Cache save error: {e}")

    def get_cluster_info(self, fen: str, clustering_engine) -> Tuple[int, List]:
        """
        Get cluster info with caching

        Args:
            fen: Position FEN
            clustering_engine: The clustering engine to query if not cached

        Returns:
            (cluster_id, similar_positions) tuple
        """
        if fen in self.cluster_cache:
            return self.cluster_cache[fen]

        # Compute cluster info
        try:
            similar = clustering_engine.find_similar_positions(fen, limit=5)
            cluster_id = similar[0][2] if similar else -1

            # Cache for future
            self.cluster_cache[fen] = (cluster_id, similar)

            # Limit cache size
            if len(self.cluster_cache) > 5000:
                # Keep only most recent 2500
                keys = list(self.cluster_cache.keys())
                for k in keys[:2500]:
                    del self.cluster_cache[k]

            return cluster_id, similar
        except Exception as e:
            logger.debug(f"Clustering error: {e}")
            return -1, []

    def should_do_expensive_queries(self) -> bool:
        """
        Decide whether to do expensive pattern matching queries

        Returns True early on (building cache), False later (using cache)
        """
        if self.patterns_checked < 1000:
            # Always do expensive queries for first 1000 patterns (learning phase)
            return True

        cache_hit_rate = self.cache_hits / max(1, self.patterns_checked)

        if cache_hit_rate < self.cache_hit_threshold:
            # Cache hit rate is low - need more data
            return True

        # Adaptive: occasionally do expensive queries to catch new patterns
        # Do expensive query 5% of the time even with good cache hit rate
        return (self.patterns_checked % 20) == 0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        cache_hit_rate = 0.0
        if self.patterns_checked > 0:
            cache_hit_rate = self.cache_hits / self.patterns_checked * 100

        persistent_count = 0
        try:
            self.cursor.execute('SELECT COUNT(*) FROM pattern_value_cache')
            persistent_count = self.cursor.fetchone()[0]
        except:
            pass

        return {
            'patterns_checked': self.patterns_checked,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'in_memory_cache_size': len(self.pattern_cache),
            'cluster_cache_size': len(self.cluster_cache),
            'persistent_cache_size': persistent_count,
            'doing_expensive_queries': self.should_do_expensive_queries()
        }

    def update_from_game_outcome(self, move_history: list, outcome: str, ai_color):
        """
        Update cached patterns based on game outcome

        If game was lost, reduce bonuses for moves that were played
        If game was won, increase bonuses for moves that were played

        This ensures the cache learns from outcomes, not just evaluations
        """
        if outcome == 'loss':
            penalty_factor = 0.5  # Reduce cached bonuses by 50%
        elif outcome == 'win':
            bonus_factor = 1.2  # Increase cached bonuses by 20%
        else:  # draw
            return  # No adjustment for draws

        # Update cache for moves we actually played
        for i, (fen, move_uci, move_san) in enumerate(move_history):
            # Only update our moves (every other move)
            board = chess.Board(fen)
            if board.turn != ai_color:
                continue

            cache_key = (fen, move_uci)

            # If this move is in cache, adjust its value based on outcome
            if cache_key in self.pattern_cache:
                old_bonus = self.pattern_cache[cache_key]['bonus']

                if outcome == 'loss':
                    new_bonus = old_bonus * penalty_factor
                else:  # win
                    new_bonus = old_bonus * bonus_factor

                self.pattern_cache[cache_key]['bonus'] = new_bonus

                # Update persistent cache too
                try:
                    self.cursor.execute('''
                        UPDATE pattern_value_cache
                        SET cached_bonus = ?
                        WHERE fen_pattern = ? AND move_pattern = ?
                    ''', (new_bonus, fen, move_uci))
                except:
                    pass

        # Commit the updates
        self.commit()

    def clear_losing_patterns(self, move_history: list, ai_color):
        """
        Remove patterns from cache that led to a loss

        This forces re-evaluation next game, allowing exploration
        """
        for fen, move_uci, move_san in move_history:
            board = chess.Board(fen)
            if board.turn != ai_color:
                continue

            cache_key = (fen, move_uci)

            # Remove from in-memory cache (will be re-evaluated next time)
            if cache_key in self.pattern_cache:
                del self.pattern_cache[cache_key]

                # Also mark in persistent cache as "needs re-evaluation"
                try:
                    self.cursor.execute('''
                        DELETE FROM pattern_value_cache
                        WHERE fen_pattern = ? AND move_pattern = ?
                    ''', (fen, move_uci))
                except:
                    pass

    def commit(self):
        """Commit any pending cache updates to database"""
        try:
            self.conn.commit()
        except Exception as e:
            logger.debug(f"Commit error: {e}")


def test_adaptive_cache():
    """Test the adaptive cache system"""
    print("=" * 70)
    print("ADAPTIVE PATTERN CACHE TEST")
    print("=" * 70)

    cache = AdaptivePatternCache()

    # Simulate expensive query
    def expensive_query():
        time.sleep(0.01)  # Simulate 10ms query
        return 15.5  # Some bonus value

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = "e2e4"

    # First call - should be slow
    start = time.time()
    bonus1, cached1 = cache.get_pattern_bonus(fen, move, expensive_query)
    time1 = (time.time() - start) * 1000

    print(f"\nFirst lookup: {time1:.1f}ms (cached={cached1}, bonus={bonus1})")

    # Second call - should be instant from cache
    start = time.time()
    bonus2, cached2 = cache.get_pattern_bonus(fen, move, expensive_query)
    time2 = (time.time() - start) * 1000

    print(f"Second lookup: {time2:.1f}ms (cached={cached2}, bonus={bonus2})")
    print(f"Speedup: {time1/time2:.0f}x faster!")

    # Show stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    cache.commit()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_adaptive_cache()
