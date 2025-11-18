#!/usr/bin/env python3
"""
Optimized Search - Intelligent Move Pruning

Only search moves that make sense:
1. Prune moves that hang pieces without compensation
2. Prune king moves into danger
3. Focus on forcing moves (checks, captures, threats)
4. Use quiescence search for tactical positions
5. Limit branching factor intelligently
"""

import chess
from typing import List, Tuple
import time


class OptimizedSearchMixin:
    """
    Mixin to add intelligent move pruning to the AI
    Dramatically reduces search space by ignoring nonsensical moves
    """

    def _filter_sensible_moves(self, board: chess.Board, perspective: chess.Color,
                               depth: int) -> List[chess.Move]:
        """
        Filter moves to only those worth considering

        Returns:
            List of sensible moves (much smaller than all legal moves)
        """
        legal_moves = list(board.legal_moves)

        # At depth 0 or few moves, return all
        if depth <= 0 or len(legal_moves) <= 5:
            return legal_moves

        sensible_moves = []
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

        for move in legal_moves:
            # Always include forcing moves
            if board.is_capture(move) or board.gives_check(move):
                sensible_moves.append(move)
                continue

            # Check if move hangs a piece badly
            piece = board.piece_at(move.from_square)
            if piece:
                board.push(move)

                # Is the moved piece now hanging?
                attackers = len(list(board.attackers(not perspective, move.to_square)))
                defenders = len(list(board.attackers(perspective, move.to_square)))

                piece_value = piece_values.get(piece.piece_type, 0)

                # If piece is hanging and valuable, skip this move
                if attackers > defenders and piece_value > 100:
                    # Unless it's a sacrifice for checkmate
                    if not board.is_checkmate():
                        board.pop()
                        continue  # Skip this bad move!

                # Don't move king next to enemy king or into check
                if piece.piece_type == chess.KING:
                    if board.is_check():
                        board.pop()
                        continue  # King walked into check - bad!

                board.pop()

            # Include development moves in opening
            if len(board.move_stack) < 15:
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
                    from_rank = chess.square_rank(move.from_square)
                    # Moving piece from back rank
                    if from_rank in [0, 7]:
                        sensible_moves.append(move)
                        continue

            # Include center control moves
            if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                sensible_moves.append(move)
                continue

            # For other quiet moves, only include a few random ones
            # (This prevents searching every pawn push)
            if len(sensible_moves) < 10:
                sensible_moves.append(move)

        # If we filtered too aggressively, add back some moves
        if len(sensible_moves) < 3 and len(legal_moves) > 3:
            # Add a few more moves
            for move in legal_moves:
                if move not in sensible_moves:
                    sensible_moves.append(move)
                    if len(sensible_moves) >= 8:
                        break

        return sensible_moves if sensible_moves else legal_moves

    def minimax_optimized(self, board: chess.Board, depth: int, alpha: float, beta: float,
                         maximizing: bool, perspective: chess.Color, start_time: float) -> float:
        """
        Optimized minimax with intelligent move pruning
        Only searches sensible moves, dramatically reducing search space
        """
        # Time check
        if time.time() - start_time > self.time_limit:
            return self.evaluate_position(board, perspective)

        # Terminal conditions
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board, perspective)

        # Get FILTERED moves (only sensible ones!)
        legal_moves = self._filter_sensible_moves(board, perspective, depth)

        # Move ordering: best moves first for better pruning
        legal_moves.sort(key=lambda m: (
            board.is_capture(m),      # Captures first
            board.gives_check(m),     # Checks second
            m.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]  # Center third
        ), reverse=True)

        if maximizing:
            max_eval = -99999
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_optimized(board, depth - 1, alpha, beta, False, perspective, start_time)
                board.pop()

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval
        else:
            min_eval = 99999
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_optimized(board, depth - 1, alpha, beta, True, perspective, start_time)
                board.pop()

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval

    def find_best_move_optimized(self, fen: str, use_optimized: bool = True) -> Tuple[str, float]:
        """
        Find best move with two-stage optimization:

        STAGE 1: Root move ordering with comprehensive database queries
                 - Uses clustering, pattern matching, everything
                 - Slow initially, builds adaptive cache
                 - Gets faster over time as cache fills

        STAGE 2: Deep minimax search with fast evaluation
                 - Uses only exact pattern matching
                 - Always fast, no expensive queries
        """
        start_time = time.time()
        board = chess.Board(fen)

        if board.is_game_over():
            return "0000", 0.0

        legal_moves = list(board.legal_moves)
        perspective = board.turn

        best_move = legal_moves[0]
        best_score = -99999

        # ===================================================================
        # STAGE 1: ROOT MOVE ORDERING (Comprehensive, uses adaptive cache)
        # ===================================================================
        move_scores = []
        for move in legal_moves:
            board.push(move)

            # Use FULL evaluation including clustering at root
            quick_score = self._evaluate_root_move(board, move, perspective)

            board.pop()
            move_scores.append((move, quick_score))

        # Sort by comprehensive evaluation
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Search only top N moves (intelligent pruning)
        top_n = min(15, len(move_scores))

        # ===================================================================
        # STAGE 2: DEEP SEARCH (Fast, exact matching only)
        # ===================================================================
        for i, (move, quick_score) in enumerate(move_scores[:top_n]):
            board.push(move)

            # Use optimized minimax WITHOUT expensive clustering queries
            if use_optimized:
                eval_score = self.minimax_optimized(
                    board, self.search_depth - 1, -99999, 99999,
                    False, perspective, start_time
                )
            else:
                eval_score = self.minimax(
                    board, self.search_depth - 1, -99999, 99999,
                    False, perspective, start_time
                )

            board.pop()

            if eval_score > best_score:
                best_score = eval_score
                best_move = move

            # Time check
            if time.time() - start_time > self.time_limit * 0.9:
                break

        return best_move.uci(), best_score

    def _evaluate_root_move(self, board: chess.Board, move: chess.Move,
                           perspective: chess.Color) -> float:
        """
        Comprehensive evaluation for root moves

        This is where we use the expensive database queries:
        - Clustering to find similar positions
        - Pattern matching across all 19.5M tactics
        - Adaptive cache learns what matters
        - OPENING PERFORMANCE TRACKING - learns from actual game outcomes!

        Called ~30 times per move (once per legal move)
        Gets faster over time as adaptive cache fills
        """
        fen_before = board.fen()

        # Basic position evaluation (fast)
        base_score = self.evaluate_position(board, perspective)

        # Add exact pattern matching (fast, always used)
        pattern_bonus = self._query_learned_patterns(board, move)
        base_score += pattern_bonus

        # Add clustering bonus if available (slow → fast with adaptive cache)
        if hasattr(self, 'adaptive_cache') and hasattr(self, 'clustering'):
            if self.adaptive_cache and self.clustering:
                cluster_bonus = self._get_root_cluster_bonus(board, move, perspective)
                base_score += cluster_bonus

        # Add opening performance adjustment (learns from wins/losses!)
        # IMPORTANT: This is applied AFTER caching, so it always reflects
        # the latest game outcomes even if other parts are cached
        if hasattr(self, 'opening_tracker') and self.opening_tracker:
            # This is the KEY: adjust score based on actual game results
            opening_adjustment = self.opening_tracker.get_opening_adjustment(
                fen_before, move.uci()
            )
            base_score += opening_adjustment

        # Check for abstract pattern violations (learns principles, not positions!)
        if hasattr(self, 'pattern_engine') and self.pattern_engine:
            violations = self.pattern_engine.check_for_known_patterns(fen_before, move.uci())
            for desc, avg_loss, confidence, win_rate in violations:
                # Apply outcome-aware penalty:
                # - High material loss = bad
                # - Low win rate with pattern = VERY bad
                # - High confidence = trust the pattern more

                # Base penalty from material loss
                material_penalty = avg_loss * 20

                # Outcome penalty: patterns with low win rates get MASSIVE penalties
                # Formula: (1 - win_rate) * 200
                # - 0% win rate → 200 point penalty
                # - 50% win rate → 100 point penalty
                # - 100% win rate → 0 point penalty (pattern doesn't hurt win rate)
                outcome_penalty = (1.0 - win_rate) * 200

                # Apply confidence multiplier
                total_penalty = (material_penalty + outcome_penalty) * confidence

                base_score -= total_penalty

        return base_score

    def _get_root_cluster_bonus(self, board: chess.Board, move: chess.Move,
                                perspective: chess.Color) -> float:
        """
        Get cluster-based pattern bonus for root move evaluation

        Uses adaptive cache:
        - First time: Slow (queries database for similar positions)
        - Subsequent: Fast (cached result)
        - Learns: Saves high-value patterns to persistent storage
        """
        if not hasattr(self, 'adaptive_cache') or not self.adaptive_cache:
            return 0.0

        fen = board.fen()
        move_uci = move.uci()

        # Define the expensive query function
        def expensive_cluster_query():
            bonus = 0.0

            try:
                # Get similar positions from clustering
                cluster_id, similar = self.adaptive_cache.get_cluster_info(
                    fen, self.clustering
                )

                if not similar:
                    return 0.0

                # Query patterns from similar positions
                for sim_fen, distance, cluster_id in similar:
                    weight = 1.0 / (1.0 + distance)  # Closer = more relevant

                    # Check tactics from similar position
                    self.cursor.execute('''
                        SELECT AVG(success_rate), AVG(material_gained), COUNT(*)
                        FROM learned_tactics
                        WHERE fen_before = ? AND success_rate > 0.5
                    ''', (sim_fen,))

                    row = self.cursor.fetchone()
                    if row and row[0]:
                        avg_success, avg_gain, count = row
                        confidence = min(1.0, count / 5.0)
                        bonus += (avg_success * 50 + (avg_gain or 0) * 10) * weight * confidence

                    # Check mistakes from similar position
                    self.cursor.execute('''
                        SELECT AVG(material_lost), COUNT(*)
                        FROM learned_mistakes
                        WHERE fen_before = ?
                    ''', (sim_fen,))

                    row = self.cursor.fetchone()
                    if row and row[0]:
                        avg_loss, count = row
                        confidence = min(1.0, count / 3.0)
                        bonus -= avg_loss * 20 * weight * confidence

            except Exception as e:
                pass  # Return 0 on error

            return bonus

        # Use adaptive cache (slow first time, fast after)
        cached_bonus, from_cache = self.adaptive_cache.get_pattern_bonus(
            fen, move_uci, expensive_cluster_query
        )

        return cached_bonus


def estimate_branching_reduction(depth: int = 3):
    """
    Estimate how much faster optimized search is
    """
    print("=" * 70)
    print("BRANCHING FACTOR COMPARISON")
    print("=" * 70)

    avg_legal_moves = 35  # Average in chess
    avg_sensible_moves = 12  # After filtering

    print(f"\nAverage legal moves per position: {avg_legal_moves}")
    print(f"Average sensible moves after filtering: {avg_sensible_moves}")
    print(f"Reduction: {(1 - avg_sensible_moves/avg_legal_moves)*100:.1f}%")

    print(f"\nPositions evaluated at depth {depth}:")
    print(f"  Without filtering: {avg_legal_moves ** depth:,} positions")
    print(f"  With filtering:    {avg_sensible_moves ** depth:,} positions")
    print(f"  Speedup:           {(avg_legal_moves/avg_sensible_moves) ** depth:.1f}x faster!")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    estimate_branching_reduction(depth=3)
