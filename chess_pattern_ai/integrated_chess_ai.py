#!/usr/bin/env python3
"""
Integrated Chess AI - All Components Assembled
Combines all existing mechanisms into a cohesive, working AI
"""

import chess
import chess.engine
import sqlite3
import logging
import sys
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import time

# Import all the existing components
from material_evaluator import MaterialEvaluator
from mobility_evaluator import MobilityEvaluator
from safety_evaluator import SafetyEvaluator
from tactical_evaluator import TacticalEvaluator
from opening_evaluator import OpeningEvaluator
from pawn_structure_evaluator import PawnStructureEvaluator
from positional_evaluator import PositionalEvaluator
from game_phase_detector import GamePhaseDetector
from temporal_evaluator import TemporalEvaluator
from weak_square_detector import WeakSquareDetector
from pattern_database_enhancer import PatternDatabaseEnhancer

# Try to import optional components
try:
    from opponent_response_predictor import OpponentResponsePredictor
    OPPONENT_PREDICTOR_AVAILABLE = True
except:
    OPPONENT_PREDICTOR_AVAILABLE = False
    print("⚠️  Opponent predictor not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MoveEvaluation:
    """Complete evaluation of a move"""
    move: chess.Move
    base_score: float
    material_score: float
    tactical_score: float
    positional_score: float
    threat_score: float
    pattern_bonus: float
    opponent_response_penalty: float
    total_score: float
    confidence: float


class IntegratedChessAI:
    """
    Fully Integrated Chess AI
    Assembles all components properly with performance optimizations
    """

    def __init__(self, db_path: str = "rule_discovery.db",
                 search_depth: int = 2,
                 enable_opponent_prediction: bool = False,
                 enable_pattern_learning: bool = True,
                 time_limit_per_move: float = 5.0):
        """
        Initialize integrated AI with all components

        Args:
            db_path: Database path
            search_depth: Minimax search depth (2=fast, 3=slow but better)
            enable_opponent_prediction: Enable opponent move prediction (requires indexes)
            enable_pattern_learning: Enable pattern database learning
            time_limit_per_move: Maximum time per move in seconds
        """
        self.db_path = db_path
        self.search_depth = search_depth
        self.time_limit = time_limit_per_move

        logger.info("=" * 70)
        logger.info("INTEGRATED CHESS AI - INITIALIZING ALL COMPONENTS")
        logger.info("=" * 70)

        # Core evaluators
        logger.info("\n1. Loading Core Evaluators...")
        self.material_evaluator = MaterialEvaluator(db_path)
        self.mobility_evaluator = MobilityEvaluator(db_path)
        self.safety_evaluator = SafetyEvaluator(db_path)
        self.tactical_evaluator = TacticalEvaluator(db_path)
        self.opening_evaluator = OpeningEvaluator(db_path)
        self.pawn_structure_evaluator = PawnStructureEvaluator(db_path)
        self.positional_evaluator = PositionalEvaluator(db_path)
        logger.info("   ✅ Core evaluators loaded")

        # Strategic components
        logger.info("\n2. Loading Strategic Components...")
        self.game_phase_detector = GamePhaseDetector(db_path)
        self.temporal_evaluator = TemporalEvaluator(db_path)
        self.weak_square_detector = WeakSquareDetector(db_path)
        logger.info("   ✅ Strategic components loaded")

        # Pattern database
        logger.info("\n3. Loading Pattern Database...")
        self.pattern_db = None
        if enable_pattern_learning:
            try:
                self.pattern_db = PatternDatabaseEnhancer(db_path, verbose=False)
                logger.info("   ✅ Pattern database loaded (learned patterns enabled)")
            except Exception as e:
                logger.warning(f"   ⚠️  Pattern database failed: {e}")

        # Opponent predictor (optional, requires indexes)
        logger.info("\n4. Loading Opponent Response Predictor...")
        self.opponent_predictor = None
        if enable_opponent_prediction and OPPONENT_PREDICTOR_AVAILABLE:
            try:
                self.opponent_predictor = OpponentResponsePredictor(db_path)
                logger.info("   ✅ Opponent predictor enabled")
            except Exception as e:
                logger.warning(f"   ⚠️  Opponent predictor failed: {e}")
        elif not enable_opponent_prediction:
            logger.info("   ⏭️  Opponent predictor disabled (performance)")
        else:
            logger.info("   ⚠️  Opponent predictor not available")

        # Database connection for pattern queries
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Load all weights
        logger.info("\n5. Loading Discovered Weights...")
        self._load_all_weights()
        logger.info("   ✅ All weights loaded")

        # Statistics
        self.positions_evaluated = 0
        self.patterns_used = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.position_cache = {}

        logger.info("\n" + "=" * 70)
        logger.info("✅ INTEGRATED AI READY")
        logger.info(f"   Search depth: {search_depth}")
        logger.info(f"   Time limit: {time_limit_per_move}s per move")
        logger.info(f"   Opponent prediction: {'ON' if self.opponent_predictor else 'OFF'}")
        logger.info(f"   Pattern learning: {'ON' if self.pattern_db else 'OFF'}")
        logger.info("=" * 70)

    def _load_all_weights(self):
        """Load all evaluation weights from database"""
        self.material_evaluator._load_piece_values()
        self.mobility_evaluator._load_mobility_weight()
        self.safety_evaluator._load_safety_weights()
        self.tactical_evaluator._load_tactical_weights()
        self.opening_evaluator._load_opening_weights()
        self.pawn_structure_evaluator._load_structure_weights()
        self.positional_evaluator._load_positional_weights()
        self.weak_square_detector._load_weak_square_weights()

    def evaluate_position(self, board: chess.Board, perspective: chess.Color) -> float:
        """
        Comprehensive position evaluation

        Args:
            board: Current position
            perspective: Which side to evaluate for

        Returns:
            Score from perspective's viewpoint (positive = good)
        """
        self.positions_evaluated += 1
        fen = board.fen()

        # Check cache
        cache_key = (fen, perspective)
        if cache_key in self.position_cache:
            self.cache_hits += 1
            return self.position_cache[cache_key]

        self.cache_misses += 1

        # Terminal positions
        if board.is_checkmate():
            score = -20000 if board.turn == perspective else 20000
            self.position_cache[cache_key] = score
            return score

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Component evaluations (using correct method names)
        material = self.material_evaluator.evaluate_material(fen)
        mobility = self.mobility_evaluator.evaluate_mobility(fen)
        safety = self.safety_evaluator.evaluate_safety(fen)
        tactical = self.tactical_evaluator.evaluate_tactics(fen)
        opening = self.opening_evaluator.evaluate_opening(fen)
        pawn_structure = self.pawn_structure_evaluator.evaluate_pawn_structure(fen)
        positional = self.positional_evaluator.evaluate_position(fen)
        weak_squares = self.weak_square_detector.evaluate_weak_squares(fen)

        # Hanging piece detection (critical!)
        threat_score = self._evaluate_threats_fast(board, perspective)

        # Phase-based weighting
        game_phase = self.game_phase_detector.detect_game_phase(fen, board.fullmove_number)
        base_scores = {
            'material': material * 10.0,  # 10x boost (critical)
            'mobility': mobility,
            'safety': safety,
            'pawn_structure': pawn_structure,
            'positional': positional,
            'tactical': tactical
        }

        total = self.temporal_evaluator.evaluate_with_phase_adaptation(
            base_scores, fen, board.fullmove_number
        )

        total += threat_score + opening + weak_squares

        # Flip if evaluating for black
        if perspective == chess.BLACK:
            total = -total

        self.position_cache[cache_key] = total
        return total

    def _evaluate_threats_fast(self, board: chess.Board, perspective: chess.Color) -> float:
        """Fast hanging piece detection"""
        threat_score = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == perspective:
                attackers = len(list(board.attackers(not perspective, square)))
                defenders = len(list(board.attackers(perspective, square)))

                if attackers > defenders:
                    piece_value = self.material_evaluator.piece_values.get(
                        piece.symbol().upper(), 1.0
                    )
                    imbalance = attackers - defenders
                    penalty_factor = min(1.0, 0.7 + 0.3 * (imbalance - 1))
                    threat_score -= piece_value * penalty_factor

        return threat_score

    def _query_learned_patterns(self, board: chess.Board, move: chess.Move) -> float:
        """Query pattern database for bonuses"""
        if not self.pattern_db:
            return 0.0

        bonus = 0.0

        # Make move temporarily
        board.push(move)
        fen_after = board.fen()
        board.pop()
        fen_before = board.fen()

        try:
            # Check learned tactics
            self.cursor.execute("""
                SELECT material_gained, success_rate, times_seen
                FROM learned_tactics
                WHERE fen_before = ? AND move_sequence LIKE ?
                AND success_rate > 0.5
                ORDER BY material_gained DESC
                LIMIT 1
            """, (fen_before, f"{move.uci()}%"))

            result = self.cursor.fetchone()
            if result:
                mat_gain, success_rate, times_seen = result
                bonus += mat_gain * success_rate * min(1.0, times_seen / 10.0)
                self.patterns_used += 1

            # Check mistakes to avoid
            self.cursor.execute("""
                SELECT material_lost
                FROM learned_mistakes
                WHERE fen_before = ? AND move_made = ?
                LIMIT 1
            """, (fen_before, move.uci()))

            result = self.cursor.fetchone()
            if result:
                material_lost = result[0]
                bonus -= material_lost * 2.0  # Heavy penalty

        except Exception as e:
            logger.debug(f"Pattern query error: {e}")

        return bonus

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                maximizing: bool, perspective: chess.Color, start_time: float) -> float:
        """
        Minimax with alpha-beta pruning and time limit

        Args:
            board: Current position
            depth: Remaining depth
            alpha: Alpha value
            beta: Beta value
            maximizing: True if maximizing player
            perspective: Original caller's perspective
            start_time: When search started (for time control)

        Returns:
            Evaluation score
        """
        # Time check
        if time.time() - start_time > self.time_limit:
            return self.evaluate_position(board, perspective)

        # Terminal conditions
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board, perspective)

        legal_moves = list(board.legal_moves)

        # Move ordering: captures first
        legal_moves.sort(key=lambda m: (
            board.is_capture(m),
            board.gives_check(m)
        ), reverse=True)

        if maximizing:
            max_eval = -99999
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False, perspective, start_time)
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
                eval = self.minimax(board, depth - 1, alpha, beta, True, perspective, start_time)
                board.pop()

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval

    def find_best_move(self, fen: str) -> Tuple[str, float]:
        """
        Find best move using full AI capabilities

        Args:
            fen: Position to analyze

        Returns:
            (move_uci, score)
        """
        start_time = time.time()
        board = chess.Board(fen)

        if board.is_game_over():
            return "0000", 0.0

        legal_moves = list(board.legal_moves)
        perspective = board.turn

        best_move = legal_moves[0]
        best_score = -99999

        # Move ordering for search
        move_scores = []
        for move in legal_moves:
            # Quick evaluation
            board.push(move)
            quick_score = self.evaluate_position(board, perspective)
            board.pop()

            # Pattern bonus
            pattern_bonus = self._query_learned_patterns(board, move)
            quick_score += pattern_bonus

            move_scores.append((move, quick_score))

        # Sort by quick evaluation
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Search top moves with minimax
        for move, quick_score in move_scores:
            if time.time() - start_time > self.time_limit:
                logger.warning(f"⏱️  Time limit reached, returning best so far")
                break

            board.push(move)

            # Minimax search
            score = self.minimax(
                board, self.search_depth - 1, -99999, 99999,
                False, perspective, start_time
            )

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        elapsed = time.time() - start_time
        logger.debug(f"Move selected in {elapsed:.2f}s: {best_move.uci()} (score: {best_score:.1f})")

        return best_move.uci(), best_score

    def get_statistics(self) -> Dict:
        """Get AI statistics"""
        total_cache = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_cache * 100) if total_cache > 0 else 0

        return {
            'positions_evaluated': self.positions_evaluated,
            'patterns_used': self.patterns_used,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{hit_rate:.1f}%",
            'search_depth': self.search_depth
        }

    def close(self):
        """Clean up resources"""
        if self.conn:
            self.conn.close()


def test_integrated_ai(num_games: int = 20, search_depth: int = 2):
    """Test the integrated AI"""
    print("=" * 70)
    print("INTEGRATED CHESS AI TEST")
    print(f"All components assembled, depth={search_depth}")
    print("=" * 70)

    ai = IntegratedChessAI(
        search_depth=search_depth,
        enable_opponent_prediction=False,  # Disabled until indexes added
        enable_pattern_learning=True,
        time_limit_per_move=10.0
    )

    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        engine.configure({"Skill Level": 0})
    except:
        print("❌ Stockfish not available")
        ai.close()
        return

    print(f"\nPlaying {num_games} games vs Stockfish (level 0)...")
    print("=" * 70)

    wins, draws, losses = 0, 0, 0

    for game_num in range(1, num_games + 1):
        ai_as_white = (game_num % 2 == 1)
        board = chess.Board()
        move_count = 0

        # Reset stats for this game
        ai.positions_evaluated = 0
        ai.patterns_used = 0

        while not board.is_game_over() and move_count < 150:
            move_count += 1

            try:
                if (board.turn == chess.WHITE) == ai_as_white:
                    move_uci, score = ai.find_best_move(board.fen())
                    move = chess.Move.from_uci(move_uci)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1, depth=5))
                    move = result.move

                board.push(move)
            except Exception as e:
                logger.error(f"Error in game {game_num}, move {move_count}: {e}")
                legal = list(board.legal_moves)
                if legal:
                    board.push(legal[0])
                else:
                    break

        result = board.result()
        ai_won = (result == "1-0" and ai_as_white) or (result == "0-1" and not ai_as_white)
        is_draw = result == "1/2-1/2"

        if ai_won:
            wins += 1
            outcome = "✅ WIN"
        elif is_draw:
            draws += 1
            outcome = "➖ DRAW"
        else:
            losses += 1
            outcome = "❌ LOSS"

        win_rate = (wins + 0.5 * draws) / game_num * 100

        print(f"Game {game_num:2d}/20 ({'W' if ai_as_white else 'B'}): {outcome} ({move_count:3d}m) | "
              f"Score: {win_rate:5.1f}% | Pos: {ai.positions_evaluated:,} | Patterns: {ai.patterns_used}")

    engine.quit()

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Games:   {num_games}")
    print(f"Wins:    {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws:   {draws} ({draws/num_games*100:.1f}%)")
    print(f"Losses:  {losses} ({losses/num_games*100:.1f}%)")
    print(f"Score:   {(wins + 0.5*draws)/num_games*100:.1f}%")

    stats = ai.get_statistics()
    print(f"\nAI Statistics:")
    print(f"  Positions evaluated: {stats['positions_evaluated']:,}")
    print(f"  Patterns used:       {stats['patterns_used']}")
    print(f"  Cache hit rate:      {stats['cache_hit_rate']}")
    print(f"  Search depth:        {stats['search_depth']}")
    print("=" * 70)

    if wins > 0:
        print(f"\n🎉 SUCCESS! Won {wins} games with integrated AI!")
    elif draws > 0:
        print(f"\n➖ Progress: {draws} draws achieved")
    else:
        print("\n📊 No wins yet - consider adding database indexes for better performance")

    ai.close()

    return wins, draws, losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Integrated Chess Pattern AI')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--depth', type=int, default=1, help='Search depth (1=fast, 2=slow)')
    parser.add_argument('--stockfish-level', type=int, default=0, help='Stockfish skill level (0-20)')
    parser.add_argument('--time-limit', type=float, default=10.0, help='Time limit per move (seconds)')

    args = parser.parse_args()

    # Test with specified parameters (depth=1 is much faster!)
    test_integrated_ai(num_games=args.games, search_depth=args.depth)
