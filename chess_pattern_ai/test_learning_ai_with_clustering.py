#!/usr/bin/env chess_env/bin/python3
"""
Test Learning AI with Clustering and Database Indexes

This test:
1. Uses depth 3 search (plans 3 moves ahead)
2. Records all games to database
3. Learns from mistakes
4. Shows improvement over time
"""

import chess
import chess.engine
import sqlite3
import logging
import sys
import time
from typing import List, Tuple, Optional
from integrated_ai_with_clustering import ClusteredIntegratedAI
from pattern_database_enhancer import PatternDatabaseEnhancer
from pattern_abstraction_engine import PatternAbstractionEngine

# Try to import Stockfish evaluator, but don't require it
try:
    from stockfish_move_evaluator import StockfishMoveEvaluator
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False
    StockfishMoveEvaluator = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LearningGameTracker:
    """Tracks games and records learning data"""

    def __init__(self, db_path: str = "rule_discovery.db",
                 use_stockfish_feedback: bool = True):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.cursor = self.conn.cursor()

        # Enable WAL mode for better concurrent access
        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.cursor.execute("PRAGMA busy_timeout=30000")

        # Initialize tables using PatternDatabaseEnhancer, then close its connection
        pattern_db = PatternDatabaseEnhancer(db_path, verbose=False)
        pattern_db.conn.close()  # Close to avoid multiple connections
        del pattern_db

        # Pattern abstraction engine - learns WHY moves are bad, not just which moves
        # Share database connection to avoid locking issues
        self.pattern_engine = PatternAbstractionEngine.__new__(PatternAbstractionEngine)
        self.pattern_engine.db_path = db_path
        self.pattern_engine.conn = self.conn
        self.pattern_engine.cursor = self.cursor
        self.pattern_engine._init_tables()
        logger.info("🧠 Pattern abstraction enabled - will learn principles, not positions")

        # Optional Stockfish feedback for enhanced learning
        self.stockfish_evaluator = None
        if use_stockfish_feedback and STOCKFISH_AVAILABLE:
            try:
                self.stockfish_evaluator = StockfishMoveEvaluator(
                    analysis_depth=12,  # Lighter analysis for speed
                    analysis_time=0.05   # 50ms per move
                )
                logger.info("✨ Stockfish feedback enabled - will analyze move quality")
            except Exception as e:
                logger.warning(f"Stockfish feedback unavailable: {e}")
                self.stockfish_evaluator = None
        elif use_stockfish_feedback:
            logger.info("Stockfish feedback requested but evaluator not available")

    def record_game(self, board: chess.Board, ai_color: chess.Color,
                   result: str, moves: List[Tuple[str, str]]):
        """
        Record game and extract learning data

        Args:
            board: Final board position
            ai_color: Which color the AI played
            result: 'win', 'loss', or 'draw'
            moves: List of (fen_before, move_uci) tuples
        """
        # Insert game record
        self.cursor.execute('''
            INSERT INTO games (white_player, black_player, result, ai_color, played_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (
            'AI' if ai_color == chess.WHITE else 'Stockfish',
            'Stockfish' if ai_color == chess.WHITE else 'AI',
            result,
            'white' if ai_color == chess.WHITE else 'black'
        ))
        game_id = self.cursor.lastrowid

        # Optional: Analyze with Stockfish for enhanced learning
        stockfish_analysis = None
        if self.stockfish_evaluator:
            stockfish_analysis = self._analyze_with_stockfish(moves, ai_color)

        # Analyze moves for learning (with optional Stockfish feedback)
        if result == 'loss':
            self._learn_from_loss(game_id, moves, ai_color, stockfish_analysis)
        elif result == 'win':
            self._learn_from_win(game_id, moves, ai_color, stockfish_analysis)

        self.conn.commit()

    def _analyze_with_stockfish(self, moves: List[Tuple], ai_color: chess.Color) -> dict:
        """
        Analyze moves with Stockfish (optional, enhances learning)

        Returns dict: {move_index: (cp_loss, classification, best_move)}
        """
        if not self.stockfish_evaluator:
            return {}

        analysis = {}
        board = chess.Board()

        try:
            self.stockfish_evaluator.start()

            for i, (fen_before, move_uci, move_san) in enumerate(moves):
                try:
                    board.set_fen(fen_before)
                except:
                    continue

                # Only analyze AI's moves
                if board.turn != ai_color:
                    board.push(chess.Move.from_uci(move_uci))
                    continue

                move = chess.Move.from_uci(move_uci)

                # Get Stockfish's evaluation
                _, _, classification, cp_loss = self.stockfish_evaluator.evaluate_move(board, move)
                best_move, _ = self.stockfish_evaluator.get_best_move(board)

                analysis[i] = {
                    'cp_loss': cp_loss,
                    'classification': classification,
                    'best_move': best_move.uci() if best_move else None
                }

                board.push(move)

            self.stockfish_evaluator.stop()

        except Exception as e:
            logger.debug(f"Stockfish analysis error: {e}")
            if self.stockfish_evaluator:
                self.stockfish_evaluator.stop()

        return analysis

    def _learn_from_loss(self, game_id: int, moves: List[Tuple], ai_color: chess.Color,
                        stockfish_analysis: Optional[dict] = None):
        """
        Extract mistakes from losing game

        Args:
            stockfish_analysis: Optional Stockfish feedback for enhanced learning
        """
        board = chess.Board()
        all_patterns_in_game = []  # Track all patterns for outcome learning

        for i, (fen_before, move_uci, move_san) in enumerate(moves):
            # Reconstruct position
            try:
                board.set_fen(fen_before)
            except:
                continue

            # Only analyze AI's moves
            if board.turn != ai_color:
                board.push(chess.Move.from_uci(move_uci))
                continue

            # Check if this move led to material loss
            material_before = self._count_material(board)

            move = chess.Move.from_uci(move_uci)
            board.push(move)

            # After AI's move, check if opponent can win material
            opponent_moves = list(board.legal_moves)
            max_material_gain = 0

            for opp_move in opponent_moves:
                board.push(opp_move)
                material_after = self._count_material(board)
                material_change = material_before - material_after

                if material_change > max_material_gain:
                    max_material_gain = material_change

                board.pop()

            # Enhanced learning with Stockfish feedback if available
            is_blunder = False
            stockfish_classification = None
            cp_loss = 0

            if stockfish_analysis and i in stockfish_analysis:
                sf_data = stockfish_analysis[i]
                cp_loss = sf_data['cp_loss']
                stockfish_classification = sf_data['classification']

                # Stockfish says it's a blunder/mistake
                if stockfish_classification in ['blunder', 'critical_blunder', 'mistake']:
                    is_blunder = True
                    max_material_gain = max(max_material_gain, cp_loss / 100.0)  # Convert cp to pawns

            # Record if significant material loss OR Stockfish identifies blunder
            if max_material_gain >= 3 or is_blunder:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO learned_mistakes
                    (fen_before, move_made, material_lost, game_id, move_number, times_seen)
                    VALUES (?, ?, ?, ?, ?, COALESCE((SELECT times_seen + 1 FROM learned_mistakes
                                                     WHERE fen_before = ? AND move_made = ?), 1))
                ''', (fen_before, move_uci, max_material_gain, game_id, i+1, fen_before, move_uci))

                if stockfish_classification:
                    logger.debug(f"Learned mistake: {move_san} ({stockfish_classification}, -{cp_loss}cp)")
                else:
                    logger.debug(f"Learned mistake: {move_san} loses {max_material_gain} material")

                # NEW: Extract abstract patterns from this mistake
                patterns = self.pattern_engine.extract_patterns_from_mistake(
                    fen_before, move_uci, max_material_gain
                )
                if patterns:
                    pattern_types = [p['type'] for p in patterns]
                    logger.debug(f"  Patterns: {', '.join(pattern_types)}")
                    all_patterns_in_game.extend(patterns)  # Track for outcome learning

        # Update patterns based on game outcome (LOSS)
        if all_patterns_in_game:
            self.pattern_engine.update_patterns_from_game_outcome(all_patterns_in_game, 'loss')

    def _learn_from_win(self, game_id: int, moves: List[Tuple], ai_color: chess.Color,
                       stockfish_analysis: Optional[dict] = None):
        """
        Extract successful tactics from winning game

        Args:
            stockfish_analysis: Optional Stockfish feedback for enhanced learning
        """
        board = chess.Board()
        all_patterns_in_game = []  # Track patterns even in wins

        for i, (fen_before, move_uci, move_san) in enumerate(moves):
            try:
                board.set_fen(fen_before)
            except:
                continue

            if board.turn != ai_color:
                board.push(chess.Move.from_uci(move_uci))
                continue

            # Check if this move gained material
            material_before = self._count_material(board)

            move = chess.Move.from_uci(move_uci)

            # Check for captures
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    board.push(move)
                    material_after = self._count_material(board)
                    material_gain = material_after - material_before

                    if material_gain >= 1:  # Gained at least a pawn
                        # Record successful tactic
                        self.cursor.execute('''
                            INSERT OR REPLACE INTO learned_tactics
                            (fen_before, move_sequence, material_gained, tactic_type,
                             who_played, game_id, move_number, times_seen, success_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?,
                                   COALESCE((SELECT times_seen + 1 FROM learned_tactics
                                            WHERE fen_before = ? AND move_sequence = ?), 1),
                                   1.0)
                        ''', (fen_before, move_uci, material_gain, 'capture', 'AI',
                              game_id, i+1, fen_before, move_uci))

                        logger.debug(f"Learned tactic: {move_san} gains {material_gain} material")

                        # Extract patterns even from winning moves
                        # (to see if we won DESPITE bad patterns or because of avoiding them)
                        patterns = self.pattern_engine.extract_patterns_from_mistake(fen_before, move_uci, 0.0)
                        if patterns:
                            all_patterns_in_game.extend(patterns)

                    board.pop()
                else:
                    board.push(move)
            else:
                board.push(move)

        # Update patterns based on game outcome (WIN)
        if all_patterns_in_game:
            self.pattern_engine.update_patterns_from_game_outcome(all_patterns_in_game, 'win')

    def _count_material(self, board: chess.Board) -> float:
        """Count material for current player"""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                material += values.get(piece.piece_type, 0)

        return material

    def get_learning_stats(self) -> dict:
        """Get statistics on learned patterns"""
        self.cursor.execute('SELECT COUNT(*) FROM learned_mistakes')
        mistakes = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT COUNT(*) FROM learned_tactics')
        tactics = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT COUNT(*) FROM games WHERE ai_color IS NOT NULL')
        games = self.cursor.fetchone()[0]

        return {'mistakes_learned': mistakes, 'tactics_learned': tactics, 'games_played': games}


def test_learning_ai(num_sessions: int = 3, games_per_session: int = 3):
    """
    Test AI learning over multiple sessions

    Each session:
    - Plays N games
    - Records to database
    - Shows improvement metrics
    """
    print("=" * 70)
    print("LEARNING AI TEST - WITH CLUSTERING AND DEPTH 3 SEARCH")
    print("=" * 70)

    tracker = LearningGameTracker()

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    engine.configure({"Skill Level": 0})

    all_results = []

    for session_num in range(1, num_sessions + 1):
        print(f"\n{'=' * 70}")
        print(f"SESSION {session_num}/{num_sessions}")
        print("=" * 70)

        # Create AI with depth 3 (plans 3 moves ahead!)
        ai = ClusteredIntegratedAI(
            search_depth=3,  # DEPTH 3!
            enable_opponent_prediction=False,
            enable_pattern_learning=True,
            enable_clustering=True,
            time_limit_per_move=30.0  # Give more time for depth 3
        )

        session_results = {'wins': 0, 'draws': 0, 'losses': 0}

        for game_num in range(1, games_per_session + 1):
            board = chess.Board()
            ai_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
            move_history = []
            move_count = 0
            max_moves = 150

            color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"
            print(f"\nGame {game_num}/{games_per_session} - AI={color_str}: ", end='', flush=True)

            while not board.is_game_over() and move_count < max_moves:
                move_count += 1
                fen_before = board.fen()

                if board.turn == ai_color:
                    # AI move
                    move_uci, score = ai.find_best_move(board.fen())
                    if move_uci == "0000":
                        break
                    move = chess.Move.from_uci(move_uci)
                    move_san = board.san(move)
                    board.push(move)
                    move_history.append((fen_before, move_uci, move_san))

                else:
                    # Stockfish move
                    result = engine.play(board, chess.engine.Limit(time=0.5))
                    move_san = board.san(result.move)
                    board.push(result.move)
                    move_history.append((fen_before, result.move.uci(), move_san))

                if move_count % 20 == 0:
                    print('.', end='', flush=True)

            # Determine result
            if board.is_checkmate():
                if board.turn != ai_color:
                    session_results['wins'] += 1
                    result = 'win'
                    print(" WIN!")
                else:
                    session_results['losses'] += 1
                    result = 'loss'
                    print(" LOSS")
            else:
                session_results['draws'] += 1
                result = 'draw'
                print(" DRAW")

            # CRITICAL: Record game for learning!
            tracker.record_game(board, ai_color, result, move_history)

        # Session results
        score = (session_results['wins'] + 0.5 * session_results['draws']) / games_per_session * 100
        print(f"\nSession Results: {session_results['wins']}W {session_results['draws']}D {session_results['losses']}L (Score: {score:.1f}%)")

        # Learning stats
        stats = tracker.get_learning_stats()
        print(f"\nLearning Database:")
        print(f"  Total games:       {stats['games_played']}")
        print(f"  Mistakes learned:  {stats['mistakes_learned']}")
        print(f"  Tactics learned:   {stats['tactics_learned']}")

        all_results.append(session_results)

    engine.quit()

    # Final analysis
    print("\n" + "=" * 70)
    print("LEARNING PROGRESSION")
    print("=" * 70)
    for i, results in enumerate(all_results, 1):
        score = (results['wins'] + 0.5 * results['draws']) / games_per_session * 100
        print(f"Session {i}: {results['wins']}W {results['draws']}D {results['losses']}L ({score:.1f}%)")

    # Check for improvement
    if len(all_results) >= 2:
        first_score = (all_results[0]['wins'] + 0.5 * all_results[0]['draws']) / games_per_session * 100
        last_score = (all_results[-1]['wins'] + 0.5 * all_results[-1]['draws']) / games_per_session * 100
        improvement = last_score - first_score

        print(f"\n📈 Score change: {first_score:.1f}% → {last_score:.1f}% ({improvement:+.1f}%)")
        if improvement > 0:
            print("✅ AI is improving!")
        elif improvement < 0:
            print("⚠️  AI is getting worse (needs more training)")
        else:
            print("➡️  No change yet")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_learning_ai(num_sessions=3, games_per_session=3)
