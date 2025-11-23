#!/usr/bin/env python3
"""
Learnable Move Prioritizer - Discovers Which Move Types Win

Instead of hardcoding "center squares are good", this system:
1. Observes which types of moves appear in winning games
2. Tracks win rates for different move characteristics
3. Uses learned priorities to guide search

Philosophy: Discovers what works through observation, not programming
"""

import sqlite3
import logging
from typing import List, Tuple, Dict, Optional
try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LearnableMovePrioritizer:
    """
    Learns which types of moves lead to wins by observing game outcomes

    Tracks observable move characteristics:
    - Piece type that moved
    - Distance moved from starting area
    - Whether it's a capture, check, or quiet move
    - Observable game state features (repetition, progress, material)

    NO hardcoded square knowledge or game stages - learns from outcomes
    """

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.move_priorities = {}

        self._init_tables()
        self._load_priorities()

    def _init_tables(self):
        """Create tables for learned move patterns"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_move_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Observable move characteristics (NO game-specific stages!)
                piece_type TEXT NOT NULL,           -- 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
                move_category TEXT NOT NULL,        -- 'capture', 'check', 'capture_check', 'quiet', 'development'
                distance_from_start INTEGER,        -- How many ranks moved from starting area (0-8)

                -- Observable game state (allows discovering draw-causing patterns)
                repetition_count INTEGER DEFAULT 0, -- How many times position repeated (0, 1, 2)
                moves_since_progress INTEGER DEFAULT 0, -- Halfmove clock / 10 (0, 1, 2, 3, 4, 5+)
                total_material_level TEXT DEFAULT 'medium', -- 'high', 'medium', 'low'

                -- Outcome tracking (binary win/loss)
                times_seen INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                games_lost INTEGER DEFAULT 0,
                games_drawn INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,

                -- SCORE-BASED tracking (new!)
                total_score REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,

                -- Statistical confidence
                confidence REAL DEFAULT 0.0,

                -- Learned priority (higher = search this type of move first)
                priority_score REAL DEFAULT 0.0,

                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(piece_type, move_category, distance_from_start,
                       repetition_count, moves_since_progress, total_material_level)
            )
        ''')

        # Index for fast lookups
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_move_pattern_priority
            ON learned_move_patterns(priority_score DESC, confidence DESC)
        ''')

        self.conn.commit()
        logger.info("✓ Learnable move prioritizer tables initialized")

    def classify_move(self, board: 'chess.Board', move: 'chess.Move') -> Dict[str, any]:
        """
        Classify a move by observable characteristics (NO hardcoded stages or square knowledge)

        Returns:
            Dict with: piece_type, move_category, distance_from_start,
                       repetition_count, moves_since_progress, total_material_level
        """
        if not CHESS_AVAILABLE:
            return {}

        piece = board.piece_at(move.from_square)
        if not piece:
            return {}

        # Observable: What type of piece moved
        piece_type_map = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight',
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }
        piece_type = piece_type_map.get(piece.piece_type, 'unknown')

        # Observable: Type of move (forcing or quiet)
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)

        if is_capture and is_check:
            move_category = 'capture_check'
        elif is_capture:
            move_category = 'capture'
        elif is_check:
            move_category = 'check'
        else:
            # Check if piece is moving away from starting area (development)
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)

            # For white pieces: starting area is ranks 0-1
            # For black pieces: starting area is ranks 6-7
            if piece.color == chess.WHITE:
                is_development = from_rank <= 1 and to_rank > 1
            else:
                is_development = from_rank >= 6 and to_rank < 6

            move_category = 'development' if is_development else 'quiet'

        # Observable: How far from starting area (observable, not "good" or "bad")
        from_rank = chess.square_rank(move.from_square)
        if piece.color == chess.WHITE:
            # Distance from white's starting area (ranks 0-1)
            distance_from_start = max(0, from_rank - 1)
        else:
            # Distance from black's starting area (ranks 6-7)
            distance_from_start = max(0, 6 - from_rank)

        # OBSERVABLE GAME STATE FEATURES (for discovering draw patterns)

        # 1. Repetition count: How many times has this position occurred?
        # After the move, check if position would be repeated
        board.push(move)
        if board.is_repetition(3):
            repetition_count = 2  # Third repetition (causes draw)
        elif board.is_repetition(2):
            repetition_count = 1  # Second occurrence
        else:
            repetition_count = 0  # First occurrence
        board.pop()

        # 2. Moves since progress: Observable halfmove clock
        # Tracks moves since last capture or pawn move (50-move rule)
        halfmove_clock = board.halfmove_clock
        # Bucket into ranges for pattern recognition
        if halfmove_clock >= 50:
            moves_since_progress = 5  # 50+ moves (draw imminent)
        elif halfmove_clock >= 40:
            moves_since_progress = 4  # 40-49 moves (danger zone)
        elif halfmove_clock >= 30:
            moves_since_progress = 3  # 30-39 moves
        elif halfmove_clock >= 20:
            moves_since_progress = 2  # 20-29 moves
        elif halfmove_clock >= 10:
            moves_since_progress = 1  # 10-19 moves
        else:
            moves_since_progress = 0  # 0-9 moves (fresh)

        # 3. Total material level: Observable piece count and values
        # Allows discovering "low material → draw" pattern
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Standard material values
                values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                total_material += values.get(piece.piece_type, 0)

        # Categorize material level
        # Full board = ~39 points, insufficient material ~= 6 points
        if total_material <= 6:
            total_material_level = 'low'  # Insufficient for checkmate
        elif total_material <= 20:
            total_material_level = 'medium'  # Endgame
        else:
            total_material_level = 'high'  # Opening/middlegame

        return {
            'piece_type': piece_type,
            'move_category': move_category,
            'distance_from_start': distance_from_start,
            'repetition_count': repetition_count,
            'moves_since_progress': moves_since_progress,
            'total_material_level': total_material_level
        }

    def record_game_moves(self, moves: List[Tuple[str, str, str]],
                         ai_color: 'chess.Color', result: str, final_score: float = 0.0):
        """
        Record moves from a game using MOVE-LEVEL SCORING

        CRITICAL FIX: Score each move based on its immediate observable effect,
        NOT the final game outcome. This prevents catastrophic learning where
        the AI learns "normal chess moves lead to stalemate" when actually only
        the LAST move caused the stalemate.

        Args:
            moves: List of (fen_before, move_uci, move_san) tuples
            ai_color: Color AI played
            result: 'win', 'loss', or 'draw'
            final_score: Game score (only used for last move context)
        """
        if not CHESS_AVAILABLE:
            logger.warning("python-chess not available, cannot record moves")
            return

        board = chess.Board()
        move_count = 0

        # Pre-count AI's moves to identify the last one
        ai_move_indices = []
        temp_board = chess.Board()
        for idx, (fen_before, move_uci, move_san) in enumerate(moves):
            try:
                temp_board.set_fen(fen_before)
                if temp_board.turn == ai_color:
                    ai_move_indices.append(idx)
                temp_board.push(chess.Move.from_uci(move_uci))
            except:
                continue

        last_ai_move_idx = ai_move_indices[-1] if ai_move_indices else -1

        for idx, (fen_before, move_uci, move_san) in enumerate(moves):
            try:
                board.set_fen(fen_before)
            except:
                continue

            # Only track AI's moves
            if board.turn != ai_color:
                board.push(chess.Move.from_uci(move_uci))
                continue

            move = chess.Move.from_uci(move_uci)
            characteristics = self.classify_move(board, move)

            if not characteristics:
                board.push(move)
                continue

            is_last_move = (idx == last_ai_move_idx)

            # MOVE-LEVEL SCORING: Calculate immediate observable effect
            move_score = self._calculate_move_score(board, move, ai_color, is_last_move, result)

            # Update statistics for this move type with MOVE score, not GAME score
            self._update_move_statistics(
                characteristics['piece_type'],
                characteristics['move_category'],
                characteristics['distance_from_start'],
                characteristics['repetition_count'],
                characteristics['moves_since_progress'],
                characteristics['total_material_level'],
                result,
                move_score  # Use move score, not final_score!
            )

            board.push(move)

        self.conn.commit()

    def _calculate_move_score(self, board: 'chess.Board', move: 'chess.Move',
                             ai_color: 'chess.Color', is_last_move: bool, game_result: str) -> float:
        """
        Calculate score for a SINGLE move based on immediate observable effects.

        This fixes the catastrophic credit assignment problem where ALL moves
        in a stalemate game were labeled "bad" even if they were good moves.

        Returns:
            Move score based on immediate effects (material, check, stalemate, etc.)
        """
        if not CHESS_AVAILABLE:
            return 0.0

        # Material values for calculating captures
        PIECE_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        score = 0.0

        # 1. MATERIAL CHANGE: Did this move gain or lose material?
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                score += PIECE_VALUES.get(captured_piece.piece_type, 0)

        # 2. CHECK BONUS: Forcing moves are generally good
        board.push(move)
        if board.is_check():
            score += 50  # Small bonus for giving check

        # 3. CATASTROPHIC MOVE PENALTIES: Did THIS move cause a bad outcome?
        if is_last_move:
            if board.is_stalemate():
                # THIS move caused stalemate - massive penalty!
                score -= 10000
            elif board.is_repetition(3):
                # THIS move caused threefold repetition
                score -= 5000
            elif board.is_fifty_moves():
                # THIS move triggered fifty-move draw
                score -= 5000
            elif board.is_insufficient_material():
                # THIS move left insufficient material (bad trade)
                score -= 3000
            elif board.is_checkmate():
                # THIS move delivered checkmate!
                if game_result == 'win':
                    score += 10000
                else:
                    score -= 10000  # Got checkmated

        # 4. MOBILITY BONUS: More legal moves after = good position
        # (Only calculate for non-catastrophic moves)
        if not is_last_move or game_result == 'win':
            legal_moves_after = len(list(board.legal_moves))
            score += legal_moves_after * 2  # Small bonus per legal move

        board.pop()

        return score

    def _update_move_statistics(self, piece_type: str, move_category: str,
                                distance: int,
                                repetition_count: int, moves_since_progress: int,
                                total_material_level: str,
                                result: str, final_score: float):
        """Update win/loss statistics and score for a move pattern"""

        # Get current stats
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn, total_score
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ?
              AND distance_from_start = ?
              AND repetition_count = ? AND moves_since_progress = ?
              AND total_material_level = ?
        ''', (piece_type, move_category, distance,
              repetition_count, moves_since_progress, total_material_level))

        row = self.cursor.fetchone()

        if row:
            times_seen, won, lost, drawn, total_score = row
            times_seen += 1
            total_score += final_score
        else:
            times_seen, won, lost, drawn = 1, 0, 0, 0
            total_score = final_score

        # Update based on result
        if result == 'win':
            won += 1
        elif result == 'loss':
            lost += 1
        else:
            drawn += 1

        # Calculate win rate and confidence
        total_games = won + lost + drawn
        win_rate = won / total_games if total_games > 0 else 0.0
        avg_score = total_score / total_games if total_games > 0 else 0.0

        # Confidence increases with more observations
        confidence = min(1.0, total_games / 50.0)  # Max confidence at 50+ games

        # Priority score: DIFFERENTIAL SCORE-BASED!
        # Moves with high avg_score (material advantage + win bonus) get priority
        #
        # Score ranges (with differential scoring):
        #   Best wins: ~1590 (win + ahead + fast)
        #   Average wins: ~1050 (win + even + slow)
        #   Draws ahead: ~300
        #   Draws behind: ~-300
        #   Losses close: ~-800 (lost but fought well)
        #   Losses crushed: ~-1500 (got destroyed)
        #
        # Normalize -1500 to +1600 → 0 to 100
        normalized_score = (avg_score + 1500) / 31  # -1500 to +1600 → 0 to 100
        priority_score = normalized_score * confidence  # 0-100, confidence-weighted

        # Insert or update
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start,
                 repetition_count, moves_since_progress, total_material_level,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start,
                        repetition_count, moves_since_progress, total_material_level)
            DO UPDATE SET
                times_seen = ?,
                games_won = ?,
                games_lost = ?,
                games_drawn = ?,
                win_rate = ?,
                total_score = ?,
                avg_score = ?,
                confidence = ?,
                priority_score = ?,
                updated_at = datetime('now')
        ''', (
            piece_type, move_category, distance,
            repetition_count, moves_since_progress, total_material_level,
            times_seen, won, lost, drawn, win_rate, total_score, avg_score, confidence, priority_score,
            times_seen, won, lost, drawn, win_rate, total_score, avg_score, confidence, priority_score
        ))

    def _load_priorities(self):
        """Load learned move priorities from database"""
        self.cursor.execute('''
            SELECT piece_type, move_category, distance_from_start,
                   repetition_count, moves_since_progress, total_material_level,
                   priority_score, win_rate, confidence
            FROM learned_move_patterns
            WHERE confidence > 0.1
            ORDER BY priority_score DESC
        ''')

        self.move_priorities = {}
        for row in self.cursor.fetchall():
            piece_type, category, distance, rep_count, moves_progress, mat_level, priority, win_rate, confidence = row
            key = (piece_type, category, distance, rep_count, moves_progress, mat_level)
            self.move_priorities[key] = {
                'priority': priority,
                'win_rate': win_rate,
                'confidence': confidence
            }

        if self.move_priorities:
            logger.info(f"✓ Loaded {len(self.move_priorities)} learned move patterns")

    def get_move_priority(self, board: 'chess.Board', move: 'chess.Move') -> float:
        """
        Get learned priority for a move (higher = better historically)

        Returns:
            Priority score (0-100), or default based on move type
        """
        if not CHESS_AVAILABLE:
            return 50.0

        characteristics = self.classify_move(board, move)
        if not characteristics:
            return 50.0

        key = (
            characteristics['piece_type'],
            characteristics['move_category'],
            characteristics['distance_from_start'],
            characteristics['repetition_count'],
            characteristics['moves_since_progress'],
            characteristics['total_material_level']
        )

        # Check if we've learned about this move type
        if key in self.move_priorities:
            return self.move_priorities[key]['priority']

        # Default priorities for unseen move types (based on forcing nature)
        # These are INITIAL guesses that get refined through learning
        category = characteristics['move_category']
        if category == 'capture_check':
            return 90.0  # Always worth considering
        elif category == 'capture':
            return 70.0
        elif category == 'check':
            return 60.0
        elif category == 'development':
            return 40.0
        else:  # quiet
            return 20.0

    def sort_moves_by_priority(self, board: 'chess.Board',
                               moves: List['chess.Move']) -> List['chess.Move']:
        """
        Sort moves by learned priority (highest priority first)

        This replaces hardcoded square preferences with learned patterns
        """
        if not CHESS_AVAILABLE:
            return moves

        # Get priority for each move
        move_priorities = [(move, self.get_move_priority(board, move)) for move in moves]

        # Sort by priority (highest first)
        move_priorities.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in move_priorities]

    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        self.cursor.execute('''
            SELECT COUNT(*), AVG(confidence), AVG(win_rate)
            FROM learned_move_patterns
            WHERE confidence > 0.1
        ''')

        count, avg_conf, avg_win = self.cursor.fetchone()

        return {
            'patterns_learned': count or 0,
            'avg_confidence': avg_conf or 0.0,
            'avg_win_rate': avg_win or 0.0
        }

    def get_top_patterns(self, limit: int = 10) -> List[Tuple]:
        """Get top performing move patterns"""
        self.cursor.execute('''
            SELECT piece_type, move_category, distance_from_start,
                   times_seen, win_rate, confidence, priority_score
            FROM learned_move_patterns
            WHERE confidence > 0.2
            ORDER BY priority_score DESC
            LIMIT ?
        ''', (limit,))

        return self.cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_learnable_prioritizer():
    """Test the learnable move prioritizer"""
    if not CHESS_AVAILABLE:
        print("python-chess not available, skipping test")
        return

    print("=" * 70)
    print("LEARNABLE MOVE PRIORITIZER TEST")
    print("=" * 70)

    prioritizer = LearnableMovePrioritizer()

    # Test move classification
    board = chess.Board()

    print("\nClassifying moves from starting position:")
    for move in list(board.legal_moves)[:5]:
        characteristics = prioritizer.classify_move(board, move)
        priority = prioritizer.get_move_priority(board, move)
        print(f"  {move.uci():6s}: {characteristics['piece_type']:8s} "
              f"{characteristics['move_category']:12s} "
              f"priority={priority:.1f}")

    # Show learned patterns
    stats = prioritizer.get_statistics()
    print(f"\nLearned Patterns: {stats['patterns_learned']}")
    print(f"Avg Confidence:   {stats['avg_confidence']:.2f}")
    print(f"Avg Win Rate:     {stats['avg_win_rate']:.1%}")

    if stats['patterns_learned'] > 0:
        print("\nTop 5 Move Patterns (by priority):")
        for row in prioritizer.get_top_patterns(5):
            piece, category, dist, seen, win_rate, conf, priority = row
            print(f"  {piece:8s} {category:12s} (dist={dist})")
            print(f"    Seen {seen:3d}x | Win rate: {win_rate:.1%} | "
                  f"Conf: {conf:.2f} | Priority: {priority:.1f}")

    prioritizer.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_learnable_prioritizer()
