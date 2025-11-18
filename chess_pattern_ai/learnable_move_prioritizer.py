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
    - Game phase when move was made

    NO hardcoded square knowledge - learns from outcomes
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

                -- Observable move characteristics
                piece_type TEXT NOT NULL,           -- 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
                move_category TEXT NOT NULL,        -- 'capture', 'check', 'capture_check', 'quiet', 'development'
                distance_from_start INTEGER,        -- How many ranks moved from starting area (0-8)
                game_phase TEXT,                    -- 'opening', 'middlegame', 'endgame'

                -- Outcome tracking
                times_seen INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                games_lost INTEGER DEFAULT 0,
                games_drawn INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,

                -- Statistical confidence
                confidence REAL DEFAULT 0.0,

                -- Learned priority (higher = search this type of move first)
                priority_score REAL DEFAULT 0.0,

                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(piece_type, move_category, distance_from_start, game_phase)
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
        Classify a move by observable characteristics (NO hardcoded square knowledge)

        Returns:
            Dict with: piece_type, move_category, distance_from_start, game_phase
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

        # Observable: Game phase based on material and moves
        game_phase = self._detect_game_phase(board)

        return {
            'piece_type': piece_type,
            'move_category': move_category,
            'distance_from_start': distance_from_start,
            'game_phase': game_phase
        }

    def _detect_game_phase(self, board: 'chess.Board') -> str:
        """Detect game phase by observable features (material count, move number)"""
        if not CHESS_AVAILABLE:
            return 'middlegame'

        # Count pieces (observable)
        piece_count = len(board.piece_map())
        move_number = board.fullmove_number

        # Opening: Many pieces, early moves
        if move_number <= 12 and piece_count >= 28:
            return 'opening'
        # Endgame: Few pieces
        elif piece_count <= 14:
            return 'endgame'
        # Middlegame: Everything else
        else:
            return 'middlegame'

    def record_game_moves(self, moves: List[Tuple[str, str, str]],
                         ai_color: 'chess.Color', result: str):
        """
        Record moves from a game to learn which types lead to wins

        Args:
            moves: List of (fen_before, move_uci, move_san) tuples
            ai_color: Color AI played
            result: 'win', 'loss', or 'draw'
        """
        if not CHESS_AVAILABLE:
            logger.warning("python-chess not available, cannot record moves")
            return

        board = chess.Board()

        for fen_before, move_uci, move_san in moves:
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

            # Update statistics for this move type
            self._update_move_statistics(
                characteristics['piece_type'],
                characteristics['move_category'],
                characteristics['distance_from_start'],
                characteristics['game_phase'],
                result
            )

            board.push(move)

        self.conn.commit()

    def _update_move_statistics(self, piece_type: str, move_category: str,
                                distance: int, phase: str, result: str):
        """Update win/loss statistics for a move pattern"""

        # Get current stats
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ?
              AND distance_from_start = ? AND game_phase = ?
        ''', (piece_type, move_category, distance, phase))

        row = self.cursor.fetchone()

        if row:
            times_seen, won, lost, drawn = row
            times_seen += 1
        else:
            times_seen, won, lost, drawn = 1, 0, 0, 0

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

        # Confidence increases with more observations
        confidence = min(1.0, total_games / 50.0)  # Max confidence at 50+ games

        # Priority score: win_rate weighted by confidence
        # High win rate + high confidence = high priority
        priority_score = win_rate * confidence * 100

        # Insert or update
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase)
            DO UPDATE SET
                times_seen = ?,
                games_won = ?,
                games_lost = ?,
                games_drawn = ?,
                win_rate = ?,
                confidence = ?,
                priority_score = ?,
                updated_at = datetime('now')
        ''', (
            piece_type, move_category, distance, phase,
            times_seen, won, lost, drawn, win_rate, confidence, priority_score,
            times_seen, won, lost, drawn, win_rate, confidence, priority_score
        ))

    def _load_priorities(self):
        """Load learned move priorities from database"""
        self.cursor.execute('''
            SELECT piece_type, move_category, distance_from_start, game_phase,
                   priority_score, win_rate, confidence
            FROM learned_move_patterns
            WHERE confidence > 0.1
            ORDER BY priority_score DESC
        ''')

        self.move_priorities = {}
        for row in self.cursor.fetchall():
            piece_type, category, distance, phase, priority, win_rate, confidence = row
            key = (piece_type, category, distance, phase)
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
            characteristics['game_phase']
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
            SELECT piece_type, move_category, distance_from_start, game_phase,
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
            piece, category, dist, phase, seen, win_rate, conf, priority = row
            print(f"  {piece:8s} {category:12s} (dist={dist}, {phase:10s})")
            print(f"    Seen {seen:3d}x | Win rate: {win_rate:.1%} | "
                  f"Conf: {conf:.2f} | Priority: {priority:.1f}")

    prioritizer.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_learnable_prioritizer()
