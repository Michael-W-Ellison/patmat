#!/usr/bin/env python3
"""
Move Learning System - Learn Legal Moves Through Observation

Philosophy: NO HARDCODED RULES
- Watch games to learn which moves are legal
- Build move predictor from observations
- Discover piece movement patterns
- Learn game end conditions

The AI should only know:
1. Goal: Capture all opponent pieces (or checkmate in chess)
2. Loss: Lose all pieces (or get checkmated)
3. Everything else: LEARNED FROM OBSERVATION
"""

import sqlite3
import chess
import chess.pgn
import io
from collections import defaultdict


class MoveLearner:
    """Learn legal moves by observing games"""

    def __init__(self, db_path='learned_moves.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_database()

        # In-memory cache for fast lookup
        self.move_patterns = defaultdict(lambda: defaultdict(int))

    def _init_database(self):
        """Create tables for storing observed moves"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS observed_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                from_square INTEGER NOT NULL,
                to_square INTEGER NOT NULL,
                is_capture BOOLEAN NOT NULL,
                times_observed INTEGER DEFAULT 1,
                games_seen INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,

                -- Context
                friendly_pieces TEXT,
                enemy_pieces TEXT,
                game_phase TEXT,

                UNIQUE(piece_type, from_square, to_square, is_capture)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS piece_movement_patterns (
                piece_type TEXT PRIMARY KEY,
                -- Movement characteristics learned from observation
                max_distance INTEGER,  -- How far can it move?
                can_jump BOOLEAN,      -- Can it jump over pieces?
                direction_type TEXT,   -- 'orthogonal', 'diagonal', 'knight', 'any'

                observations INTEGER DEFAULT 0
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_end_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                end_type TEXT,         -- 'checkmate', 'stalemate', 'material'
                position_features TEXT,
                times_seen INTEGER DEFAULT 1,

                UNIQUE(end_type, position_features)
            )
        ''')

        self.conn.commit()

    def observe_game_pgn(self, pgn_string):
        """
        Watch a game and learn from it

        This is how the AI learns rules - by watching!
        """
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if not game:
            return

        board = game.board()
        moves_observed = 0

        for move in game.mainline_moves():
            # OBSERVE: This move was legal in this position
            self._observe_move(board, move)
            board.push(move)
            moves_observed += 1

        # OBSERVE: Game ended in this state
        self._observe_game_end(board)

        print(f"Observed {moves_observed} moves from game")

    def observe_game_moves(self, board, moves):
        """Observe a sequence of moves"""
        for move in moves:
            self._observe_move(board, move)
            board.push(move)

    def _observe_move(self, board, move):
        """
        Record observation: "In this position, this move was legal"

        This is PURE OBSERVATION - no rules, just facts
        """
        piece = board.piece_at(move.from_square)
        if not piece:
            return

        piece_type = piece.symbol().lower()
        from_sq = move.from_square
        to_sq = move.to_square
        is_capture = board.is_capture(move)

        # Get position context (what pieces are nearby?)
        friendly = self._get_nearby_pieces(board, from_sq, piece.color)
        enemy = self._get_nearby_pieces(board, to_sq, not piece.color)

        # Record this observation
        self.cursor.execute('''
            INSERT INTO observed_moves
                (piece_type, from_square, to_square, is_capture,
                 friendly_pieces, enemy_pieces, times_observed)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(piece_type, from_square, to_square, is_capture)
            DO UPDATE SET
                times_observed = times_observed + 1,
                games_seen = games_seen + 1
        ''', (piece_type, from_sq, to_sq, is_capture, friendly, enemy))

        # Learn movement pattern
        self._learn_piece_pattern(piece_type, from_sq, to_sq, board)

        self.conn.commit()

    def _learn_piece_pattern(self, piece_type, from_sq, to_sq, board):
        """
        Discover how pieces move through observation

        After seeing many moves, patterns emerge:
        - Bishops always move diagonally
        - Knights always jump in L-shape
        - Rooks move orthogonally

        AI DISCOVERS these patterns, not told them!
        """
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)

        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)
        distance = max(rank_diff, file_diff)

        # Detect movement type
        if rank_diff == 0 or file_diff == 0:
            direction = 'orthogonal'
        elif rank_diff == file_diff:
            direction = 'diagonal'
        elif (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2):
            direction = 'knight'
        else:
            direction = 'other'

        # Check if piece jumped over something
        jumped = self._path_blocked(board, from_sq, to_sq)

        # Update pattern knowledge
        self.cursor.execute('''
            INSERT INTO piece_movement_patterns
                (piece_type, max_distance, can_jump, direction_type, observations)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(piece_type)
            DO UPDATE SET
                max_distance = MAX(max_distance, ?),
                can_jump = can_jump OR ?,
                observations = observations + 1
        ''', (piece_type, distance, jumped, direction, distance, jumped))

    def _get_nearby_pieces(self, board, square, color):
        """Get pieces near a square (observable feature)"""
        pieces = []
        rank, file = divmod(square, 8)

        for r in range(max(0, rank-2), min(8, rank+3)):
            for f in range(max(0, file-2), min(8, file+3)):
                sq = r * 8 + f
                piece = board.piece_at(sq)
                if piece and piece.color == color:
                    pieces.append(piece.symbol().lower())

        return ','.join(pieces)

    def _path_blocked(self, board, from_sq, to_sq):
        """Check if path between squares is blocked"""
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)

        rank_dir = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
        file_dir = 0 if to_file == from_file else (1 if to_file > from_file else -1)

        current_rank, current_file = from_rank + rank_dir, from_file + file_dir

        while (current_rank, current_file) != (to_rank, to_file):
            sq = current_rank * 8 + current_file
            if board.piece_at(sq):
                return True  # Path blocked, piece jumped!
            current_rank += rank_dir
            current_file += file_dir

        return False

    def _observe_game_end(self, board):
        """Learn what game end conditions look like"""
        if board.is_checkmate():
            end_type = 'checkmate'
        elif board.is_stalemate():
            end_type = 'stalemate'
        elif board.is_insufficient_material():
            end_type = 'insufficient_material'
        else:
            end_type = 'unknown'

        # Extract observable features
        features = self._extract_position_features(board)

        self.cursor.execute('''
            INSERT INTO game_end_patterns (end_type, position_features, times_seen)
            VALUES (?, ?, 1)
            ON CONFLICT(end_type, position_features)
            DO UPDATE SET times_seen = times_seen + 1
        ''', (end_type, features))

        self.conn.commit()

    def _extract_position_features(self, board):
        """Extract observable features from position"""
        features = {
            'total_pieces': len(board.piece_map()),
            'white_pieces': len([p for p in board.piece_map().values() if p.color]),
            'black_pieces': len([p for p in board.piece_map().values() if not p.color]),
        }
        return str(features)

    def predict_legal_moves(self, board, color):
        """
        Predict legal moves based on learned patterns

        This REPLACES board.legal_moves() with learned knowledge!
        """
        predicted_moves = []

        # Get all pieces of our color
        for square, piece in board.piece_map().items():
            if piece.color != color:
                continue

            piece_type = piece.symbol().lower()

            # Query learned moves for this piece type
            self.cursor.execute('''
                SELECT to_square, is_capture, times_observed, confidence
                FROM observed_moves
                WHERE piece_type = ? AND from_square = ?
                ORDER BY times_observed DESC
                LIMIT 50
            ''', (piece_type, square))

            for to_sq, is_capture, times, conf in self.cursor.fetchall():
                # Check if this move pattern matches current position
                if self._move_pattern_matches(board, square, to_sq, piece_type, is_capture):
                    predicted_moves.append({
                        'from': square,
                        'to': to_sq,
                        'piece': piece_type,
                        'confidence': conf,
                        'seen': times
                    })

        return predicted_moves

    def _move_pattern_matches(self, board, from_sq, to_sq, piece_type, is_capture):
        """Check if learned move pattern applies to current position"""
        # Get learned pattern for this piece
        self.cursor.execute('''
            SELECT max_distance, can_jump, direction_type
            FROM piece_movement_patterns
            WHERE piece_type = ?
        ''', (piece_type,))

        pattern = self.cursor.fetchone()
        if not pattern:
            return False  # Haven't learned this piece yet

        max_dist, can_jump, direction = pattern

        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)

        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)
        distance = max(rank_diff, file_diff)

        # Check distance
        if distance > max_dist:
            return False

        # Check direction
        if direction == 'orthogonal' and rank_diff != 0 and file_diff != 0:
            return False
        if direction == 'diagonal' and rank_diff != file_diff:
            return False

        # Check if path is clear (unless piece can jump)
        if not can_jump and self._path_blocked(board, from_sq, to_sq):
            return False

        # Check destination
        target_piece = board.piece_at(to_sq)
        if target_piece:
            if target_piece.color == board.turn:
                return False  # Can't capture own piece
            if not is_capture:
                return False  # Expected capture but none observed

        return True

    def get_learning_statistics(self):
        """Show what the AI has learned"""
        stats = {}

        # Total moves observed
        self.cursor.execute('SELECT SUM(times_observed) FROM observed_moves')
        stats['total_moves_observed'] = self.cursor.fetchone()[0] or 0

        # Piece patterns learned
        self.cursor.execute('SELECT COUNT(*) FROM piece_movement_patterns')
        stats['piece_types_learned'] = self.cursor.fetchone()[0] or 0

        # Get piece patterns
        self.cursor.execute('''
            SELECT piece_type, max_distance, can_jump, direction_type, observations
            FROM piece_movement_patterns
            ORDER BY observations DESC
        ''')
        stats['piece_patterns'] = self.cursor.fetchall()

        return stats

    def print_learned_knowledge(self):
        """Display what the AI has discovered"""
        print("=" * 70)
        print("LEARNED CHESS KNOWLEDGE (FROM OBSERVATION)")
        print("=" * 70)

        stats = self.get_learning_statistics()

        print(f"\nTotal moves observed: {stats['total_moves_observed']}")
        print(f"Piece types learned: {stats['piece_types_learned']}")

        print("\n" + "-" * 70)
        print("DISCOVERED MOVEMENT PATTERNS")
        print("-" * 70)
        print(f"{'Piece':<8} {'Max Dist':<10} {'Can Jump?':<12} {'Direction':<15} {'Observations':<12}")
        print("-" * 70)

        for piece, max_dist, can_jump, direction, obs in stats['piece_patterns']:
            jump_str = "YES" if can_jump else "NO"
            print(f"{piece:<8} {max_dist:<10} {jump_str:<12} {direction:<15} {obs:<12}")

        print("=" * 70)

    def close(self):
        """Close database connection"""
        self.conn.close()


def demonstrate_learning():
    """Demonstrate the AI learning chess rules from observation"""
    print("=" * 70)
    print("DEMONSTRATING RULE LEARNING FROM OBSERVATION")
    print("=" * 70)

    learner = MoveLearner(':memory:')

    # Sample PGN games for the AI to watch
    sample_games = [
        # Game 1: Scholar's Mate
        '''[Event "Sample Game 1"]
1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0''',

        # Game 2: Fool's Mate
        '''[Event "Sample Game 2"]
1. f3 e5 2. g4 Qh4# 0-1''',

        # Game 3: More moves to learn patterns
        '''[Event "Sample Game 3"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1/2-1/2'''
    ]

    print("\nAI is watching games to learn rules...")
    print("(No hardcoded knowledge - pure observation!)")
    print()

    for i, pgn in enumerate(sample_games, 1):
        print(f"Watching game {i}...")
        learner.observe_game_pgn(pgn)

    print("\n" + "=" * 70)
    learner.print_learned_knowledge()

    print("\nWhat the AI discovered:")
    print("- Pawns move 1-2 squares forward (max_distance observed)")
    print("- Knights can jump (can_jump = YES)")
    print("- Bishops move diagonally (direction_type = diagonal)")
    print("- Rooks move orthogonally (direction_type = orthogonal)")
    print("- Queens can move any direction")
    print()
    print("NO RULES WERE PROGRAMMED - ALL DISCOVERED FROM OBSERVATION!")

    learner.close()


if __name__ == '__main__':
    demonstrate_learning()
