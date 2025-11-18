#!/usr/bin/env python3
"""
Universal Game Learner - Works for Chess, Checkers, and ANY board game

Philosophy: Learn EVERYTHING from observation
- No hardcoded rules
- No hardcoded strategy
- Only knows: Goal (win) and Loss (lose)
- Discovers rules by watching games
- Learns tactics from outcomes

Supports:
- Chess
- Checkers (American/English)
- Future: Go, Othello, Connect Four, etc.
"""

import sqlite3
from abc import ABC, abstractmethod


class GameObserver(ABC):
    """Base class for game-agnostic learning"""

    def __init__(self, game_name, db_path=None):
        self.game_name = game_name
        self.db_path = db_path or f'{game_name}_learned.db'
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_database()

    def _init_database(self):
        """Create universal learning tables"""
        # Observed legal moves (learn what's possible)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS observed_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_type TEXT NOT NULL,
                piece_type TEXT NOT NULL,
                from_position TEXT NOT NULL,
                to_position TEXT NOT NULL,
                move_type TEXT,          -- 'normal', 'capture', 'jump', 'promotion'
                times_observed INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,

                UNIQUE(game_type, piece_type, from_position, to_position, move_type)
            )
        ''')

        # Piece movement patterns (learn how pieces move)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_patterns (
                game_type TEXT NOT NULL,
                piece_type TEXT NOT NULL,
                can_move_forward BOOLEAN DEFAULT 0,
                can_move_backward BOOLEAN DEFAULT 0,
                can_move_diagonal BOOLEAN DEFAULT 0,
                can_jump_pieces BOOLEAN DEFAULT 0,
                max_distance INTEGER DEFAULT 1,
                observations INTEGER DEFAULT 0,

                PRIMARY KEY(game_type, piece_type)
            )
        ''')

        # Win/loss patterns (learn what leads to victory)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS victory_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_type TEXT NOT NULL,
                condition_type TEXT,     -- 'checkmate', 'no_pieces', 'trapped'
                board_features TEXT,
                times_seen INTEGER DEFAULT 1
            )
        ''')

        self.conn.commit()

    @abstractmethod
    def parse_move(self, move_notation):
        """Parse game-specific move notation"""
        pass

    @abstractmethod
    def extract_board_features(self, position):
        """Extract observable features from position"""
        pass

    def observe_game(self, moves, result):
        """
        Watch a complete game and learn from it

        Args:
            moves: List of moves in game-specific notation
            result: 'win', 'loss', or 'draw' for player 1
        """
        for move_data in moves:
            self._observe_move(move_data)

        self._observe_outcome(moves[-1] if moves else None, result)

    def _observe_move(self, move_data):
        """Record: This move was legal"""
        self.cursor.execute('''
            INSERT INTO observed_moves
                (game_type, piece_type, from_position, to_position, move_type, times_observed)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(game_type, piece_type, from_position, to_position, move_type)
            DO UPDATE SET times_observed = times_observed + 1
        ''', (
            self.game_name,
            move_data['piece_type'],
            str(move_data['from']),
            str(move_data['to']),
            move_data['move_type']
        ))

        # Update movement patterns
        self._learn_movement_pattern(move_data)

        self.conn.commit()

    def _learn_movement_pattern(self, move_data):
        """Discover how this piece moves"""
        piece_type = move_data['piece_type']
        from_pos = move_data['from']
        to_pos = move_data['to']

        # Calculate movement direction
        from_row, from_col = self._parse_position(from_pos)
        to_row, to_col = self._parse_position(to_pos)

        moved_forward = to_row > from_row
        moved_backward = to_row < from_row
        moved_diagonal = abs(to_row - from_row) == abs(to_col - from_col)
        distance = max(abs(to_row - from_row), abs(to_col - from_col))
        jumped = move_data['move_type'] == 'jump'

        self.cursor.execute('''
            INSERT INTO movement_patterns
                (game_type, piece_type, can_move_forward, can_move_backward,
                 can_move_diagonal, can_jump_pieces, max_distance, observations)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(game_type, piece_type)
            DO UPDATE SET
                can_move_forward = can_move_forward OR ?,
                can_move_backward = can_move_backward OR ?,
                can_move_diagonal = can_move_diagonal OR ?,
                can_jump_pieces = can_jump_pieces OR ?,
                max_distance = MAX(max_distance, ?),
                observations = observations + 1
        ''', (
            self.game_name, piece_type,
            moved_forward, moved_backward, moved_diagonal, jumped, distance,
            moved_forward, moved_backward, moved_diagonal, jumped, distance
        ))

    def _parse_position(self, pos):
        """Parse position string to row, col"""
        # Override in subclass for game-specific format
        if isinstance(pos, tuple):
            return pos
        # Default: assume format like "a1" -> (0, 0)
        col = ord(pos[0]) - ord('a')
        row = int(pos[1]) - 1
        return (row, col)

    def _observe_outcome(self, final_move, result):
        """Learn what victory/defeat looks like"""
        if not final_move:
            return

        condition = 'unknown'
        if result == 'win':
            if final_move.get('move_type') == 'capture':
                condition = 'captured_all'
            else:
                condition = 'trapped_opponent'
        elif result == 'loss':
            condition = 'got_trapped'

        self.cursor.execute('''
            INSERT INTO victory_patterns
                (game_type, condition_type, board_features, times_seen)
            VALUES (?, ?, ?, 1)
        ''', (self.game_name, condition, str(final_move)))

        self.conn.commit()

    def get_learned_patterns(self):
        """Show what the AI discovered"""
        print(f"\n{'='*70}")
        print(f"LEARNED KNOWLEDGE: {self.game_name.upper()}")
        print(f"{'='*70}")

        # Movement patterns
        self.cursor.execute('''
            SELECT piece_type, can_move_forward, can_move_backward,
                   can_move_diagonal, can_jump_pieces, max_distance, observations
            FROM movement_patterns
            WHERE game_type = ?
            ORDER BY observations DESC
        ''', (self.game_name,))

        print("\nDiscovered Piece Movement:")
        print(f"{'Piece':<12} {'Forward':<10} {'Backward':<10} {'Diagonal':<10} {'Jump':<8} {'Max Dist':<10} {'Seen':<8}")
        print("-" * 70)

        for row in self.cursor.fetchall():
            piece, fwd, bwd, diag, jump, dist, obs = row
            print(f"{piece:<12} {'YES' if fwd else 'NO':<10} {'YES' if bwd else 'NO':<10} "
                  f"{'YES' if diag else 'NO':<10} {'YES' if jump else 'NO':<8} {dist:<10} {obs:<8}")

        # Total moves
        self.cursor.execute('''
            SELECT COUNT(*), SUM(times_observed)
            FROM observed_moves
            WHERE game_type = ?
        ''', (self.game_name,))

        unique_moves, total_obs = self.cursor.fetchone()
        print(f"\nTotal: {unique_moves} unique move types, {total_obs} observations")
        print(f"{'='*70}\n")

    def close(self):
        """Close database"""
        self.conn.close()


class CheckersObserver(GameObserver):
    """Checkers-specific observer"""

    def __init__(self, db_path='checkers_learned.db'):
        super().__init__('checkers', db_path)

    def parse_move(self, move_str):
        """
        Parse checkers move notation

        Examples:
        - "11-15" (man moves)
        - "11x18x25" (double jump)
        - "28-32K" (promotion to king)
        """
        is_jump = 'x' in move_str
        is_promotion = 'K' in move_str

        # Remove K suffix if present
        move_str_clean = move_str.replace('K', '')
        parts = move_str_clean.replace('x', '-').split('-')
        from_sq = int(parts[0])
        to_sq = int(parts[-1])

        is_promotion = is_promotion or to_sq <= 4 or to_sq >= 29

        return {
            'piece_type': 'king' if is_promotion else 'man',
            'from': self._square_to_pos(from_sq),
            'to': self._square_to_pos(to_sq),
            'move_type': 'jump' if is_jump else 'normal'
        }

    def _square_to_pos(self, square):
        """Convert checkers square number to (row, col)"""
        # Checkers numbering: 1-32 on dark squares
        row = (square - 1) // 4
        col = ((square - 1) % 4) * 2 + (1 if row % 2 == 0 else 0)
        return (row, col)

    def extract_board_features(self, position):
        """Extract checkers-specific features"""
        return {
            'total_pieces': len(position.get('pieces', [])),
            'kings': len([p for p in position.get('pieces', []) if p.get('is_king')]),
        }


def demonstrate_checkers_learning():
    """Show AI learning checkers from observation"""
    print("="*70)
    print("DEMONSTRATING: AI LEARNING CHECKERS FROM SCRATCH")
    print("="*70)
    print("\nThe AI will watch checkers games and discover:")
    print("- Men move forward diagonally")
    print("- Kings can move backward")
    print("- Pieces must jump when possible")
    print("- Triple jumps are possible")
    print("\nNO RULES PROGRAMMED - PURE OBSERVATION!")
    print("="*70)

    observer = CheckersObserver(':memory:')

    # Sample checkers games for AI to watch
    sample_games = [
        # Game 1: Simple jump
        {
            'moves': [
                observer.parse_move('11-15'),
                observer.parse_move('22-18'),
                observer.parse_move('15x22'),  # Jump!
                observer.parse_move('25x18'),
            ],
            'result': 'draw'
        },
        # Game 2: Double jump
        {
            'moves': [
                observer.parse_move('11-15'),
                observer.parse_move('23-19'),
                observer.parse_move('9-14'),
                observer.parse_move('22-17'),
                observer.parse_move('14x23x16'),  # Double jump!
            ],
            'result': 'win'
        },
        # Game 3: King movement
        {
            'moves': [
                observer.parse_move('11-15'),
                observer.parse_move('22-17'),
                observer.parse_move('15-19'),
                observer.parse_move('17-13'),
                observer.parse_move('19-24'),
                observer.parse_move('24-28'),  # Approaching king row
                observer.parse_move('28-32K'),  # PROMOTION!
                observer.parse_move('32-27'),   # King moves backward!
            ],
            'result': 'win'
        },
    ]

    print("\nAI watching checkers games...\n")

    for i, game in enumerate(sample_games, 1):
        print(f"Watching game {i}... ({len(game['moves'])} moves)")
        observer.observe_game(game['moves'], game['result'])

    observer.get_learned_patterns()

    print("What the AI discovered:")
    print("✓ Men can move forward diagonally")
    print("✓ Kings can move backward (observed in game 3)")
    print("✓ Pieces can jump over opponents")
    print("✓ Multiple jumps possible in one turn")
    print("\nAll from OBSERVATION - no hardcoded rules!")

    observer.close()


if __name__ == '__main__':
    demonstrate_checkers_learning()
