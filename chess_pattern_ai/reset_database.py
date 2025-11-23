#!/usr/bin/env python3
"""
Reset database to clean state with new schema
"""

import sqlite3
import os
import sys

def reset_database(db_path='headless_training.db'):
    """Delete old database and create new one with correct schema"""

    # Delete all database-related files
    for suffix in ['', '-journal', '-wal', '-shm']:
        file_path = db_path + suffix
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Deleted {file_path}")

    print(f"\nCreating fresh database: {db_path}")

    # Create new database with correct schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE learned_move_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Observable move characteristics (NO game-specific stages!)
            piece_type TEXT NOT NULL,
            move_category TEXT NOT NULL,
            distance_from_start INTEGER,

            -- Observable game state (allows discovering draw-causing patterns)
            repetition_count INTEGER DEFAULT 0,
            moves_since_progress INTEGER DEFAULT 0,
            total_material_level TEXT DEFAULT 'medium',

            -- Outcome tracking
            times_seen INTEGER DEFAULT 0,
            games_won INTEGER DEFAULT 0,
            games_lost INTEGER DEFAULT 0,
            games_drawn INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,

            -- Score-based tracking
            total_score REAL DEFAULT 0.0,
            avg_score REAL DEFAULT 0.0,

            -- Statistical confidence
            confidence REAL DEFAULT 0.0,

            -- Learned priority
            priority_score REAL DEFAULT 0.0,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(piece_type, move_category, distance_from_start,
                   repetition_count, moves_since_progress, total_material_level)
        )
    ''')

    cursor.execute('''
        CREATE INDEX idx_move_pattern_priority
        ON learned_move_patterns(priority_score DESC, confidence DESC)
    ''')

    conn.commit()
    conn.close()

    print("✓ Created new database with enhanced pattern schema")
    print("\nNew observable features:")
    print("  - repetition_count: 0, 1, or 2")
    print("  - moves_since_progress: 0-5 (halfmove clock / 10)")
    print("  - total_material_level: low, medium, high")
    print("\nAI will now discover draw patterns through observation!")

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'chess_pattern_ai/headless_training.db'
    reset_database(db_path)
