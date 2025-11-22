#!/usr/bin/env python3
"""
Migrate pattern database to include observable game state features
that allow the AI to discover draw-causing patterns naturally.
"""

import sqlite3
import sys

def migrate_database(db_path='headless_training.db'):
    """Add observable game state columns to pattern table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Migrating database: {db_path}")

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(learned_move_patterns)")
    columns = [row[1] for row in cursor.fetchall()]

    new_columns = [
        ('repetition_count', 'INTEGER DEFAULT 0'),
        ('moves_since_progress', 'INTEGER DEFAULT 0'),
        ('total_material_level', 'TEXT DEFAULT "medium"')
    ]

    for col_name, col_def in new_columns:
        if col_name not in columns:
            print(f"  Adding column: {col_name}")
            cursor.execute(f'ALTER TABLE learned_move_patterns ADD COLUMN {col_name} {col_def}')
        else:
            print(f"  Column already exists: {col_name}")

    # Update unique constraint to include new dimensions
    # SQLite doesn't support modifying constraints, so we need to check if data would be affected
    print("\n  Note: New patterns will use extended unique key:")
    print("    (piece_type, move_category, distance_from_start, game_phase,")
    print("     repetition_count, moves_since_progress, total_material_level)")

    conn.commit()
    conn.close()

    print("\n✓ Migration complete!")
    print("\nThe AI can now discover patterns like:")
    print("  - 'When repetition_count=2, games score -10000 → avoid'")
    print("  - 'When moves_since_progress=40-50, games draw → make progress'")
    print("  - 'When total_material=low, games draw → avoid trades'")

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'headless_training.db'
    migrate_database(db_path)
