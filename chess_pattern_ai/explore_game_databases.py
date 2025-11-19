#!/usr/bin/env python3
"""
Explore all game databases to find available game data

User's insight: "Every element of a 'game' environment should be included as data.
Maybe running through some additional game tests will provide additional patterns."
"""

import sqlite3
import os

databases = [
    '../checkers_training.db',
    '../demo_learning.db',
    'rule_discovery.db',
    'test_integration.db'
]

print("Exploring Game Databases")
print("="*70)

for db_path in databases:
    if not os.path.exists(db_path):
        print(f"\n✗ {db_path}: Not found")
        continue

    print(f"\n{db_path}")
    print("-"*70)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        if not tables:
            print("  No tables found")
            conn.close()
            continue

        print(f"  Tables: {len(tables)}")

        for table_name, in tables:
            print(f"\n  Table: {table_name}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"    Rows: {count}")

            if count > 0:
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print(f"    Columns: {', '.join([col[1] for col in columns[:5]])}...")

                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                samples = cursor.fetchall()
                if samples:
                    print(f"    Sample: {len(samples)} rows")

        conn.close()

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*70)
print("Summary:")
print("="*70)

# Look for specific game types
print("\nLooking for game types in databases...")

for db_path in databases:
    if not os.path.exists(db_path):
        continue

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check for game_type column
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table_name, in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]

                if 'game_type' in columns:
                    cursor.execute(f"SELECT DISTINCT game_type FROM {table_name}")
                    game_types = cursor.fetchall()
                    if game_types:
                        print(f"\n  {db_path} - {table_name}:")
                        for gt, in game_types:
                            print(f"    - {gt}")
            except:
                pass

        conn.close()

    except Exception as e:
        pass
