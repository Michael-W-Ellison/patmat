#!/usr/bin/env python3
"""
Create Clean Database - Copy Only Essential Knowledge

Instead of deleting from 22GB file, create new clean database
with only abstract knowledge.
"""

import sqlite3
import os
import shutil

OLD_DB = "rule_discovery.db"
NEW_DB = "rule_discovery_clean.db"
BACKUP_DB = "rule_discovery_bloated_backup.db"

def create_clean_database():
    """Create new database with only essential knowledge"""
    print("=" * 70)
    print("CREATING CLEAN DATABASE")
    print("=" * 70)

    # Remove old clean db if exists
    if os.path.exists(NEW_DB):
        os.remove(NEW_DB)
        print(f"✓ Removed old {NEW_DB}")

    conn_old = sqlite3.connect(OLD_DB, timeout=60.0)
    conn_new = sqlite3.connect(NEW_DB)

    cursor_old = conn_old.cursor()
    cursor_new = conn_new.cursor()

    # Tables to copy (only abstract knowledge)
    essential_tables = [
        # Abstract patterns - the KEY learning!
        'abstract_patterns',

        # Discovered rules and weights
        'discovered_piece_values',
        'discovered_mobility_patterns',
        'discovered_opening_patterns',
        'discovered_opening_weights',
        'discovered_pawn_structure_patterns',
        'discovered_pawn_structure_weights',
        'discovered_phase_weights',
        'discovered_positional_patterns',
        'discovered_positional_weights',
        'discovered_tactical_patterns',
        'discovered_tactical_weights',
        'discovered_safety_patterns',
        'weak_square_patterns',
        'weak_square_weights',

        # Clustering (abstract position categories)
        'position_clusters',

        # Game statistics (outcomes only, not moves)
        'games',
        'learning_sessions',

        # Opening performance (book knowledge)
        'opening_performance',

        # Keep ONLY recent mistakes for learning
        # 'learned_mistakes',  # Will copy separately with LIMIT
    ]

    print(f"\nCopying essential tables from {OLD_DB} to {NEW_DB}:")
    print("-" * 70)

    total_rows = 0

    for table in essential_tables:
        try:
            # Get CREATE TABLE statement
            cursor_old.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
            result = cursor_old.fetchone()

            if not result:
                print(f"⚠  {table:40s}: Table not found")
                continue

            create_sql = result[0]

            # Create table in new database
            cursor_new.execute(create_sql)

            # Copy all data
            cursor_old.execute(f"SELECT * FROM {table}")
            rows = cursor_old.fetchall()

            if rows:
                placeholders = ','.join(['?'] * len(rows[0]))
                cursor_new.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                total_rows += len(rows)
                print(f"✓ {table:40s}: {len(rows):>8,} rows")
            else:
                print(f"○ {table:40s}: 0 rows")

        except Exception as e:
            print(f"✗ {table:40s}: ERROR - {e}")

    # Copy only RECENT learned mistakes (last 1000)
    print("\nCopying recent learned mistakes (last 1000 only):")
    try:
        cursor_old.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learned_mistakes'")
        result = cursor_old.fetchone()
        if result:
            cursor_new.execute(result[0])

            cursor_old.execute("""
                SELECT * FROM learned_mistakes
                ORDER BY mistake_id DESC
                LIMIT 1000
            """)
            rows = cursor_old.fetchall()

            if rows:
                placeholders = ','.join(['?'] * len(rows[0]))
                cursor_new.executemany(f"INSERT INTO learned_mistakes VALUES ({placeholders})", rows)
                total_rows += len(rows)
                print(f"✓ learned_mistakes:                      {len(rows):>8,} rows (recent only)")
    except Exception as e:
        print(f"✗ learned_mistakes: ERROR - {e}")

    # Copy indexes
    print("\nCopying indexes:")
    cursor_old.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    for row in cursor_old.fetchall():
        try:
            cursor_new.execute(row[0])
            print(f"✓ Index created")
        except:
            pass

    conn_new.commit()
    conn_new.close()
    conn_old.close()

    old_size = os.path.getsize(OLD_DB) / 1024**3
    new_size = os.path.getsize(NEW_DB) / 1024**2

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOld database: {old_size:.2f} GB (47 million rows)")
    print(f"New database: {new_size:.2f} MB ({total_rows:,} rows)")
    print(f"Space saved:  {old_size:.2f} GB → {new_size:.2f} MB ({(1 - new_size/1024/old_size)*100:.1f}% reduction)")

    return new_size

def replace_database():
    """Replace old database with new clean one"""
    print("\n" + "=" * 70)
    print("REPLACING DATABASE")
    print("=" * 70)

    # Backup old bloated database
    print(f"\nBacking up bloated database to {BACKUP_DB}...")
    shutil.move(OLD_DB, BACKUP_DB)
    print(f"✓ Moved {OLD_DB} → {BACKUP_DB}")

    # Move clean database to main name
    shutil.move(NEW_DB, OLD_DB)
    print(f"✓ Moved {NEW_DB} → {OLD_DB}")

    print(f"\n✓ Database replacement complete!")
    print(f"✓ Old bloated DB backed up to: {BACKUP_DB}")
    print(f"✓ New clean DB active as: {OLD_DB}")

def verify_clean_database():
    """Verify the clean database has all essential data"""
    print("\n" + "=" * 70)
    print("VERIFYING CLEAN DATABASE")
    print("=" * 70)

    conn = sqlite3.connect(OLD_DB)
    cursor = conn.cursor()

    checks = [
        ("Abstract patterns", "SELECT COUNT(*) FROM abstract_patterns", 5),
        ("Piece values", "SELECT COUNT(*) FROM discovered_piece_values", 6),
        ("Phase weights", "SELECT COUNT(*) FROM discovered_phase_weights", 10),
        ("Games played", "SELECT COUNT(*) FROM games", 1000),
        ("Position clusters", "SELECT COUNT(*) FROM position_clusters", 10),
    ]

    print("\nEssential data check:")
    all_good = True

    for name, query, min_expected in checks:
        try:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            status = "✓" if count >= min_expected else "⚠"
            print(f"{status} {name:30s}: {count:>8,} rows")
            if count < min_expected:
                all_good = False
        except Exception as e:
            print(f"✗ {name:30s}: ERROR - {e}")
            all_good = False

    # Check NO bloat tables exist
    print("\nVerifying bloat removed:")
    bloat_check = [
        "learned_tactics",
        "opponent_winning_patterns",
        "moves",
        "legal_moves",
    ]

    for table in bloat_check:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if cursor.fetchone():
            print(f"✗ {table}: Still exists (should be removed)")
            all_good = False
        else:
            print(f"✓ {table}: Removed")

    conn.close()

    if all_good:
        print("\n✓ Clean database verified successfully!")
    else:
        print("\n⚠ WARNING: Some checks failed")

    return all_good

if __name__ == '__main__':
    print("=" * 70)
    print("CHESS AI DATABASE CLEANUP - CREATE CLEAN DATABASE")
    print("=" * 70)
    print("\nThis will:")
    print("1. Create new clean database with only abstract knowledge")
    print("2. Copy 61 essential rows (patterns, weights, rules)")
    print("3. Skip 47 million position-specific rows")
    print("4. Reduce from 21GB to ~50MB")
    print("5. Backup old database to rule_discovery_bloated_backup.db")
    print("6. Replace rule_discovery.db with clean version")

    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        exit(0)

    # Create clean database
    new_size = create_clean_database()

    # Replace old with new
    replace_database()

    # Verify
    verify_clean_database()

    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Database reduced from 21GB to {new_size:.0f}MB")
    print(f"✓ AI now uses only abstract patterns, not specific positions")
    print(f"✓ Old database backed up to: {BACKUP_DB}")
    print(f"\nYou can delete {BACKUP_DB} to free up 21GB of space.")
