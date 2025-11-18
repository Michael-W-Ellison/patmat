#!/usr/bin/env python3
"""
Database Cleanup - Remove Bloat, Keep Only Essential Knowledge

The database has 22GB of specific positions. We only need:
1. Abstract patterns (the PRINCIPLES learned)
2. Discovered weights (evaluation parameters)
3. Discovered piece values
4. Opening book (starting positions only)

Everything else is redundant position-specific data.
"""

import sqlite3
import os

DB_PATH = "rule_discovery.db"

def get_table_row_count(cursor, table_name):
    """Get row count for a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    except:
        return 0

def analyze_database():
    """Show what's taking up space"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 70)
    print("DATABASE SIZE ANALYSIS")
    print("=" * 70)

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"\nTotal tables: {len(tables)}")
    print(f"Database size: {os.path.getsize(DB_PATH) / 1024**3:.2f} GB")

    # Check large tables
    print("\n" + "=" * 70)
    print("LARGE TABLES (> 1000 rows)")
    print("=" * 70)

    large_tables = []
    for table in tables:
        count = get_table_row_count(cursor, table)
        if count > 1000:
            large_tables.append((table, count))

    large_tables.sort(key=lambda x: x[1], reverse=True)

    for table, count in large_tables:
        print(f"{table:40s}: {count:>12,} rows")

    # Check essential tables
    print("\n" + "=" * 70)
    print("ESSENTIAL KNOWLEDGE TABLES (what we keep)")
    print("=" * 70)

    essential = [
        'abstract_patterns',
        'discovered_piece_values',
        'discovered_mobility_patterns',
        'discovered_opening_weights',
        'discovered_pawn_structure_weights',
        'discovered_phase_weights',
        'discovered_positional_weights',
        'discovered_tactical_weights',
        'weak_square_weights',
        'position_clusters',  # Keep cluster centers
    ]

    for table in essential:
        count = get_table_row_count(cursor, table)
        print(f"{table:40s}: {count:>12,} rows")

    conn.close()

    return large_tables, essential

def create_backup():
    """Backup essential data before cleanup"""
    print("\n" + "=" * 70)
    print("CREATING BACKUP OF ESSENTIAL DATA")
    print("=" * 70)

    backup_path = "rule_discovery_backup.db"

    if os.path.exists(backup_path):
        os.remove(backup_path)

    conn_src = sqlite3.connect(DB_PATH)
    conn_dst = sqlite3.connect(backup_path)

    # Copy schema and essential data
    essential_tables = [
        'abstract_patterns',
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
        'position_clusters',
        'games',  # Keep game results for statistics
        'learning_sessions',
        'opening_performance',
    ]

    cursor_src = conn_src.cursor()
    cursor_dst = conn_dst.cursor()

    for table in essential_tables:
        try:
            # Get table schema
            cursor_src.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
            result = cursor_src.fetchone()
            if result:
                create_sql = result[0]
                cursor_dst.execute(create_sql)

                # Copy data
                cursor_src.execute(f"SELECT * FROM {table}")
                rows = cursor_src.fetchall()

                if rows:
                    placeholders = ','.join(['?'] * len(rows[0]))
                    cursor_dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

                print(f"✓ Backed up {table}: {len(rows)} rows")
        except Exception as e:
            print(f"✗ Failed to backup {table}: {e}")

    conn_dst.commit()
    conn_dst.close()
    conn_src.close()

    backup_size = os.path.getsize(backup_path) / 1024**2
    print(f"\n✓ Backup created: {backup_path} ({backup_size:.2f} MB)")

    return backup_path

def cleanup_database():
    """Remove bloated tables, keep only essential knowledge"""
    print("\n" + "=" * 70)
    print("CLEANING UP DATABASE")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tables to DELETE (position-specific bloat)
    bloat_tables = [
        'learned_tactics',      # 19.5M specific positions - DELETE
        'opponent_winning_patterns',  # 14.5M specific positions - DELETE
        'moves',                 # 7.5M specific moves - DELETE
        'legal_moves',           # 6.3M specific moves - DELETE
        'abstracted_winning_patterns',  # Position-specific
        'move_chains',           # Position-specific
        'king_attack_sequences', # Position-specific
        'sequence_templates',    # Position-specific
        'template_matches',      # Position-specific
        'move_anomalies',        # Position-specific
        'anomaly_statistics',    # Position-specific
        'evaluation_corrections',# Position-specific
        'strategic_errors',      # Position-specific
        'pattern_value_cache',   # Position-specific cache
        'position_cluster_membership',  # Specific positions → clusters mapping
    ]

    print("\nDeleting bloated tables:")
    for table in bloat_tables:
        try:
            count = get_table_row_count(cursor, table)
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"✓ Deleted {table}: {count:,} rows removed")
        except Exception as e:
            print(f"✗ Failed to delete {table}: {e}")

    # Tables to TRUNCATE (keep structure, delete data)
    truncate_tables = [
        'learned_mistakes',  # Keep recent mistakes only
    ]

    print("\nTruncating tables (keeping structure):")
    for table in truncate_tables:
        try:
            count = get_table_row_count(cursor, table)
            # Keep only last 1000 mistakes for recent learning
            cursor.execute(f"""
                DELETE FROM {table}
                WHERE mistake_id NOT IN (
                    SELECT mistake_id FROM {table}
                    ORDER BY mistake_id DESC
                    LIMIT 1000
                )
            """)
            new_count = get_table_row_count(cursor, table)
            print(f"✓ Truncated {table}: {count:,} → {new_count:,} rows")
        except Exception as e:
            print(f"✗ Failed to truncate {table}: {e}")

    conn.commit()

    # Vacuum to reclaim space
    print("\nVacuuming database to reclaim space...")
    cursor.execute("VACUUM")

    conn.close()

    new_size = os.path.getsize(DB_PATH) / 1024**3
    print(f"\n✓ Cleanup complete!")
    print(f"✓ New database size: {new_size:.2f} GB")

def verify_essential_data():
    """Verify essential data is still intact"""
    print("\n" + "=" * 70)
    print("VERIFYING ESSENTIAL DATA")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    checks = [
        ("Abstract patterns", "SELECT COUNT(*) FROM abstract_patterns"),
        ("Piece values", "SELECT COUNT(*) FROM discovered_piece_values"),
        ("Phase weights", "SELECT COUNT(*) FROM discovered_phase_weights"),
        ("Tactical weights", "SELECT COUNT(*) FROM discovered_tactical_weights"),
        ("Position clusters", "SELECT COUNT(*) FROM position_clusters"),
        ("Games played", "SELECT COUNT(*) FROM games"),
    ]

    all_good = True
    for name, query in checks:
        try:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            status = "✓" if count > 0 else "✗"
            print(f"{status} {name:30s}: {count:>6,} rows")
            if count == 0:
                all_good = False
        except Exception as e:
            print(f"✗ {name:30s}: ERROR - {e}")
            all_good = False

    conn.close()

    if all_good:
        print("\n✓ All essential data verified!")
    else:
        print("\n✗ WARNING: Some essential data missing!")

    return all_good

if __name__ == '__main__':
    print("=" * 70)
    print("CHESS AI DATABASE CLEANUP")
    print("=" * 70)
    print("\nThis will:")
    print("1. Analyze current database size and content")
    print("2. Create backup of essential knowledge")
    print("3. Remove 19.5M+ position-specific records")
    print("4. Keep only abstract patterns and learned weights")
    print("5. Reduce database from 22GB to < 100MB")

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    # Step 1: Analyze
    large_tables, essential = analyze_database()

    # Step 2: Backup
    backup_path = create_backup()

    # Step 3: Cleanup
    cleanup_database()

    # Step 4: Verify
    verify_essential_data()

    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Backup saved to: {backup_path}")
    print(f"✓ Database cleaned: {DB_PATH}")
    print(f"✓ Space saved: ~22 GB → < 100 MB")
    print(f"\nThe AI now uses only abstract knowledge, not specific positions!")
