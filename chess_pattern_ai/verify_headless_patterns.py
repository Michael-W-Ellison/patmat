#!/usr/bin/env python3
"""
Verify and Extract Patterns from Headless Training

This script:
1. Checks if headless_training.db exists and what's in it
2. Extracts patterns from the trained games
3. Adds them to the universal pattern database
"""

import sqlite3
import os
import json

def check_headless_training_db(db_path='headless_training.db'):
    """Check what's in the headless training database"""

    if not os.path.exists(db_path):
        print(f"❌ {db_path} not found!")
        print("\nThe database should be created when you run:")
        print("  python chess_pattern_ai/headless_trainer.py 100")
        print("\nMake sure you're in the project root directory.")
        return None

    print(f"✓ Found {db_path}")
    print("="*70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    print(f"\nTables: {len(tables)}")
    for table_name, in tables:
        print(f"  - {table_name}")

    # Check learned_move_patterns
    if any(t[0] == 'learned_move_patterns' for t in tables):
        print("\n" + "="*70)
        print("Learned Move Patterns:")
        print("="*70)

        cursor.execute('''
            SELECT COUNT(*) as total,
                   AVG(confidence) as avg_conf,
                   AVG(win_rate) as avg_win_rate,
                   SUM(times_seen) as total_observations
            FROM learned_move_patterns
        ''')

        total, avg_conf, avg_wr, total_obs = cursor.fetchone()

        print(f"\nTotal Patterns: {total}")
        print(f"Average Confidence: {avg_conf:.2f}")
        print(f"Average Win Rate: {avg_wr:.2f}%")
        print(f"Total Observations: {total_obs}")

        # Show top patterns
        print("\nTop 10 Patterns by Confidence:")
        cursor.execute('''
            SELECT piece_type, move_category, game_phase,
                   times_seen, confidence, win_rate, avg_score
            FROM learned_move_patterns
            ORDER BY confidence DESC
            LIMIT 10
        ''')

        print(f"\n{'Piece':<8} {'Category':<15} {'Phase':<12} {'Seen':<6} {'Conf':<6} {'WR%':<6} {'Score':<7}")
        print("-"*70)

        for piece, category, phase, seen, conf, wr, score in cursor.fetchall():
            print(f"{piece:<8} {category:<15} {phase:<12} {seen:<6} {conf:<6.2f} {wr*100:<6.1f} {score:<7.0f}")

        # Movement patterns
        print("\n" + "="*70)
        print("Movement Pattern Summary:")
        print("="*70)

        cursor.execute('''
            SELECT piece_type, COUNT(*) as num_patterns, SUM(times_seen) as observations
            FROM learned_move_patterns
            GROUP BY piece_type
            ORDER BY observations DESC
        ''')

        print(f"\n{'Piece':<10} {'Patterns':<10} {'Observations':<15}")
        print("-"*40)
        for piece, num, obs in cursor.fetchall():
            print(f"{piece:<10} {num:<10} {obs:<15}")

        return conn
    else:
        print("\n❌ No learned_move_patterns table found!")
        print("This might mean no training has been completed yet.")
        return None

def extract_patterns_from_headless(db_path='headless_training.db'):
    """Extract patterns from headless training database"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    patterns = []

    print("\n" + "="*70)
    print("Extracting Patterns for Universal Database:")
    print("="*70)

    # Get all learned patterns
    cursor.execute('''
        SELECT piece_type, move_category, game_phase,
               times_seen, confidence, win_rate, avg_score
        FROM learned_move_patterns
        WHERE times_seen > 5  -- Only patterns with enough observations
        ORDER BY confidence DESC
    ''')

    for piece, category, phase, seen, conf, wr, score in cursor.fetchall():
        # Create pattern entries based on move characteristics

        # Capture patterns
        if 'capture' in category.lower():
            patterns.append({
                'name': 'tactical_capture',
                'category': 'tactical',
                'game': 'chess_headless_training',
                'frequency': wr,
                'observation_count': seen,
                'description': f'{piece} {category} in {phase}',
                'features': {
                    'piece_type': piece,
                    'move_type': category,
                    'game_phase': phase,
                    'confidence': conf,
                    'avg_score': score
                }
            })

        # Development patterns
        if 'development' in category.lower() or 'quiet' in category.lower():
            patterns.append({
                'name': 'positional_development',
                'category': 'positional',
                'game': 'chess_headless_training',
                'frequency': wr,
                'observation_count': seen,
                'description': f'{piece} {category} in {phase}',
                'features': {
                    'piece_type': piece,
                    'move_type': category,
                    'game_phase': phase,
                    'confidence': conf
                }
            })

        # Check patterns (tactical)
        if 'check' in category.lower():
            patterns.append({
                'name': 'forcing_check',
                'category': 'tactical',
                'game': 'chess_headless_training',
                'frequency': wr,
                'observation_count': seen,
                'description': f'{piece} delivers check in {phase}',
                'features': {
                    'piece_type': piece,
                    'forcing_move': True,
                    'game_phase': phase,
                    'confidence': conf
                }
            })

    conn.close()

    print(f"\n✓ Extracted {len(patterns)} patterns from headless training")

    # Show summary
    if patterns:
        print("\nPattern Summary:")
        pattern_types = {}
        for p in patterns:
            ptype = p['name']
            if ptype not in pattern_types:
                pattern_types[ptype] = {'count': 0, 'observations': 0}
            pattern_types[ptype]['count'] += 1
            pattern_types[ptype]['observations'] += p['observation_count']

        for ptype, stats in sorted(pattern_types.items(), key=lambda x: x[1]['observations'], reverse=True):
            print(f"  {ptype}: {stats['count']} patterns, {stats['observations']} observations")

    return patterns

def add_to_universal_database(patterns, universal_db='chess_pattern_ai/universal_patterns.db'):
    """Add extracted patterns to universal database"""

    if not patterns:
        print("\n❌ No patterns to add!")
        return

    # Try multiple possible locations for the database
    possible_paths = [
        universal_db,
        'universal_patterns.db',
        os.path.join('chess_pattern_ai', 'universal_patterns.db'),
        os.path.join(os.path.dirname(__file__), 'universal_patterns.db')
    ]

    db_path = None
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            print(f"\n✓ Found universal database at: {path}")
            break

    if not db_path:
        print(f"\n❌ Universal database not found!")
        print(f"Searched:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nRun this first:")
        print("  python chess_pattern_ai/universal_pattern_extractor.py")
        return

    print(f"\n" + "="*70)
    print(f"Adding patterns to {db_path}:")
    print("="*70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    patterns_added = 0
    patterns_updated = 0

    for pattern in patterns:
        pattern_name = pattern['name']
        category = pattern['category']
        game_type = pattern['game']
        frequency = pattern['frequency']
        obs_count = pattern['observation_count']
        description = pattern['description']
        features = pattern['features']

        # Check if pattern exists
        cursor.execute('''
            SELECT pattern_id, observation_count FROM universal_patterns
            WHERE pattern_name = ?
        ''', (pattern_name,))

        result = cursor.fetchone()

        if result:
            pattern_id, total_obs = result

            # Update observation count
            cursor.execute('''
                UPDATE universal_patterns
                SET observation_count = observation_count + ?
                WHERE pattern_id = ?
            ''', (obs_count, pattern_id))

            patterns_updated += 1
        else:
            # New pattern
            cursor.execute('''
                INSERT INTO universal_patterns
                (pattern_name, pattern_category, confidence, observation_count, description)
                VALUES (?, ?, 0.5, ?, ?)
            ''', (pattern_name, category, obs_count, description))

            pattern_id = cursor.lastrowid
            patterns_added += 1

            # Store features
            for feature_name, feature_value in features.items():
                cursor.execute('''
                    INSERT INTO universal_pattern_features
                    (pattern_id, feature_name, feature_value)
                    VALUES (?, ?, ?)
                ''', (pattern_id, feature_name, json.dumps(feature_value)))

        # Add/update game_to_pattern link
        cursor.execute('''
            SELECT id FROM game_to_pattern
            WHERE game_type = ? AND pattern_id = ?
        ''', (game_type, pattern_id))

        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO game_to_pattern (game_type, pattern_id, frequency, observation_count)
                VALUES (?, ?, ?, ?)
            ''', (game_type, pattern_id, frequency, obs_count))

    conn.commit()
    conn.close()

    print(f"\n✓ Added {patterns_added} new patterns")
    print(f"✓ Updated {patterns_updated} existing patterns")
    print(f"✓ Total: {patterns_added + patterns_updated} patterns processed")

if __name__ == '__main__':
    import sys

    print("="*70)
    print("Headless Training Pattern Extraction")
    print("="*70)

    # Determine database path
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Try both locations
        if os.path.exists('headless_training.db'):
            db_path = 'headless_training.db'
        elif os.path.exists('chess_pattern_ai/headless_training.db'):
            db_path = 'chess_pattern_ai/headless_training.db'
        else:
            db_path = 'headless_training.db'  # Will show error

    # Step 1: Check what's in the database
    conn = check_headless_training_db(db_path)

    if conn:
        conn.close()

        # Step 2: Extract patterns
        patterns = extract_patterns_from_headless(db_path)

        # Step 3: Add to universal database
        add_to_universal_database(patterns)

        print("\n" + "="*70)
        print("✓ Pattern extraction complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run more training to strengthen patterns")
        print("  2. Test on ARC puzzles: python chess_pattern_ai/test_enriched_patterns.py")
    else:
        print("\n" + "="*70)
        print("How to create headless_training.db:")
        print("="*70)
        print("\n1. Run training:")
        print("   python chess_pattern_ai/headless_trainer.py 100")
        print("\n2. Then run this script again:")
        print("   python chess_pattern_ai/verify_headless_patterns.py")
