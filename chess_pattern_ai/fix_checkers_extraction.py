#!/usr/bin/env python3
"""
Fix checkers pattern extraction and add more game data
"""

import sqlite3
import json

# Fix checkers database
conn = sqlite3.connect('../checkers_training.db')
cursor = conn.cursor()

# Check actual schema
cursor.execute("PRAGMA table_info(learned_move_patterns)")
columns = [col[1] for col in cursor.fetchall()]
print("Checkers database columns:", columns)

# Get checkers patterns
cursor.execute("SELECT * FROM learned_move_patterns")
rows = cursor.fetchall()

print(f"\nCheckers patterns: {len(rows)}")

patterns = []
for row in rows:
    print(f"  Row: {row}")

    # Extract pattern based on actual columns
    # Assuming columns are: id, piece_type, move_category, distance_from_start, game_phase, times_seen
    if len(row) >= 6:
        pid, piece, category, dist, phase, seen = row[:6]

        patterns.append({
            'name': 'diagonal_movement',
            'category': 'movement',
            'game': 'checkers',
            'frequency': 1.0,
            'observation_count': seen,
            'description': f'Checkers {category} pattern - ALL checkers moves are diagonal',
            'features': {
                'direction': 'diagonal',
                'piece_type': piece,
                'move_type': category,
                'always_diagonal': True
            }
        })

        if 'jump' in category.lower():
            patterns.append({
                'name': 'multi_hop_diagonal',
                'category': 'movement',
                'game': 'checkers',
                'frequency': 0.40,
                'observation_count': seen,
                'description': f'Checkers multi-hop diagonal jump',
                'features': {
                    'direction': 'diagonal',
                    'skips_cells': True,
                    'multi_hop': True
                }
            })

conn.close()

# Store in universal database
universal_conn = sqlite3.connect('universal_patterns.db')
universal_cursor = universal_conn.cursor()

for pattern in patterns:
    pattern_name = pattern['name']
    category = pattern['category']
    game_type = pattern['game']
    frequency = pattern['frequency']
    obs_count = pattern['observation_count']
    description = pattern['description']
    features = pattern['features']

    # Check if pattern exists
    universal_cursor.execute('''
        SELECT pattern_id FROM universal_patterns
        WHERE pattern_name = ?
    ''', (pattern_name,))

    result = universal_cursor.fetchone()

    if result:
        pattern_id = result[0]

        # Update observation count
        universal_cursor.execute('''
            UPDATE universal_patterns
            SET observation_count = observation_count + ?
            WHERE pattern_id = ?
        ''', (obs_count, pattern_id))

        # Get game count for confidence boost
        universal_cursor.execute('''
            SELECT COUNT(DISTINCT game_type) FROM game_to_pattern
            WHERE pattern_id = ?
        ''', (pattern_id,))

        game_count = universal_cursor.fetchone()[0]

        # Update confidence
        if game_count >= 2:
            confidence = 0.85
        elif game_count == 1:
            confidence = 0.7
        else:
            confidence = 0.5

        universal_cursor.execute('''
            UPDATE universal_patterns
            SET confidence = ?
            WHERE pattern_id = ?
        ''', (confidence, pattern_id))

    else:
        # New pattern
        universal_cursor.execute('''
            INSERT INTO universal_patterns
            (pattern_name, pattern_category, confidence, observation_count, description)
            VALUES (?, ?, 0.5, ?, ?)
        ''', (pattern_name, category, obs_count, description))

        pattern_id = universal_cursor.lastrowid

        # Store features
        for feature_name, feature_value in features.items():
            universal_cursor.execute('''
                INSERT INTO universal_pattern_features
                (pattern_id, feature_name, feature_value)
                VALUES (?, ?, ?)
            ''', (pattern_id, feature_name, json.dumps(feature_value)))

    # Add game_to_pattern link
    universal_cursor.execute('''
        SELECT id FROM game_to_pattern
        WHERE game_type = ? AND pattern_id = ?
    ''', (game_type, pattern_id))

    if not universal_cursor.fetchone():
        universal_cursor.execute('''
            INSERT INTO game_to_pattern (game_type, pattern_id, frequency, observation_count)
            VALUES (?, ?, ?, ?)
        ''', (game_type, pattern_id, frequency, obs_count))

universal_conn.commit()

print(f"\n✓ Stored {len(patterns)} checkers patterns")

# Show updated diagonal patterns
universal_cursor.execute('''
    SELECT
        p.pattern_name,
        p.confidence,
        p.observation_count,
        GROUP_CONCAT(DISTINCT g.game_type) as games
    FROM universal_patterns p
    LEFT JOIN game_to_pattern g ON p.pattern_id = g.pattern_id
    WHERE p.pattern_name LIKE '%diagonal%'
    GROUP BY p.pattern_id
''')

print("\nDiagonal patterns now available:")
for name, conf, obs, games in universal_cursor.fetchall():
    print(f"  {name}: {conf:.2f} confidence, {obs} observations")
    print(f"    Games: {games}")

universal_conn.close()
