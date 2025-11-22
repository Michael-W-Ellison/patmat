#!/usr/bin/env python3
"""
Diagnose why imported patterns aren't improving AI performance

Checks:
1. Pattern statistics in database
2. Whether LearnableMovePrioritizer loads them correctly
3. What priorities it's actually using during games
"""

import sqlite3
import sys
import os

# Add chess_pattern_ai to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chess_pattern_ai'))

from learnable_move_prioritizer import LearnableMovePrioritizer

print("="*70)
print("PATTERN USAGE DIAGNOSTIC")
print("="*70)
print()

# Step 1: Check database directly
print("STEP 1: Database Check")
print("-"*70)

conn = sqlite3.connect('headless_training.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT COUNT(*),
           SUM(times_seen),
           SUM(games_won),
           SUM(games_lost),
           SUM(games_drawn),
           AVG(confidence),
           AVG(priority_score)
    FROM learned_move_patterns
''')

count, seen, won, lost, drawn, avg_conf, avg_pri = cursor.fetchone()
total_games = won + lost + drawn

print(f"Total patterns in database: {count:,}")
print(f"Total observations: {seen:,}")
print(f"Win rate: {won/total_games*100:.1f}% ({won:,}W-{lost:,}L-{drawn:,}D)")
print(f"Average confidence: {avg_conf:.3f}")
print(f"Average priority: {avg_pri:.1f}")
print()

# Check patterns with low confidence (won't be loaded)
cursor.execute('SELECT COUNT(*) FROM learned_move_patterns WHERE confidence <= 0.1')
low_conf_count = cursor.fetchone()[0]

if low_conf_count > 0:
    print(f"⚠ WARNING: {low_conf_count} patterns have confidence <= 0.1")
    print(f"  These won't be loaded by LearnableMovePrioritizer!")
    print()

# Check top patterns
print("Top 10 patterns by priority:")
cursor.execute('''
    SELECT piece_type, move_category, game_phase, distance_from_start,
           times_seen, win_rate, confidence, priority_score
    FROM learned_move_patterns
    ORDER BY priority_score DESC
    LIMIT 10
''')

print(f"{'Piece':<8} {'Category':<12} {'Phase':<10} {'Dist':>4} {'Seen':>10} {'WR':>6} {'Conf':>5} {'Pri':>5}")
for row in cursor.fetchall():
    piece, cat, phase, dist, seen, wr, conf, pri = row
    print(f"{piece:<8} {cat:<12} {phase:<10} {dist:>4} {seen:>10,} {wr:>6.1%} {conf:>5.2f} {pri:>5.1f}")

conn.close()
print()

# Step 2: Check if LearnableMovePrioritizer loads them
print("="*70)
print("STEP 2: LearnableMovePrioritizer Check")
print("-"*70)

prioritizer = LearnableMovePrioritizer('headless_training.db')

# Check loaded patterns
loaded_count = len(prioritizer.move_priorities)
print(f"Patterns loaded into memory: {loaded_count:,}")

if loaded_count == 0:
    print("❌ ERROR: No patterns loaded!")
    print("   Possible causes:")
    print("   1. All patterns have confidence <= 0.1")
    print("   2. Database connection failed")
    print("   3. Wrong database file")
elif loaded_count < count * 0.5:
    print(f"⚠ WARNING: Only {loaded_count/count*100:.0f}% of patterns loaded!")
    print("   Many patterns have low confidence")
else:
    print(f"✓ {loaded_count/count*100:.0f}% of patterns loaded successfully")

print()

# Step 3: Show what priorities it's using
if loaded_count > 0:
    print("="*70)
    print("STEP 3: Sample Pattern Priorities")
    print("-"*70)

    print("\nShowing first 10 loaded patterns:")
    print(f"{'Key':<50} {'Priority':>8}")
    print("-"*70)

    for i, (key, data) in enumerate(list(prioritizer.move_priorities.items())[:10]):
        piece, category, distance, phase = key
        key_str = f"({piece}, {category}, d={distance}, {phase})"
        print(f"{key_str:<50} {data['priority']:>8.1f}")

    # Check if patterns have good priorities
    priorities = [data['priority'] for data in prioritizer.move_priorities.values()]
    avg_loaded_pri = sum(priorities) / len(priorities)
    max_loaded_pri = max(priorities)

    print()
    print(f"Average priority of loaded patterns: {avg_loaded_pri:.1f}")
    print(f"Maximum priority: {max_loaded_pri:.1f}")
    print()

    if max_loaded_pri < 30:
        print("❌ ERROR: All loaded patterns have very low priority!")
        print("   This means patterns lead to poor outcomes.")
        print("   Win rates in PGN games are likely very low.")
    elif avg_loaded_pri < 30:
        print("⚠ WARNING: Average priority is low.")
        print("   Most patterns don't show strong advantages.")
    else:
        print("✓ Priorities look reasonable")

prioritizer.close()

print()
print("="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if loaded_count == 0:
    print("❌ CRITICAL: No patterns are being used!")
    print("   Fix: Check pattern confidence values")
elif loaded_count > 0 and max_loaded_pri < 30:
    print("❌ CRITICAL: Patterns loaded but have poor priorities!")
    print("   Fix: PGN data may have wrong win rates or scoring")
elif loaded_count > 0:
    print("✓ Patterns are loaded and have reasonable priorities")
    print("  If AI still performs poorly, the issue is elsewhere:")
    print("  - Move evaluation logic may override pattern priorities")
    print("  - Pattern matching may not be working correctly")
    print("  - Need more specific patterns for the positions encountered")

print()
