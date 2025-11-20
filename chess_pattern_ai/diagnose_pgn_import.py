#!/usr/bin/env python3
"""
Diagnose PGN Import Issues

Checks database for anomalies and calculates correct statistics.
"""

import sqlite3
import sys

def diagnose_database(db_path):
    """Diagnose issues with pattern database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error opening {db_path}: {e}")
        return

    print("=" * 80)
    print(f"DIAGNOSING: {db_path}")
    print("=" * 80)

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
    if not cursor.fetchone():
        print("✗ No learned_move_patterns table found!")
        print("  This database hasn't been used for pattern learning yet.")
        conn.close()
        return

    # Get all patterns
    cursor.execute('''
        SELECT piece_type, move_category, game_phase, distance_from_start,
               times_seen, games_won, games_lost, games_drawn,
               win_rate, confidence, priority_score
        FROM learned_move_patterns
    ''')

    patterns = cursor.fetchall()

    if not patterns:
        print("✗ No patterns in database!")
        conn.close()
        return

    print(f"✓ Found {len(patterns):,} patterns")
    print()

    # Check for inconsistencies
    print("Checking for data inconsistencies...")
    print("-" * 80)

    issues = 0
    total_obs = 0
    total_won = 0
    total_lost = 0
    total_drawn = 0

    for i, row in enumerate(patterns):
        piece, cat, phase, dist, seen, won, lost, drawn, wr, conf, pri = row

        total_obs += seen
        total_won += won
        total_lost += lost
        total_drawn += drawn

        # Check: times_seen should equal won + lost + drawn
        calculated_total = won + lost + drawn
        if seen != calculated_total:
            if issues < 5:  # Only show first 5
                print(f"✗ Pattern {i+1}: times_seen={seen} but won+lost+drawn={calculated_total}")
            issues += 1

        # Check: win_rate should equal won / total_games
        if calculated_total > 0:
            correct_wr = won / calculated_total
            if abs(wr - correct_wr) > 0.001:  # Allow small floating point errors
                if issues < 5:
                    print(f"✗ Pattern {i+1}: stored WR={wr:.4f} but should be {correct_wr:.4f}")
                issues += 1

        # Check: confidence calculation
        correct_conf = min(1.0, seen / 100.0)
        if abs(conf - correct_conf) > 0.001:
            if issues < 5:
                print(f"✗ Pattern {i+1}: stored conf={conf:.4f} but should be {correct_conf:.4f}")
            issues += 1

        # Check: priority score
        correct_pri = correct_wr * correct_conf * 100
        if abs(pri - correct_pri) > 0.1:
            if issues < 5:
                print(f"✗ Pattern {i+1} ({piece} {cat} {phase}): stored pri={pri:.2f} but should be {correct_pri:.2f}")
                print(f"   WR={wr:.4f} (correct={correct_wr:.4f}), conf={conf:.4f} (correct={correct_conf:.4f})")
            issues += 1

    if issues > 5:
        print(f"... and {issues-5} more issues")

    if issues == 0:
        print("✓ No inconsistencies found!")

    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_games = total_won + total_lost + total_drawn
    overall_wr = total_won / total_games if total_games > 0 else 0

    print(f"Total observations: {total_obs:,}")
    print(f"Total game outcomes: {total_games:,}")
    print(f"  Won:   {total_won:,} ({total_won/total_games*100:.1f}%)")
    print(f"  Lost:  {total_lost:,} ({total_lost/total_games*100:.1f}%)")
    print(f"  Drawn: {total_drawn:,} ({total_drawn/total_games*100:.1f}%)")
    print(f"Overall win rate: {overall_wr:.1%}")
    print()

    if total_obs != total_games:
        print(f"⚠ WARNING: times_seen total ({total_obs:,}) != game outcomes ({total_games:,})")
        print(f"  Difference: {abs(total_obs - total_games):,}")
        print()

    if overall_wr < 0.30 or overall_wr > 0.70:
        print(f"⚠ WARNING: Overall win rate ({overall_wr:.1%}) is unusual!")
        print(f"  Expected: ~48-52% for balanced training data")
        print(f"  This suggests a bug in win/loss counting")
        print()

    # Show top patterns by actual calculated priority
    print("=" * 80)
    print("TOP 10 PATTERNS (sorted by corrected priority)")
    print("=" * 80)

    corrected_patterns = []
    for row in patterns:
        piece, cat, phase, dist, seen, won, lost, drawn, wr, conf, pri = row
        total = won + lost + drawn
        if total > 0:
            correct_wr = won / total
            correct_conf = min(1.0, seen / 100.0)
            correct_pri = correct_wr * correct_conf * 100
            corrected_patterns.append((piece, cat, phase, dist, seen, won, lost, drawn,
                                      correct_wr, correct_conf, correct_pri))

    corrected_patterns.sort(key=lambda x: x[10], reverse=True)  # Sort by corrected priority

    print(f"{'Piece':<10} {'Category':<15} {'Phase':<12} {'Seen':>8} {'Won':>7} {'Lost':>7} {'WR':>6} {'Pri':>7}")
    print("-" * 80)

    for i, row in enumerate(corrected_patterns[:10], 1):
        piece, cat, phase, dist, seen, won, lost, drawn, wr, conf, pri = row
        print(f"{piece:<10} {cat:<15} {phase:<12} {seen:>8,} {won:>7,} {lost:>7,} {wr:>6.1%} {pri:>7.1f}")

    conn.close()

    return issues == 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = 'headless_training.db'

    print("PGN Import Diagnostic Tool")
    print()

    diagnose_database(db_path)
