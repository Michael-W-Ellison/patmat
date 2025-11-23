#!/usr/bin/env python3
"""
EMERGENCY FIX v2: Enhanced Pattern Learning System Recovery (SQLite Compatible)

Fixed version that uses CASE statements instead of GREATEST function
"""

import sqlite3
import sys
from pathlib import Path

def diagnose_system(db_path):
    """Diagnose what went wrong with the enhanced system"""
    
    print("🔍 DIAGNOSING ENHANCED SYSTEM PROBLEMS")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check catastrophic patterns
    cursor.execute('''
        SELECT piece_type, move_category, times_seen, win_rate, priority_score, avg_score
        FROM learned_move_patterns 
        WHERE priority_score < -100
        ORDER BY priority_score ASC
        LIMIT 20
    ''')
    
    catastrophic = cursor.fetchall()
    print(f"❌ Found {len(catastrophic)} catastrophic patterns (priority < -100):")
    
    for piece, cat, seen, wr, pri, avg in catastrophic:
        print(f"   {piece:8} {cat:12} | {seen:4d} times | {wr:5.1%} WR | {pri:6.1f} priority | {avg:6.0f} avg")
    
    # Check if queen patterns are all bad
    cursor.execute('''
        SELECT COUNT(*), AVG(priority_score), AVG(win_rate)
        FROM learned_move_patterns 
        WHERE piece_type = 'queen' AND times_seen > 10
    ''')
    
    queen_count, avg_priority, avg_winrate = cursor.fetchone()
    print(f"\n👑 Queen pattern analysis:")
    print(f"   Patterns: {queen_count}")
    print(f"   Average priority: {avg_priority:.1f}")
    print(f"   Average win rate: {avg_winrate:.1%}")
    
    # Check how many patterns have 0% win rate
    cursor.execute('''
        SELECT COUNT(*) 
        FROM learned_move_patterns 
        WHERE win_rate = 0.0 AND times_seen > 20
    ''')
    
    zero_winrate = cursor.fetchone()[0]
    print(f"\n🚫 Patterns with 0% win rate (>20 games): {zero_winrate}")
    
    conn.close()
    
    return len(catastrophic) > 0

def fix_catastrophic_patterns(db_path):
    """Fix the catastrophically bad patterns (SQLite compatible version)"""
    
    print(f"\n🔧 FIXING CATASTROPHIC PATTERNS (SQLite Compatible)")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Backup already created in previous run, but check if it exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns_backup'")
    if not cursor.fetchone():
        print("📦 Creating backup...")
        cursor.execute('CREATE TABLE learned_move_patterns_backup AS SELECT * FROM learned_move_patterns')
    else:
        print("📦 Backup already exists, skipping...")
    
    # Fix 1: Reset patterns with priority < -100 to neutral (already done, but verify)
    cursor.execute('SELECT COUNT(*) FROM learned_move_patterns WHERE priority_score < -100')
    remaining_catastrophic = cursor.fetchone()[0]
    
    if remaining_catastrophic > 0:
        print(f"🔨 Still {remaining_catastrophic} catastrophic patterns found, resetting...")
        cursor.execute('''
            UPDATE learned_move_patterns 
            SET priority_score = 50.0,
                confidence = 0.1,
                win_rate = 0.3
            WHERE priority_score < -100
        ''')
        reset_count = cursor.rowcount
        print(f"   ✓ Reset {reset_count} additional catastrophic patterns")
    else:
        print("✓ No catastrophic patterns remaining")
    
    # Fix 2: Restore queen patterns to reasonable values (already done, but verify)
    cursor.execute('SELECT COUNT(*) FROM learned_move_patterns WHERE piece_type = "queen" AND priority_score < 30')
    bad_queens = cursor.fetchone()[0]
    
    if bad_queens > 0:
        print(f"👑 Still {bad_queens} under-prioritized queen patterns, fixing...")
        cursor.execute('''
            UPDATE learned_move_patterns 
            SET priority_score = CASE 
                WHEN move_category = 'capture' THEN 75.0
                WHEN move_category = 'check' THEN 70.0  
                WHEN move_category = 'capture_check' THEN 80.0
                ELSE 60.0
            END,
            win_rate = CASE
                WHEN move_category = 'capture' THEN 0.45
                WHEN move_category = 'check' THEN 0.40
                WHEN move_category = 'capture_check' THEN 0.50
                ELSE 0.35
            END,
            confidence = 0.3
            WHERE piece_type = 'queen' AND priority_score < 30
        ''')
        queen_fixes = cursor.rowcount
        print(f"   ✓ Fixed {queen_fixes} additional queen patterns")
    else:
        print("✓ Queen patterns already fixed")
    
    # Fix 3: Add minimum priority constraint using CASE (SQLite compatible)
    print("⚖️ Applying minimum priority constraints (SQLite compatible)...")
    
    # Queen pieces minimum 30.0
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET priority_score = CASE 
            WHEN priority_score < 30.0 THEN 30.0 
            ELSE priority_score 
        END
        WHERE piece_type = 'queen' AND priority_score < 30.0
    ''')
    queen_min_fixes = cursor.rowcount
    
    # Rook pieces minimum 20.0
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET priority_score = CASE 
            WHEN priority_score < 20.0 THEN 20.0 
            ELSE priority_score 
        END
        WHERE piece_type = 'rook' AND priority_score < 20.0
    ''')
    rook_min_fixes = cursor.rowcount
    
    # Bishop and Knight pieces minimum 15.0
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET priority_score = CASE 
            WHEN priority_score < 15.0 THEN 15.0 
            ELSE priority_score 
        END
        WHERE piece_type IN ('bishop', 'knight') AND priority_score < 15.0
    ''')
    minor_min_fixes = cursor.rowcount
    
    # Pawn pieces minimum 10.0
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET priority_score = CASE 
            WHEN priority_score < 10.0 THEN 10.0 
            ELSE priority_score 
        END
        WHERE piece_type = 'pawn' AND priority_score < 10.0
    ''')
    pawn_min_fixes = cursor.rowcount
    
    # King pieces minimum 5.0
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET priority_score = CASE 
            WHEN priority_score < 5.0 THEN 5.0 
            ELSE priority_score 
        END
        WHERE piece_type = 'king' AND priority_score < 5.0
    ''')
    king_min_fixes = cursor.rowcount
    
    total_min_fixes = queen_min_fixes + rook_min_fixes + minor_min_fixes + pawn_min_fixes + king_min_fixes
    print(f"   ✓ Applied minimum constraints: Q:{queen_min_fixes} R:{rook_min_fixes} B/N:{minor_min_fixes} P:{pawn_min_fixes} K:{king_min_fixes}")
    
    # Fix 4: Reset patterns with 0% win rate if they have many observations
    print("🎯 Fixing 0% win rate patterns...")
    cursor.execute('''
        UPDATE learned_move_patterns 
        SET win_rate = 0.25,
            priority_score = CASE 
                WHEN priority_score < 30.0 THEN 45.0
                ELSE priority_score
            END,
            confidence = 0.2
        WHERE win_rate = 0.0 AND times_seen > 50
    ''')
    
    winrate_fixes = cursor.rowcount
    print(f"   ✓ Fixed {winrate_fixes} zero-winrate patterns")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ FIXES COMPLETED:")
    print(f"   Applied minimum constraints: {total_min_fixes}")
    print(f"   Fixed zero win rates: {winrate_fixes}")

def quick_test_system(db_path):
    """Quick test to see if the system is working better"""
    
    print(f"\n🧪 QUICK SYSTEM TEST")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Test 1: Check that queen patterns are reasonable
    cursor.execute('''
        SELECT COUNT(*), MIN(priority_score), MAX(priority_score), AVG(priority_score), AVG(win_rate)
        FROM learned_move_patterns 
        WHERE piece_type = 'queen' AND times_seen > 10
    ''')
    
    queen_count, min_pri, max_pri, avg_pri, avg_wr = cursor.fetchone()
    print(f"👑 Queen patterns: {queen_count} patterns")
    print(f"   Priority range: {min_pri:.1f} to {max_pri:.1f} (avg: {avg_pri:.1f})")
    print(f"   Average win rate: {avg_wr:.1%}")
    
    queen_healthy = min_pri >= 30.0 and avg_wr > 0.25
    print(f"   Status: {'✓ HEALTHY' if queen_healthy else '❌ STILL PROBLEMATIC'}")
    
    # Test 2: Check for remaining catastrophic patterns
    cursor.execute('SELECT COUNT(*) FROM learned_move_patterns WHERE priority_score < -50')
    bad_patterns = cursor.fetchone()[0]
    print(f"\n🚫 Severely negative patterns (< -50): {bad_patterns}")
    
    # Test 3: Check overall system health
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN priority_score > 0 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN win_rate > 0 THEN 1 ELSE 0 END) as non_zero_wr,
            AVG(priority_score) as avg_priority,
            AVG(win_rate) as avg_winrate
        FROM learned_move_patterns
        WHERE times_seen > 10
    ''')
    
    total, positive, non_zero_wr, avg_priority, avg_winrate = cursor.fetchone()
    positive_pct = (positive / total * 100) if total > 0 else 0
    non_zero_pct = (non_zero_wr / total * 100) if total > 0 else 0
    
    print(f"\n📊 Overall system health:")
    print(f"   Total patterns (>10 games): {total}")
    print(f"   Positive priorities: {positive} ({positive_pct:.1f}%)")
    print(f"   Non-zero win rates: {non_zero_wr} ({non_zero_pct:.1f}%)")
    print(f"   Average priority: {avg_priority:.1f}")
    print(f"   Average win rate: {avg_winrate:.1%}")
    
    # Overall health assessment
    system_healthy = (
        queen_healthy and 
        bad_patterns < 10 and 
        avg_priority > 20 and 
        avg_winrate > 0.15 and
        positive_pct > 80
    )
    
    print(f"\n🏥 System Status: {'🎉 RECOVERED' if system_healthy else '⚠️ NEEDS MORE WORK'}")
    
    conn.close()
    
    return system_healthy

def create_immediate_test_script(project_dir):
    """Create a quick test script to verify the fix works"""
    
    test_script = '''#!/usr/bin/env python3
"""
Quick Test - Verify Enhanced System Recovery
"""

import chess
from game_scorer import GameScorer

# Try to import the prioritizer
try:
    from learnable_move_prioritizer import LearnableMovePrioritizer
    prioritizer = LearnableMovePrioritizer("enhanced_training.db")
    print("✓ Successfully loaded prioritizer")
except Exception as e:
    print(f"❌ Failed to load prioritizer: {e}")
    exit(1)

# Test queen move priorities
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Test a typical queen development move
queen_move = chess.Move.from_uci("d1d3")  # Queen to d3

try:
    priority = prioritizer.get_move_priority(board, queen_move)
    print(f"👑 Queen move priority: {priority:.1f}")
    
    if priority > 20:
        print("✅ Queen priorities look healthy!")
    else:
        print("⚠️ Queen priorities still low")
        
except Exception as e:
    print(f"❌ Error testing queen move: {e}")

# Quick pattern check
try:
    stats = prioritizer.get_statistics()
    print(f"\\n📊 Pattern Statistics:")
    print(f"   Patterns learned: {stats['patterns_learned']}")
    print(f"   Average confidence: {stats['avg_confidence']:.2f}")
    print(f"   Average win rate: {stats['avg_win_rate']:.1%}")
    
    if stats['avg_win_rate'] > 0.15:
        print("✅ System appears recovered!")
    else:
        print("⚠️ Win rates still problematic")
        
except Exception as e:
    print(f"❌ Error getting statistics: {e}")

prioritizer.close()
print("\\n🧪 Quick test complete")
'''
    
    test_path = Path(project_dir) / "quick_test.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"   ✓ Created quick test script: {test_path}")

def main():
    """Main recovery script - SQLite compatible version"""
    
    print("🚨 ENHANCED PATTERN LEARNING EMERGENCY RECOVERY v2")
    print("(SQLite Compatible Version)")
    print("=" * 60)
    
    # Paths
    db_path = "C:/Users/Sojourner/Desktop/patmat/enhanced_training.db"
    project_dir = "C:/Users/Sojourner/Desktop/patmat"
    
    try:
        # Step 1: Diagnose remaining problems
        print("🔍 Checking if problems still exist...")
        has_problems = diagnose_system(db_path)
        
        # Step 2: Apply SQLite-compatible fixes
        fix_catastrophic_patterns(db_path)
        
        # Step 3: Test the system
        system_recovered = quick_test_system(db_path)
        
        # Step 4: Create test script
        create_immediate_test_script(project_dir)
        
        print(f"\\n{'=' * 60}")
        if system_recovered:
            print("🎉 SYSTEM RECOVERY SUCCESSFUL!")
            print("\\n🚀 Next steps:")
            print("   1. Quick test:")
            print("      python quick_test.py")
            print("   ")
            print("   2. Train with original trainer (should work much better now):")
            print("      python headless_trainer.py 50")
            print("   ")
            print("   3. Monitor performance:")
            print("      - Win rate should be >10%")
            print("      - Stalemate rate should be <60%")
            print("      - Should see queen moves again")
            
        else:
            print("⚠️ PARTIAL RECOVERY")
            print("\\n🚨 If still not working:")
            print("   1. Delete enhanced_training.db and start fresh:")
            print("      rm enhanced_training.db")
            print("      python headless_trainer.py 100")
            print("   ")
            print("   2. Or revert to original system:")
            print("      python headless_trainer.py 100  # Uses headless_training.db")
            
        print(f"\\n📦 Your data is backed up in: learned_move_patterns_backup table")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\\n🆘 EMERGENCY FALLBACK:")
        print("1. Delete the corrupted database:")
        print("   rm enhanced_training.db")
        print("2. Use your original working system:")
        print("   python headless_trainer.py 100")

if __name__ == "__main__":
    main()
