#!/usr/bin/env python3
"""
Fix Unique Constraint - Complete the Database Migration

The columns were added but the unique constraint wasn't updated.
This script completes the migration by updating the constraint.
"""

import sqlite3
import sys
from pathlib import Path

def fix_unique_constraint(db_path):
    """Fix the unique constraint to include enhanced columns"""
    
    print("🔧 FIXING UNIQUE CONSTRAINT")
    print("=" * 50)
    
    # Create backup first
    backup_path = db_path.replace('.db', '_constraint_backup.db')
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"📦 Backup created: {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("🔄 Creating new table with correct constraint...")
        
        # Create new table with enhanced schema and correct constraint
        cursor.execute('''
            CREATE TABLE learned_move_patterns_fixed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                distance_from_start INTEGER,
                game_phase TEXT,
                repetition_count INTEGER DEFAULT 0,
                moves_since_progress INTEGER DEFAULT 25,
                total_material_level TEXT DEFAULT 'medium',
                times_seen INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                games_lost INTEGER DEFAULT 0,
                games_drawn INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_score REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(piece_type, move_category, distance_from_start, game_phase,
                       repetition_count, moves_since_progress, total_material_level)
            )
        ''')
        
        print("✅ New table created with enhanced constraint")
        
        print("📋 Copying existing data...")
        
        # Copy all data from old table to new table
        cursor.execute('''
            INSERT INTO learned_move_patterns_fixed 
            SELECT * FROM learned_move_patterns
        ''')
        
        copied_rows = cursor.rowcount
        print(f"✅ Copied {copied_rows} rows")
        
        print("🔄 Replacing old table...")
        
        # Replace old table with new table
        cursor.execute('DROP TABLE learned_move_patterns')
        cursor.execute('ALTER TABLE learned_move_patterns_fixed RENAME TO learned_move_patterns')
        
        print("✅ Table replacement complete")
        
        # Verify the fix
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
        new_schema = cursor.fetchone()[0]
        
        if 'repetition_count, moves_since_progress, total_material_level' in new_schema:
            print("✅ Constraint fix verified!")
            success = True
        else:
            print("❌ Constraint fix failed verification")
            success = False
        
        conn.commit()
        
    except Exception as e:
        print(f"❌ Constraint fix failed: {e}")
        conn.rollback()
        success = False
        
    finally:
        conn.close()
    
    return success

def test_fixed_constraint():
    """Test that the constraint fix worked"""
    
    print("\n🧪 TESTING FIXED CONSTRAINT")
    print("=" * 50)
    
    try:
        # Try importing and using the prioritizer
        import sys
        sys.path.append('chess_pattern_ai')
        
        from learnable_move_prioritizer import LearnableMovePrioritizer
        
        prioritizer = LearnableMovePrioritizer("headless_training.db")
        print("✅ Prioritizer loads without errors")
        
        stats = prioritizer.get_statistics()
        print(f"📊 Patterns: {stats['patterns_learned']}")
        print(f"📊 Avg win rate: {stats['avg_win_rate']:.1%}")
        
        prioritizer.close()
        
        print("✅ Database works with enhanced code!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Fix the unique constraint issue"""
    
    print("🛠️ CONSTRAINT FIX: Complete Database Migration")
    print("This will fix the unique constraint mismatch")
    print("=" * 70)
    
    db_path = "C:/Users/Sojourner/Desktop/patmat/headless_training.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # Fix the constraint
        fix_success = fix_unique_constraint(db_path)
        
        if not fix_success:
            print("❌ Constraint fix failed")
            return False
        
        # Test the fix
        test_success = test_fixed_constraint()
        
        print(f"\n{'=' * 70}")
        
        if test_success:
            print("🎉 CONSTRAINT FIX SUCCESSFUL!")
            print("\n✅ Database fully migrated to enhanced schema")
            print("✅ Unique constraint updated")
            print("✅ Compatible with enhanced code")
            
            print(f"\n🚀 Ready to train:")
            print(f"   python chess_pattern_ai/headless_trainer.py 50")
            
            print(f"\nExpected behavior:")
            print(f"   - No more constraint errors")
            print(f"   - Enhanced pattern learning")
            print(f"   - Tracks repetition, progress, material levels")
            
        else:
            print("⚠️ CONSTRAINT FIX HAD ISSUES")
            print("You may need to restore from backup and try again")
            
        print("=" * 70)
        return test_success
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
