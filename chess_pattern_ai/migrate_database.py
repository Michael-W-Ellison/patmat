#!/usr/bin/env python3
"""
Database Migration Script: Add Enhanced Schema Columns

This script adds the missing columns to your existing headless_training.db
so it works with the enhanced code without losing any existing data.

Much cleaner than creating compatibility layers!
"""

import sqlite3
import sys
from pathlib import Path

def backup_database(db_path):
    """Create a backup of the database before migration"""
    
    backup_path = db_path.replace('.db', '_backup.db')
    
    try:
        # Simple file copy for SQLite
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None

def check_current_schema(db_path):
    """Check what columns currently exist"""
    
    print("🔍 Checking current database schema...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current columns
    cursor.execute("PRAGMA table_info(learned_move_patterns)")
    columns = cursor.fetchall()
    
    current_columns = [col[1] for col in columns]
    
    print(f"📊 Current columns ({len(current_columns)}):")
    for col in current_columns:
        print(f"   ✓ {col}")
    
    # Check which enhanced columns are missing
    enhanced_columns = ['repetition_count', 'moves_since_progress', 'total_material_level']
    missing_columns = [col for col in enhanced_columns if col not in current_columns]
    
    if missing_columns:
        print(f"\n❌ Missing enhanced columns ({len(missing_columns)}):")
        for col in missing_columns:
            print(f"   ✗ {col}")
    else:
        print(f"\n✅ All enhanced columns already present!")
    
    conn.close()
    return missing_columns

def add_enhanced_columns(db_path, missing_columns):
    """Add the missing enhanced schema columns"""
    
    if not missing_columns:
        print("✅ No columns to add - database already up to date!")
        return True
    
    print(f"\n🔧 Adding {len(missing_columns)} missing columns...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Add missing columns with appropriate defaults
        column_specs = {
            'repetition_count': 'INTEGER DEFAULT 0',
            'moves_since_progress': 'INTEGER DEFAULT 25', 
            'total_material_level': 'TEXT DEFAULT "medium"'
        }
        
        for column in missing_columns:
            if column in column_specs:
                spec = column_specs[column]
                sql = f"ALTER TABLE learned_move_patterns ADD COLUMN {column} {spec}"
                
                print(f"   Adding {column}...")
                cursor.execute(sql)
                print(f"   ✅ Added {column}")
            else:
                print(f"   ⚠️ Unknown column: {column}")
        
        # Update the unique constraint to include new columns
        print("\n🔄 Updating unique constraint...")
        
        # Check if we need to update the constraint
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
        create_sql = cursor.fetchone()[0]
        
        if 'repetition_count' not in create_sql:
            print("   Creating new table with updated constraint...")
            
            # Create new table with enhanced schema
            cursor.execute('''
                CREATE TABLE learned_move_patterns_new (
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
            
            # Copy data from old table to new table
            print("   Copying existing data...")
            cursor.execute('''
                INSERT INTO learned_move_patterns_new 
                (piece_type, move_category, distance_from_start, game_phase,
                 repetition_count, moves_since_progress, total_material_level,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
                SELECT 
                 piece_type, move_category, distance_from_start, game_phase,
                 COALESCE(repetition_count, 0),
                 COALESCE(moves_since_progress, 25), 
                 COALESCE(total_material_level, 'medium'),
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at
                FROM learned_move_patterns
            ''')
            
            # Replace old table with new table
            cursor.execute('DROP TABLE learned_move_patterns')
            cursor.execute('ALTER TABLE learned_move_patterns_new RENAME TO learned_move_patterns')
            print("   ✅ Table structure updated")
        
        conn.commit()
        print("✅ Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def verify_migration(db_path):
    """Verify the migration worked correctly"""
    
    print("\n🔍 Verifying migration...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check final schema
    cursor.execute("PRAGMA table_info(learned_move_patterns)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Check row count
    cursor.execute("SELECT COUNT(*) FROM learned_move_patterns")
    row_count = cursor.fetchone()[0]
    
    # Check that enhanced columns exist and have data
    enhanced_columns = ['repetition_count', 'moves_since_progress', 'total_material_level']
    all_present = all(col in columns for col in enhanced_columns)
    
    print(f"📊 Final schema: {len(columns)} columns")
    print(f"📊 Data preserved: {row_count} patterns")
    print(f"📊 Enhanced columns: {'✅ All present' if all_present else '❌ Some missing'}")
    
    # Sample a few rows to check data integrity
    cursor.execute("SELECT piece_type, move_category, times_seen, repetition_count, moves_since_progress, total_material_level FROM learned_move_patterns LIMIT 3")
    sample_rows = cursor.fetchall()
    
    print(f"\n📝 Sample data:")
    print(f"Piece      Category      Seen  RepCnt  MovSince  MatLvl")
    print("-" * 50)
    for row in sample_rows:
        piece, cat, seen, rep, mov, mat = row
        print(f"{piece:10} {cat:12} {seen:4d} {rep:6d} {mov:8d}  {mat}")
    
    conn.close()
    
    success = all_present and row_count > 0
    
    if success:
        print("\n🎉 MIGRATION VERIFICATION PASSED!")
    else:
        print("\n⚠️ Migration verification found issues")
    
    return success

def create_test_script(project_dir):
    """Create a test script to verify the enhanced code works"""
    
    test_script = '''#!/usr/bin/env python3
"""
Test Enhanced Code with Migrated Database
"""

import sys
sys.path.append('chess_pattern_ai')

try:
    from learnable_move_prioritizer import LearnableMovePrioritizer
    
    print("🧪 Testing enhanced code with migrated database...")
    
    # Initialize with migrated database
    prioritizer = LearnableMovePrioritizer("headless_training.db")
    print("✅ Enhanced prioritizer loaded successfully")
    
    # Get statistics
    stats = prioritizer.get_statistics()
    print(f"📊 Patterns: {stats['patterns_learned']}")
    print(f"📊 Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"📊 Avg win rate: {stats['avg_win_rate']:.1%}")
    
    prioritizer.close()
    
    print("\\n🎉 Enhanced code works with migrated database!")
    print("\\nReady to train:")
    print("   python chess_pattern_ai/headless_trainer.py 50")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Migration may not have completed successfully")
'''
    
    test_path = Path(project_dir) / "test_migration.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"✅ Created test script: {test_path}")

def main():
    """Main migration function"""
    
    print("🚀 DATABASE MIGRATION: ADD ENHANCED SCHEMA COLUMNS")
    print("This will add missing columns to your existing database")
    print("=" * 70)
    
    # Default database path
    db_path = "C:/Users/Sojourner/Desktop/patmat/headless_training.db"
    project_dir = "C:/Users/Sojourner/Desktop/patmat"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Please run this script from your project directory")
        return False
    
    print(f"📁 Database: {db_path}")
    
    try:
        # Step 1: Backup the database
        backup_path = backup_database(db_path)
        if not backup_path:
            print("❌ Cannot proceed without backup")
            return False
        
        # Step 2: Check current schema
        missing_columns = check_current_schema(db_path)
        
        # Step 3: Add missing columns
        migration_success = add_enhanced_columns(db_path, missing_columns)
        
        if not migration_success:
            print(f"❌ Migration failed - restore from backup: {backup_path}")
            return False
        
        # Step 4: Verify migration
        verification_success = verify_migration(db_path)
        
        # Step 5: Create test script
        create_test_script(project_dir)
        
        print(f"\n{'=' * 70}")
        if verification_success:
            print("🎉 DATABASE MIGRATION SUCCESSFUL!")
            print(f"\n✅ Your database now has the enhanced schema")
            print(f"✅ All existing data preserved ({backup_path})")
            print(f"✅ Compatible with enhanced code")
            
            print(f"\n🧪 Test the migration:")
            print(f"   python test_migration.py")
            
            print(f"\n🚀 Run enhanced training:")
            print(f"   python chess_pattern_ai/headless_trainer.py 50")
            
            print(f"\n📝 Benefits of enhanced schema:")
            print(f"   - Tracks repetition patterns")
            print(f"   - Monitors game progress") 
            print(f"   - Considers material levels")
            print(f"   - More sophisticated pattern learning")
            
        else:
            print("⚠️ MIGRATION HAD ISSUES")
            print(f"📦 Restore from backup if needed: {backup_path}")
            
        print(f"=" * 70)
        return verification_success
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        print(f"📦 Restore from backup: {backup_path if 'backup_path' in locals() else 'N/A'}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
