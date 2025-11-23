import sqlite3

def fix_database_schema():
    """Fix the unique constraint in the learned_move_patterns table"""
    conn = sqlite3.connect('headless_training.db')
    cursor = conn.cursor()
    
    print("Backing up existing data...")
    
    # Create backup table with current data
    cursor.execute('''
        CREATE TABLE learned_move_patterns_backup AS 
        SELECT * FROM learned_move_patterns
    ''')
    
    print("Dropping existing table...")
    cursor.execute('DROP TABLE learned_move_patterns')
    
    print("Creating new table with correct unique constraint...")
    cursor.execute('''
        CREATE TABLE learned_move_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            piece_type TEXT NOT NULL,
            move_category TEXT NOT NULL,
            distance_from_start INTEGER,
            game_phase TEXT,
            repetition_count INTEGER,
            moves_since_progress INTEGER,
            total_material_level TEXT,
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
    
    print("Restoring data to new table...")
    # Copy data back, handling the new unique constraint
    cursor.execute('''
        INSERT OR IGNORE INTO learned_move_patterns
        SELECT * FROM learned_move_patterns_backup
    ''')
    
    print("Creating performance index...")
    cursor.execute('''
        CREATE INDEX idx_move_pattern_priority 
        ON learned_move_patterns(priority_score DESC, confidence DESC)
    ''')
    
    print("Cleaning up backup...")
    cursor.execute('DROP TABLE learned_move_patterns_backup')
    
    # Check results
    cursor.execute('SELECT COUNT(*) FROM learned_move_patterns')
    count = cursor.fetchone()[0]
    print(f"✓ Database updated successfully! {count} patterns restored.")
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    fix_database_schema()
