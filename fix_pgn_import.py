#!/usr/bin/env python3
"""
Quick fix for import_pgn_patterns.py database schema mismatch.
This updates the _update_pattern method to work with the new 7-column unique constraint.
"""

def fix_pgn_import_script():
    """Fix the PGN import script to work with new database schema"""
    import os
    script_path = "C:\\Users\\Sojourner\\Desktop\\patmat\\chess_pattern_ai\\import_pgn_patterns.py"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return
    
    # Read the original file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the problematic _update_pattern method
    old_update_method = '''        # Insert or update (use ON CONFLICT like LearnableMovePrioritizer)
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase)
            DO UPDATE SET'''
    
    new_update_method = '''        # Add default values for new schema columns (PGN doesn't have this context)
        repetition_count = 0  # Default: no repetition context in PGN
        moves_since_progress = 25  # Default: assume moderate progress
        total_material_level = 'medium'  # Default: assume mid-game material
        
        # Insert or update (use ON CONFLICT with new 7-column constraint)
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 repetition_count, moves_since_progress, total_material_level,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase,
                        repetition_count, moves_since_progress, total_material_level)
            DO UPDATE SET'''
    
    # Also need to fix the VALUES parameters
    old_values_line = '''        ''', (piece_type, move_category, distance, phase,
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score,'''
    
    new_values_line = '''        ''', (piece_type, move_category, distance, phase,
              repetition_count, moves_since_progress, total_material_level,
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score,'''
    
    # Find and fix the database initialization too
    old_db_init = '''            CREATE TABLE IF NOT EXISTS learned_move_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                distance_from_start INTEGER,
                game_phase TEXT,
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
                UNIQUE(piece_type, move_category, distance_from_start, game_phase)
            )'''
    
    new_db_init = '''            CREATE TABLE IF NOT EXISTS learned_move_patterns (
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
            )'''
    
    # Apply all fixes
    if old_update_method in content:
        content = content.replace(old_update_method, new_update_method)
        print("✓ Fixed _update_pattern method")
    else:
        print("⚠ Could not find _update_pattern method to fix")
    
    if old_values_line in content:
        content = content.replace(old_values_line, new_values_line)
        print("✓ Fixed VALUES parameters")
    else:
        print("⚠ Could not find VALUES line to fix")
    
    if old_db_init in content:
        content = content.replace(old_db_init, new_db_init)
        print("✓ Fixed database initialization")
    else:
        print("⚠ Could not find database init to fix")
    
    # Write back to file
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Updated {script_path}")
    print("The PGN import script should now work with the new database schema.")

if __name__ == '__main__':
    fix_pgn_import_script()
