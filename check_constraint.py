import sqlite3

conn = sqlite3.connect('headless_training.db')
cursor = conn.cursor()

# Get table schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
result = cursor.fetchone()

if result:
    print('Current table constraint:')
    print(result[0])
    print()
else:
    print('Table not found')

# Check columns
cursor.execute('PRAGMA table_info(learned_move_patterns)')
columns = [col[1] for col in cursor.fetchall()]

enhanced_columns = ['repetition_count', 'moves_since_progress', 'total_material_level']
print('Column check:')
for col in enhanced_columns:
    present = col in columns
    print(f'  {col}: {"YES" if present else "NO"}')

conn.close()
