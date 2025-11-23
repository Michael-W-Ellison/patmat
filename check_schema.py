#!/usr/bin/env python3
import sqlite3

# Connect to database
conn = sqlite3.connect('headless_training.db')
cursor = conn.cursor()

# Get table schema
print("Table schema:")
cursor.execute('PRAGMA table_info(learned_move_patterns)')
for row in cursor.fetchall():
    print(f'  {row}')

print()

# Get CREATE statement
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
schema = cursor.fetchone()
print('CREATE statement:')
print(schema[0] if schema else 'Table not found')

print()

# Check for indexes/unique constraints
cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='learned_move_patterns'")
indexes = cursor.fetchall()
if indexes:
    print('Indexes:')
    for idx in indexes:
        print(f'  {idx[0]}')
else:
    print('No indexes found')

conn.close()
