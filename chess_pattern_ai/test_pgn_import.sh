#!/bin/bash
# Test PGN Import - Diagnostic Script

echo "=========================================="
echo "PGN Import Diagnostic"
echo "=========================================="
echo ""

# Check if PGN file exists
if [ -z "$1" ]; then
    echo "Usage: bash chess_pattern_ai/test_pgn_import.sh your_games.pgn"
    echo ""
    echo "This will:"
    echo "  1. Test import with first 100 games"
    echo "  2. Verify database is created correctly"
    echo "  3. Show what's in the database"
    exit 1
fi

PGN_FILE="$1"

if [ ! -f "$PGN_FILE" ]; then
    echo "❌ Error: PGN file not found: $PGN_FILE"
    echo ""
    echo "Try:"
    echo "  ls *.pgn              # List PGN files in current directory"
    echo "  find ~ -name '*.pgn'  # Find all PGN files"
    exit 1
fi

echo "✓ Found PGN file: $PGN_FILE"
PGN_SIZE=$(ls -lh "$PGN_FILE" | awk '{print $5}')
echo "  Size: $PGN_SIZE"
echo ""

# Check first few lines
echo "First 10 lines of PGN:"
echo "----------------------------------------"
head -10 "$PGN_FILE"
echo "----------------------------------------"
echo ""

# Test with small sample
echo "Testing import with first 100 games..."
echo "----------------------------------------"

# Remove old test database
rm -f test_pgn_import.db

# Run import
python3 chess_pattern_ai/import_pgn_patterns.py --db test_pgn_import.db --limit 100 "$PGN_FILE"

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

# Check if database exists and has data
if [ -f test_pgn_import.db ]; then
    DB_SIZE=$(ls -lh test_pgn_import.db | awk '{print $5}')
    echo "✓ Database created: test_pgn_import.db ($DB_SIZE)"

    # Check contents
    python3 -c "
import sqlite3
conn = sqlite3.connect('test_pgn_import.db')
cursor = conn.cursor()

try:
    cursor.execute('SELECT COUNT(*), SUM(times_seen), SUM(games_won), SUM(games_lost), SUM(games_drawn) FROM learned_move_patterns')
    count, seen, won, lost, drawn = cursor.fetchone()
    total_games = won + lost + drawn

    print(f'  Patterns: {count:,}')
    print(f'  Observations: {seen:,}')
    print(f'  Games: {total_games:,} (Won={won:,}, Lost={lost:,}, Drawn={drawn:,})')

    if total_games > 0:
        win_rate = won / total_games * 100
        print(f'  Win rate: {win_rate:.1f}%')

        if win_rate < 40 or win_rate > 60:
            print(f'  ⚠ WARNING: Win rate should be ~48-52% for balanced data!')
        else:
            print(f'  ✓ Win rate looks good!')

    if count == 0:
        print('  ❌ ERROR: No patterns were created!')
        print('  This means the PGN file may be corrupted or in wrong format.')
    elif count < 20:
        print('  ⚠ WARNING: Very few patterns created from 100 games!')
        print('  Expected: 40-60 patterns')
    else:
        print(f'  ✓ Pattern count looks reasonable!')

except Exception as e:
    print(f'  ❌ ERROR reading database: {e}')

conn.close()
"
else
    echo "❌ ERROR: Database was not created!"
    echo "  This means the import failed completely."
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""

if [ -f test_pgn_import.db ]; then
    python3 -c "
import sqlite3
conn = sqlite3.connect('test_pgn_import.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM learned_move_patterns')
count = cursor.fetchone()[0]
conn.close()

if count > 0:
    print('✓ Test import successful!')
    print('')
    print('To import ALL games:')
    print('  1. rm headless_training.db  # Remove empty database')
    print('  2. python3 chess_pattern_ai/import_pgn_patterns.py --db headless_training.db $PGN_FILE')
    print('  3. Wait ~20-30 minutes for 1M games')
    print('  4. ls -lh headless_training.db  # Should be ~50-100 MB')
    print('  5. python3 chess_pattern_ai/headless_trainer.py 100  # Test AI')
else:
    print('❌ Import failed - see errors above')
" PGN_FILE="$PGN_FILE"
else
    echo "Import failed. Check:"
    echo "  1. Is the PGN file in correct format?"
    echo "  2. Does it contain game results (1-0, 0-1, 1/2-1/2)?"
    echo "  3. Are there any error messages above?"
fi

echo ""
