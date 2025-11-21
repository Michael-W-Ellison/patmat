# PGN Import Test - Windows PowerShell Version
param(
    [Parameter(Mandatory=$true)]
    [string]$PgnFile
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PGN Import Diagnostic (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if PGN file exists
if (-not (Test-Path $PgnFile)) {
    Write-Host "Error: PGN file not found: $PgnFile" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try:" -ForegroundColor Yellow
    Write-Host "  Get-ChildItem *.pgn                    # List PGN files"
    Write-Host "  Get-ChildItem -Recurse -Filter *.pgn   # Search for PGN files"
    exit 1
}

Write-Host "Found PGN file: $PgnFile" -ForegroundColor Green
$fileSize = (Get-Item $PgnFile).Length / 1MB
Write-Host "  Size: $($fileSize.ToString('0.0')) MB"
Write-Host ""

# Check first few lines
Write-Host "First 10 lines of PGN:" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Get-Content $PgnFile -First 10
Write-Host "----------------------------------------"
Write-Host ""

# Test with small sample
Write-Host "Testing import with first 100 games..." -ForegroundColor Cyan
Write-Host "----------------------------------------"

# Remove old test database
if (Test-Path "test_pgn_import.db") {
    Remove-Item "test_pgn_import.db"
}

# Run import
python chess_pattern_ai/import_pgn_patterns.py --db test_pgn_import.db --limit 100 $PgnFile

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if database exists and has data
if (Test-Path "test_pgn_import.db") {
    $dbSize = (Get-Item "test_pgn_import.db").Length
    Write-Host "Database created: test_pgn_import.db ($($dbSize/1KB) KB)" -ForegroundColor Green

    # Check contents
    $pythonCheck = @"
import sqlite3
conn = sqlite3.connect('test_pgn_import.db')
cursor = conn.cursor()

try:
    cursor.execute('SELECT COUNT(*), SUM(times_seen), SUM(games_won), SUM(games_lost), SUM(games_drawn) FROM learned_move_patterns')
    count, seen, won, lost, drawn = cursor.fetchone()
    total_games = won + lost + drawn if won else 0

    print(f'  Patterns: {count:,}')
    print(f'  Observations: {seen:,}')
    print(f'  Games: {total_games:,} (Won={won:,}, Lost={lost:,}, Drawn={drawn:,})')

    if total_games > 0:
        win_rate = won / total_games * 100
        print(f'  Win rate: {win_rate:.1f}%')

        if win_rate < 40 or win_rate > 60:
            print(f'  WARNING: Win rate should be ~48-52% for balanced data!')
        else:
            print(f'  Win rate looks good!')

    if count == 0:
        print(f'  ERROR: No patterns were created!')
        print(f'  This means the PGN file may be corrupted or in wrong format.')
    elif count < 20:
        print(f'  WARNING: Very few patterns created from 100 games!')
        print(f'  Expected: 40-60 patterns')
    else:
        print(f'  Pattern count looks reasonable!')

except Exception as e:
    print(f'  ERROR reading database: {e}')

conn.close()
"@

    python -c $pythonCheck

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Next Steps" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""

    # Check if test was successful
    $patternCount = python -c "import sqlite3; conn = sqlite3.connect('test_pgn_import.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM learned_move_patterns'); print(cursor.fetchone()[0]); conn.close()" 2>$null

    if ($patternCount -gt 0) {
        Write-Host "Test import successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To import ALL games:"
        Write-Host "  1. Remove-Item headless_training.db  # Remove old database"
        Write-Host "  2. python chess_pattern_ai/import_pgn_patterns.py --db headless_training.db `"$PgnFile`""
        Write-Host "  3. Wait ~20-30 minutes for 1M games"
        Write-Host "  4. Get-Item headless_training.db  # Should be ~50-100 MB"
        Write-Host "  5. python chess_pattern_ai/headless_trainer.py 100  # Test AI"
    } else {
        Write-Host "Import failed - see errors above" -ForegroundColor Red
    }
} else {
    Write-Host "ERROR: Database was not created!" -ForegroundColor Red
    Write-Host "  This means the import failed completely."
}

Write-Host ""
