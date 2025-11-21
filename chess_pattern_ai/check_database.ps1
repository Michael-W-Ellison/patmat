# Quick PGN Import Diagnostic for Windows
# Run this in PowerShell to check your database

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Database Diagnostic" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check headless_training.db
if (Test-Path "headless_training.db") {
    $dbSize = (Get-Item "headless_training.db").Length
    Write-Host "Database: headless_training.db" -ForegroundColor Green
    Write-Host "  Size: $($dbSize/1KB) KB"

    if ($dbSize -eq 0) {
        Write-Host "  ERROR: Database is empty (0 bytes)!" -ForegroundColor Red
        Write-Host "  The import did not write any data."
    } elseif ($dbSize -lt 100000) {
        Write-Host "  WARNING: Database is very small!" -ForegroundColor Yellow
        Write-Host "  Expected: 50-100 MB for 1M games"
        Write-Host "  Actual: $($dbSize/1KB) KB"
    }

    Write-Host ""

    # Check database contents
    $pythonCheck = @"
import sqlite3
import os

db_path = 'headless_training.db'
db_size = os.path.getsize(db_path)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
    if not cursor.fetchone():
        print('  ERROR: No learned_move_patterns table found!')
        print('  The database structure was not created.')
    else:
        cursor.execute('SELECT COUNT(*), SUM(times_seen), SUM(games_won), SUM(games_lost), SUM(games_drawn) FROM learned_move_patterns')
        count, seen, won, lost, drawn = cursor.fetchone()

        if count == 0:
            print('  ERROR: Database has 0 patterns!')
            print('  Import did not save any data.')
        else:
            total_games = (won or 0) + (lost or 0) + (drawn or 0)
            print(f'  Patterns: {count:,}')
            print(f'  Observations: {seen:,}')
            print(f'  Games: {total_games:,}')

            if total_games > 0:
                win_rate = (won or 0) / total_games * 100
                print(f'  Win rate: {win_rate:.1f}%')

                if count < 100:
                    print(f'  WARNING: Very few patterns! Expected 400+ for 1M games.')
                if seen and seen < 1000000:
                    print(f'  WARNING: Very few observations! Expected 60M+ for 1M games.')

    conn.close()
except Exception as e:
    print(f'  ERROR: {e}')
"@

    python -c $pythonCheck
} else {
    Write-Host "ERROR: headless_training.db not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Recommendations" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$dbSize = 0
if (Test-Path "headless_training.db") {
    $dbSize = (Get-Item "headless_training.db").Length
}

if ($dbSize -lt 1000000) {
    Write-Host "Your database is too small. This means:" -ForegroundColor Yellow
    Write-Host "  1. Import was interrupted or crashed"
    Write-Host "  2. PGN file format is incorrect"
    Write-Host "  3. Import script has a bug"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run: .\chess_pattern_ai\test_pgn_import.ps1 your_games.pgn"
    Write-Host "  2. This tests with 100 games to find the issue"
    Write-Host "  3. If test passes, do full import with --limit removed"
} else {
    Write-Host "Database size looks good!" -ForegroundColor Green
    Write-Host "Run training to test: python chess_pattern_ai/headless_trainer.py 100"
}
