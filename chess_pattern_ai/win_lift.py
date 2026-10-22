#!/usr/bin/env python3
"""
Win Lift - decayed "does using this pattern help win?" statistics

Shared by the game-specific pattern table (learnable_move_prioritizer) and
the game-independent pattern table (general_move_patterns).

For every pattern a table row keeps, with exponential forgetting:
- decayed_games: games the pattern was used in
- decayed_wins:  of those, games that were won
- lift_sum:      sum over those games of (won - baseline win rate)

Forgetting: after every recorded game all stored evidence is multiplied by
DECAY, so evidence from HALF_LIFE_GAMES games ago counts half as much.

Rank: win_lift = lift_sum / (decayed_games + PRIOR_GAMES)
- > 0: games using the pattern are won more often than usual (attempt it)
- < 0: games using the pattern are won less often than usual (avoid it)
- rarely used patterns stay near 0 until there is evidence
priority_score = 50 + 50 * win_lift (0-100, 50 = no evidence either way)
"""

import sqlite3
from typing import Iterable

HALF_LIFE_GAMES = 250
DECAY = 0.5 ** (1.0 / HALF_LIFE_GAMES)
PRIOR_GAMES = 10.0      # shrinks the lift of rarely used patterns toward 0
BASELINE_PRIOR = 2.0    # pseudo-games at a 50% win rate before any games are recorded

STAT_COLUMNS = {
    'decayed_games': 'REAL DEFAULT 0.0',
    'decayed_wins': 'REAL DEFAULT 0.0',
    'lift_sum': 'REAL DEFAULT 0.0',
    'win_lift': 'REAL DEFAULT 0.0',
}


def ensure_columns(cursor: sqlite3.Cursor, table: str):
    """Add the decayed statistics columns to an existing pattern table"""
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for name, decl in STAT_COLUMNS.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS win_lift_baseline (
            scope TEXT PRIMARY KEY,         -- what the baseline is for (a table or a game)
            decayed_games REAL DEFAULT 0.0,
            decayed_wins REAL DEFAULT 0.0,
            games_recorded INTEGER DEFAULT 0
        )
    ''')


def baseline(cursor: sqlite3.Cursor, scope: str) -> float:
    """Decayed win rate of all recorded games in scope"""
    row = cursor.execute(
        "SELECT decayed_games, decayed_wins FROM win_lift_baseline WHERE scope = ?",
        (scope,)).fetchone()
    games, wins = row if row else (0.0, 0.0)
    return (wins + 0.5 * BASELINE_PRIOR) / (games + BASELINE_PRIOR)


def record_game(cursor: sqlite3.Cursor, table: str, row_ids: Iterable[int],
                won: bool, scope: str):
    """
    Apply one finished game: forget a little, then credit the patterns used.

    Args:
        table: pattern table holding the STAT_COLUMNS
        row_ids: ids of the rows (patterns) used in this game; each is credited once
        won: whether the game was won
        scope: baseline scope; lift is measured against this scope's win rate
    """
    win = 1.0 if won else 0.0

    # Forgetting: uniform decay keeps baseline ratios, shrinks old evidence
    cursor.execute(f'''
        UPDATE {table}
        SET decayed_games = decayed_games * ?,
            decayed_wins = decayed_wins * ?,
            lift_sum = lift_sum * ?
    ''', (DECAY, DECAY, DECAY))
    cursor.execute('''
        UPDATE win_lift_baseline
        SET decayed_games = decayed_games * ?, decayed_wins = decayed_wins * ?
    ''', (DECAY, DECAY))

    # Lift is measured against the win rate before this game
    base = baseline(cursor, scope)
    cursor.executemany(f'''
        UPDATE {table}
        SET decayed_games = decayed_games + 1.0,
            decayed_wins = decayed_wins + ?,
            lift_sum = lift_sum + ?
        WHERE id = ?
    ''', [(win, win - base, row_id) for row_id in set(row_ids)])

    cursor.execute('''
        INSERT INTO win_lift_baseline (scope, decayed_games, decayed_wins, games_recorded)
        VALUES (?, 1.0, ?, 1)
        ON CONFLICT(scope) DO UPDATE SET
            decayed_games = decayed_games + 1.0,
            decayed_wins = decayed_wins + excluded.decayed_wins,
            games_recorded = games_recorded + 1
    ''', (scope, win))

    refresh_ranks(cursor, table)


def refresh_ranks(cursor: sqlite3.Cursor, table: str):
    """Recompute win_lift, confidence and priority_score from the decayed statistics"""
    cursor.execute(f'''
        UPDATE {table}
        SET win_lift = lift_sum / (decayed_games + ?),
            confidence = decayed_games / (decayed_games + ?),
            priority_score = 50.0 + 50.0 * lift_sum / (decayed_games + ?)
    ''', (PRIOR_GAMES, PRIOR_GAMES, PRIOR_GAMES))
