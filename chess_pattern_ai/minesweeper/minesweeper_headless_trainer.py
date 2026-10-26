#!/usr/bin/env python3
"""
Minesweeper baseline trainer

Two players, both choosing only from what the player can see:
- random:  reveals a random hidden cell
- learner: describes every hidden cell by its visible surroundings and
           reveals the cell whose description has the highest win lift
           (win_lift.py: decayed "games using it are won more often than
           usual"), exploring a random cell with probability EXPLORE

A cell's description (feature string):
    rev  revealed neighbours: 0, 1-2, 3-5 or 6+
    min  smallest number among them (- if none)
    max  largest number among them (- if none)
    pos  corner, edge or inner

Usage:
    python3 -m minesweeper.minesweeper_headless_trainer [games] [--player random|learner]
        [--db path] [--seed n] [--rows 9 --cols 9 --mines 10] [--block 250]
"""

import argparse
import os
import random
import sqlite3
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import win_lift  # noqa: E402

from minesweeper.minesweeper_adapter import PLAYER, MinesweeperAdapter  # noqa: E402
from minesweeper.minesweeper_game import HIDDEN  # noqa: E402

TABLE = 'minesweeper_patterns'
SCOPE = 'minesweeper'
EXPLORE = 0.1


def cell_features(obs: List[List[int]], pos) -> str:
    rows, cols = len(obs), len(obs[0])
    r, c = pos
    numbers = [obs[r + dr][c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols and obs[r + dr][c + dc] != HIDDEN]
    n = len(numbers)
    rev = '0' if n == 0 else '1-2' if n <= 2 else '3-5' if n <= 5 else '6+'
    border = (r in (0, rows - 1)) + (c in (0, cols - 1))
    pos_name = 'corner' if border == 2 else 'edge' if border == 1 else 'inner'
    lo, hi = (min(numbers), max(numbers)) if numbers else ('-', '-')
    return f'rev={rev}|min={lo}|max={hi}|pos={pos_name}'


class MinesweeperLearner:
    def __init__(self, db_path: str, rng: random.Random, explore: float = EXPLORE):
        self.conn = sqlite3.connect(db_path)
        self.rng, self.explore = rng, explore
        cur = self.conn.cursor()
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id INTEGER PRIMARY KEY,
                features TEXT UNIQUE,
                times_used INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 50.0
            )''')
        win_lift.ensure_columns(cur, TABLE)
        self.conn.commit()
        self.used: set = set()

    def _priority(self, features: str) -> float:
        row = self.conn.execute(f'SELECT priority_score FROM {TABLE} WHERE features = ?', (features,)).fetchone()
        return row[0] if row else 50.0

    def choose(self, obs, moves):
        described = [(m, cell_features(obs, m)) for m in moves]
        if self.rng.random() < self.explore:
            move, features = self.rng.choice(described)
        else:
            scores = {}
            for _, f in described:
                if f not in scores:
                    scores[f] = self._priority(f)
            best = max(scores.values())
            move, features = self.rng.choice([(m, f) for m, f in described if scores[f] == best])
        self.used.add(features)
        return move

    def record_game(self, won: bool):
        cur = self.conn.cursor()
        ids = []
        for f in self.used:
            cur.execute(f'INSERT OR IGNORE INTO {TABLE} (features) VALUES (?)', (f,))
            cur.execute(f'UPDATE {TABLE} SET times_used = times_used + 1 WHERE features = ?', (f,))
            ids.append(cur.execute(f'SELECT id FROM {TABLE} WHERE features = ?', (f,)).fetchone()[0])
        win_lift.record_game(cur, TABLE, ids, won, SCOPE)
        self.conn.commit()
        self.used = set()

    def close(self):
        self.conn.close()


def play_game(adapter: MinesweeperAdapter, seed: int, choose) -> bool:
    """One game; choose(observation, moves) picks a move from what the player can see"""
    game = adapter.new_game(seed)
    while not adapter.is_over(game):
        move = choose(adapter.observe(game, PLAYER), adapter.legal_moves(game, PLAYER))
        game = adapter.apply(game, move)
    return adapter.winner(game) == PLAYER


def train(games: int, player: str = 'learner', db_path: Optional[str] = None, seed: int = 0,
          rows: int = 9, cols: int = 9, mines: int = 10, block: int = 250, verbose: bool = True) -> List[float]:
    """Play `games` games; returns the win rate of each block of `block` games"""
    adapter = MinesweeperAdapter(rows, cols, mines)
    rng = random.Random(seed)
    learner = MinesweeperLearner(db_path or 'minesweeper_training.db', rng) if player == 'learner' else None
    choose = learner.choose if learner else (lambda obs, moves: rng.choice(moves))
    rates, wins = [], 0
    for i in range(games):
        won = play_game(adapter, seed * 1_000_003 + i, choose)
        wins += won
        if learner:
            learner.record_game(won)
        if (i + 1) % block == 0 or i + 1 == games:
            n = (i % block) + 1
            rates.append(wins / n)
            if verbose:
                print(f'games {i + 2 - n}-{i + 1}: won {wins}/{n} ({100 * wins / n:.1f}%)', flush=True)
            wins = 0
    if learner:
        learner.close()
    return rates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('games', type=int, nargs='?', default=1000)
    ap.add_argument('--player', choices=('random', 'learner'), default='learner')
    ap.add_argument('--db', default='minesweeper_training.db')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rows', type=int, default=9)
    ap.add_argument('--cols', type=int, default=9)
    ap.add_argument('--mines', type=int, default=10)
    ap.add_argument('--block', type=int, default=250)
    args = ap.parse_args()
    train(args.games, args.player, args.db, args.seed, args.rows, args.cols, args.mines, args.block)


if __name__ == '__main__':
    main()
