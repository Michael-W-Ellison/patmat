#!/usr/bin/env python3
"""
General Move Patterns - game-independent patterns shared by all games

Game-specific patterns describe a move with that game's vocabulary (piece
type, capture/check, distance...). General patterns describe a move only by
how it compares with the other moves available and with the opponent's
possible replies, so the same pattern means the same thing in every game:

- gain:   change in the game's own advantage measure caused by the move,
          compared with the other candidates (best / positive / zero / negative)
- net:    advantage after the opponent's best reply, compared with now (+ / 0 / -)
- mob:    opponent's number of replies compared with the median over the
          candidates (restricts / neutral / frees), or extra_turn / terminal
- threat: wins_now / draws_now / loses_now if the move ends the game;
          otherwise allows_loss / allows_draw if a reply ends it, else safe

All games record into one database, so evidence learned in one game is used
in the others. Each game's lift is measured against that game's own win
rate (win_lift scope = game name), so games with very different win rates
can share patterns. Forgetting and ranking are the same as for game-specific
patterns (see win_lift.py).

A game takes part by providing an adapter with:
    legal_moves(state, color) -> list of moves
    apply(state, move)        -> new state (state itself unchanged)
    advantage(state, color)   -> float, color's measure minus the opponent's,
                                 from the game's own scorer (higher = better)
    is_over(state)            -> bool
    winner(state)             -> winning color, or None (draw / not over)
    to_move(state)            -> color to move
    opponent(color)           -> the other color
Optionally, for games whose state is cheap to change in place but costly
to copy (chess), the adapter may also provide push(state, move) and
pop(state); apply() is then not used during feature extraction.
"""

import os
import random
import sqlite3
import statistics
from typing import Dict, List, Sequence

import win_lift

DB_NAME = 'universal_move_patterns.db'
MAX_REPLIES = 30       # opponent replies examined per candidate (sampled above this)
MAX_CANDIDATES = 12    # candidates a trainer should pass in (its best by its own score)
EPSILON = 1e-9


def universal_db_path(training_db_path: str) -> str:
    """The shared general-pattern database lives next to a game's training database"""
    return os.path.join(os.path.dirname(os.path.abspath(training_db_path)), DB_NAME)


def _sign_class(value: float) -> str:
    if value > EPSILON:
        return '+'
    if value < -EPSILON:
        return '-'
    return '0'


def extract_features(adapter, state, color, candidates: Sequence,
                     rng: random.Random = random, max_replies: int = MAX_REPLIES) -> List[str]:
    """
    Describe each candidate move with game-independent features.

    Returns:
        One feature string per candidate, in the same order
    """
    opp = adapter.opponent(color)
    adv_now = adapter.advantage(state, color)
    info = []

    in_place = hasattr(adapter, 'push') and hasattr(adapter, 'pop')

    def play(current, move):
        if in_place:
            adapter.push(current, move)
            return current
        return adapter.apply(current, move)

    def unplay(current):
        if in_place:
            adapter.pop(current)

    for move in candidates:
        after = play(state, move)
        gain = adapter.advantage(after, color) - adv_now
        entry = {'gain': gain, 'replies': None, 'net': gain, 'threat': 'safe', 'mob': None}

        if adapter.is_over(after):
            winner = adapter.winner(after)
            entry['threat'] = ('wins_now' if winner == color
                               else 'draws_now' if winner is None else 'loses_now')
            entry['mob'] = 'terminal'
        elif adapter.to_move(after) == color:
            entry['mob'] = 'extra_turn'
        else:
            replies = adapter.legal_moves(after, opp)
            entry['replies'] = len(replies)
            if len(replies) > max_replies:
                replies = rng.sample(list(replies), max_replies)

            worst = None
            for reply in list(replies):
                after_reply = play(after, reply)
                if adapter.is_over(after_reply):
                    winner = adapter.winner(after_reply)
                    if winner == opp:
                        entry['threat'] = 'allows_loss'
                    elif winner is None and entry['threat'] == 'safe':
                        entry['threat'] = 'allows_draw'
                value = adapter.advantage(after_reply, color)
                worst = value if worst is None else min(worst, value)
                unplay(after_reply)
            if worst is not None:
                entry['net'] = worst - adv_now
        unplay(after)
        info.append(entry)

    best_gain = max((e['gain'] for e in info), default=0.0)
    reply_counts = [e['replies'] for e in info if e['replies'] is not None]
    median_replies = statistics.median(reply_counts) if reply_counts else 0

    features = []
    for e in info:
        if e['gain'] > EPSILON:
            gain_class = 'best' if e['gain'] >= best_gain - EPSILON else 'positive'
        else:
            gain_class = 'zero' if e['gain'] >= -EPSILON else 'negative'

        mob = e['mob']
        if mob is None:
            if e['replies'] < median_replies:
                mob = 'restricts'
            elif e['replies'] > median_replies:
                mob = 'frees'
            else:
                mob = 'neutral'

        features.append(f"gain={gain_class}|net={_sign_class(e['net'])}|mob={mob}|threat={e['threat']}")
    return features


class GeneralPatternLearner:
    """Learns which game-independent move patterns help win, across all games"""

    TABLE = 'general_move_patterns'

    def __init__(self, db_path: str = DB_NAME):
        self.db_path = db_path
        # Several trainers may share this database; wait for locks instead of failing
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.cursor = self.conn.cursor()
        self.values: Dict[str, float] = {}

        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT NOT NULL UNIQUE,
                times_seen INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 50.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        win_lift.ensure_columns(self.cursor, self.TABLE)
        self.conn.commit()
        self._load_values()

    def _load_values(self):
        self.values = {
            features: 50.0 * lift
            for features, lift in self.cursor.execute(
                f"SELECT features, win_lift FROM {self.TABLE}")
        }

    def value(self, features: str) -> float:
        """
        Learned value of a general pattern, on the game-specific priority scale
        (50 * win lift, -50..+50, 0 = no evidence either way)
        """
        return self.values.get(features, 0.0)

    def record_game(self, features_used: List[str], result: str, game_name: str):
        """Credit the general patterns used in one game with its result"""
        counts: Dict[str, int] = {}
        for features in features_used:
            counts[features] = counts.get(features, 0) + 1

        ids = []
        for features, count in counts.items():
            self.cursor.execute(f'''
                INSERT INTO {self.TABLE} (features, times_seen) VALUES (?, ?)
                ON CONFLICT(features) DO UPDATE SET
                    times_seen = times_seen + excluded.times_seen,
                    updated_at = datetime('now')
            ''', (features, count))
            ids.append(self.cursor.execute(
                f"SELECT id FROM {self.TABLE} WHERE features = ?", (features,)).fetchone()[0])

        win_lift.record_game(self.cursor, self.TABLE, ids, won=(result == 'win'), scope=game_name)
        self.conn.commit()
        self._load_values()

    def get_top_patterns(self, limit: int = 10, avoid: bool = False) -> List[tuple]:
        """Best (or, with avoid=True, worst) general patterns by win lift"""
        order = 'ASC' if avoid else 'DESC'
        return self.cursor.execute(f'''
            SELECT features, times_seen, decayed_games, win_lift, confidence
            FROM {self.TABLE}
            WHERE confidence > 0.2
            ORDER BY win_lift {order}
            LIMIT ?
        ''', (limit,)).fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
