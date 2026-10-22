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
in the others, weighted lower than a game's own evidence (see
GeneralPatternLearner). Each game's lift is measured against that game's own
win rate (win_lift scope = game name), so games with very different win
rates can share patterns. Forgetting and ranking are the same as for game-specific
patterns (see win_lift.py).

A game takes part by providing an adapter with:
    legal_moves(state, color) -> list of moves
    apply(state, move)        -> new state (state itself unchanged)
    advantage(state, color)   -> float, color's measure minus the opponent's,
                                 from the game's own scorer (higher = better);
                                 it should measure progress toward the game's
                                 winning condition, not only material
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
WIN_VALUE = 1_000_000.0  # value of a won position; above any adapter's advantage


def universal_db_path(training_db_path: str) -> str:
    """The shared general-pattern database lives next to a game's training database"""
    return os.path.join(os.path.dirname(os.path.abspath(training_db_path)), DB_NAME)


def _sign_class(value: float) -> str:
    if value > EPSILON:
        return '+'
    if value < -EPSILON:
        return '-'
    return '0'


def evaluate(adapter, state, color) -> float:
    """
    Value of a position for color, judged first by the game's winning condition:
    a finished game is worth +WIN_VALUE (won), -WIN_VALUE (lost) or 0 (draw);
    otherwise the adapter's advantage measure.
    """
    if adapter.is_over(state):
        winner = adapter.winner(state)
        if winner is None:
            return 0.0
        return WIN_VALUE if winner == color else -WIN_VALUE
    return adapter.advantage(state, color)


def analyze_candidates(adapter, state, color, candidates: Sequence,
                       rng: random.Random = random,
                       max_replies: int = MAX_REPLIES) -> List[Dict]:
    """
    Evaluate each candidate move one ply deep and describe it with
    game-independent features.

    Returns:
        One dict per candidate, in the same order:
        - 'features': feature string (see the module docstring)
        - 'gain': evaluate() change caused by the move
        - 'net':  evaluate() change after the opponent's best examined reply
                  (equal to gain when the move ends the game or keeps the turn)
    """
    opp = adapter.opponent(color)
    value_now = evaluate(adapter, state, color)
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
        entry = {'replies': None, 'threat': 'safe', 'mob': None}

        if adapter.is_over(after):
            winner = adapter.winner(after)
            entry['threat'] = ('wins_now' if winner == color
                               else 'draws_now' if winner is None else 'loses_now')
            entry['mob'] = 'terminal'
            entry['gain'] = entry['net'] = evaluate(adapter, after, color) - value_now
        else:
            entry['gain'] = entry['net'] = adapter.advantage(after, color) - value_now
            if adapter.to_move(after) == color:
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
                            value = -WIN_VALUE
                        elif winner is None:
                            if entry['threat'] == 'safe':
                                entry['threat'] = 'allows_draw'
                            value = 0.0
                        else:
                            value = WIN_VALUE
                    else:
                        value = adapter.advantage(after_reply, color)
                    worst = value if worst is None else min(worst, value)
                    unplay(after_reply)
                if worst is not None:
                    entry['net'] = worst - value_now
        unplay(after)
        info.append(entry)

    best_gain = max((e['gain'] for e in info), default=0.0)
    reply_counts = [e['replies'] for e in info if e['replies'] is not None]
    median_replies = statistics.median(reply_counts) if reply_counts else 0

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

        e['features'] = f"gain={gain_class}|net={_sign_class(e['net'])}|mob={mob}|threat={e['threat']}"
    return [{'features': e['features'], 'gain': e['gain'], 'net': e['net']} for e in info]


def decisive_value(net: float) -> float:
    """
    The part of an analyze_candidates() 'net' value decided by the winning
    condition: about +WIN_VALUE for a move that wins, -WIN_VALUE for one that
    lets the opponent win at once, else 0 (game scales are left alone)
    """
    return net if abs(net) >= WIN_VALUE / 2 else 0.0


def winning_moves(adapter, state, color, moves: Sequence) -> List:
    """Moves that achieve the game's winning condition immediately"""
    in_place = hasattr(adapter, 'push') and hasattr(adapter, 'pop')
    wins = []
    for move in moves:
        if in_place:
            adapter.push(state, move)
            after = state
        else:
            after = adapter.apply(state, move)
        if adapter.is_over(after) and adapter.winner(after) == color:
            wins.append(move)
        if in_place:
            adapter.pop(state)
    return wins


def extract_features(adapter, state, color, candidates: Sequence,
                     rng: random.Random = random, max_replies: int = MAX_REPLIES) -> List[str]:
    """
    Describe each candidate move with game-independent features.

    Returns:
        One feature string per candidate, in the same order
    """
    return [entry['features'] for entry in
            analyze_candidates(adapter, state, color, candidates, rng, max_replies)]


class GeneralPatternLearner:
    """
    Learns which game-independent move patterns help win, in each game and
    across games.

    Evidence is kept per (pattern, game). A game ranks a pattern mainly by
    its own evidence; evidence from the other games is weighted lower: it
    enters as a prior worth at most CROSS_GAME_GAMES games at its average
    lift, however much of it there is. So imported evidence guides a game
    while its own evidence is scarce and gives way as its own grows:

        lift = (own lift_sum + imported mean lift * k) / (own games + k + PRIOR_GAMES)
        k    = min(imported games, CROSS_GAME_GAMES)

    With no imported evidence this is the plain win lift (see win_lift.py).
    """

    TABLE = 'general_pattern_evidence'
    LEGACY_TABLE = 'general_move_patterns'   # summed over games; imported as game ''
    CROSS_GAME_GAMES = 10.0

    def __init__(self, db_path: str = DB_NAME, game_name: str = ''):
        self.db_path = db_path
        self.game_name = game_name
        # Several trainers may share this database; wait for locks instead of failing
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.cursor = self.conn.cursor()
        self.values: Dict[str, float] = {}

        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT NOT NULL,
                game TEXT NOT NULL,             -- game the evidence came from ('' = unknown)
                times_seen INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 50.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(features, game)
            )
        ''')
        win_lift.ensure_columns(self.cursor, self.TABLE)
        self._import_legacy()
        self.conn.commit()
        self._load_values()

    def _import_legacy(self):
        """
        Databases from before per-game evidence hold one row per pattern summed
        over all games; that evidence cannot be split, so it is kept as
        evidence from an unknown game (imported evidence for every game)
        """
        legacy = self.cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (self.LEGACY_TABLE,)).fetchone()
        if not legacy or self.cursor.execute(f"SELECT 1 FROM {self.TABLE} LIMIT 1").fetchone():
            return
        self.cursor.execute(f'''
            INSERT INTO {self.TABLE} (features, game, times_seen, decayed_games,
                                      decayed_wins, lift_sum, win_lift, confidence, priority_score)
            SELECT features, '', times_seen, decayed_games, decayed_wins, lift_sum,
                   win_lift, confidence, priority_score
            FROM {self.LEGACY_TABLE}
        ''')

    def _load_values(self):
        own: Dict[str, List[float]] = {}
        imported: Dict[str, List[float]] = {}
        for features, game, games, lift_sum in self.cursor.execute(
                f"SELECT features, game, decayed_games, lift_sum FROM {self.TABLE}"):
            target = own if game == self.game_name else imported
            totals = target.setdefault(features, [0.0, 0.0])
            totals[0] += games
            totals[1] += lift_sum

        self.values = {}
        for features in own.keys() | imported.keys():
            own_games, own_sum = own.get(features, (0.0, 0.0))
            imp_games, imp_sum = imported.get(features, (0.0, 0.0))
            k = min(imp_games, self.CROSS_GAME_GAMES)
            prior = (imp_sum / imp_games) * k if imp_games > 0 else 0.0
            lift = (own_sum + prior) / (own_games + k + win_lift.PRIOR_GAMES)
            self.values[features] = 50.0 * lift

    def value(self, features: str) -> float:
        """
        Learned value of a general pattern for this game, on the game-specific
        priority scale (50 * win lift, -50..+50, 0 = no evidence either way)
        """
        return self.values.get(features, 0.0)

    def record_game(self, features_used: List[str], result: str, game_name: str = None):
        """Credit the general patterns used in one game of game_name (default: this learner's game)"""
        game = self.game_name if game_name is None else game_name
        counts: Dict[str, int] = {}
        for features in features_used:
            counts[features] = counts.get(features, 0) + 1

        ids = []
        for features, count in counts.items():
            self.cursor.execute(f'''
                INSERT INTO {self.TABLE} (features, game, times_seen) VALUES (?, ?, ?)
                ON CONFLICT(features, game) DO UPDATE SET
                    times_seen = times_seen + excluded.times_seen,
                    updated_at = datetime('now')
            ''', (features, game, count))
            ids.append(self.cursor.execute(
                f"SELECT id FROM {self.TABLE} WHERE features = ? AND game = ?",
                (features, game)).fetchone()[0])

        win_lift.record_game(self.cursor, self.TABLE, ids, won=(result == 'win'), scope=game)
        self.conn.commit()
        self._load_values()

    def get_top_patterns(self, limit: int = 10, avoid: bool = False) -> List[tuple]:
        """Best (or, with avoid=True, worst) general patterns for this game: (features, value)"""
        ranked = sorted(self.values.items(), key=lambda item: item[1], reverse=not avoid)
        return ranked[:limit]

    def close(self):
        if self.conn:
            self.conn.close()
