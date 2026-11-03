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

Games of other kinds declare it with optional attributes (absent: a
two-player game with everything visible):
    players = 1               single player; opponent(color) is a pseudo-player
                              for the environment, which never moves, and
                              winner() returns it when the game is lost
    hidden_information = True the state holds information the player may not
                              see; observe(state, color) returns what the
                              player sees, and move choice must use only that.
                              apply() is then the environment carrying out a
                              move, so analyze_candidates (which looks ahead
                              with apply) refuses such games
    new_game(seed)            a fresh state; games with a random setup draw it
                              from the seed, so a seed repeats a game
"""

import os
import random
import sqlite3
import statistics
from typing import Dict, List, Sequence

import win_lift
from refutation_patterns import (BLUNDER_MEDIANS, EXTENSION_DEPTH, REPLIES_EXTENDED, RefutationLearner,
                                 reply_features)

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
                       max_replies: int = MAX_REPLIES, refutations=None) -> List[Dict]:
    """
    Evaluate each candidate move one ply deep and describe it with
    game-independent features.

    With refutations (a refutation_patterns.RefutationLearner), each
    candidate's REPLIES_EXTENDED replies whose patterns most often refuted
    blunders are searched EXTENSION_DEPTH plies further, and 'net' becomes
    the lower of the two values (the features are those of the one-ply look).

    Returns:
        One dict per candidate, in the same order:
        - 'features': feature string (see the module docstring)
        - 'gain': evaluate() change caused by the move
        - 'net':  evaluate() change after the opponent's best examined reply
                  (equal to gain when the move ends the game or keeps the turn)
    """
    if getattr(adapter, 'hidden_information', False):
        raise ValueError('looking ahead with apply() would reveal hidden information')
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
                entry['net_plain'] = entry['net']
                if refutations is not None and replies:
                    extended = _extend_refutations(adapter, after, color, opp, list(replies), refutations)
                    if extended is not None:
                        entry['net'] = min(entry['net'], extended - value_now)
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

        e['features'] = (f"gain={gain_class}|net={_sign_class(e.get('net_plain', e['net']))}|mob={mob}"
                         f"|threat={e['threat']}")
    return [{'features': e['features'], 'gain': e['gain'], 'net': e['net']} for e in info]


def _extend_refutations(adapter, after, color, opp, replies, refutations):
    """
    color's lowest value after the replies (opp's, in after) rated likeliest to refute, each searched
    EXTENSION_DEPTH plies further; None if none is rated above average
    """
    from move_judges import SearchJudge
    chosen = refutations.ranked(reply_features(adapter, after, opp, replies), REPLIES_EXTENDED)
    if not chosen:
        return None
    search = SearchJudge(adapter, EXTENSION_DEPTH)
    worst = None
    for i in chosen:
        state = search._play(after, replies[i])
        try:
            v = search._search(state, color, EXTENSION_DEPTH, -float('inf'), float('inf'), False)
        finally:
            search._unplay(state)
        worst = v if worst is None else min(worst, v)
    return worst


def decisive_value(net: float) -> float:
    """
    The part of an analyze_candidates() 'net' value decided by the winning
    condition: about +WIN_VALUE for a move that wins, -WIN_VALUE for one that
    lets the opponent win at once, else 0 (game scales are left alone)
    """
    return net if abs(net) >= WIN_VALUE / 2 else 0.0


def winning_moves(adapter, state, color, moves: Sequence) -> List:
    """Moves that achieve the game's winning condition immediately"""
    if getattr(adapter, 'hidden_information', False):
        raise ValueError('looking ahead with apply() would reveal hidden information')
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
    A game created with import_cross_game=False uses only its own evidence
    (its evidence is still recorded and available to the other games).

    Judged evidence: with a judge (move_judges; self.judge, or JUDGE_DEPTH > 0
    for a SearchJudge over the trainer's adapter), every move passed to
    note_move() is also rated by the judge's loss for it: quality
    q = 1 - min(2, loss / m), where m is a running median of the game's
    nonzero losses, so the judge's best move is +1, a median loss 0 and twice
    the median or more -1 (a scale-free rating, comparable between games).
    A pattern's judged lift is its moves' mean quality minus the game's,
    shrunk toward 0 by JUDGED_PRIOR moves, with other games' evidence as a
    prior worth at most CROSS_GAME_MOVES moves (as for the win lift). Where
    any judged evidence is loaded, a pattern's value is the weighted mean
    of its win lift and judged lift (JUDGED_WEIGHT), each a dense signal for
    the other's sparse one: a game result rates all of a game's moves at once,
    the judge rates each move.
    """

    TABLE = 'general_pattern_evidence'
    LEGACY_TABLE = 'general_move_patterns'   # summed over games; imported as game ''
    CROSS_GAME_GAMES = 10.0
    LEARN_REFUTATIONS = False   # with a judge: learn which replies refute the agent's blunders (refutation_patterns)
    USE_REFUTATIONS = False     # analyze(): search the replies likeliest to refute further
    JUDGED_TABLE = 'general_judged_evidence'
    JUDGE_SCALE_TABLE = 'general_judge_scale'
    JUDGE_DEPTH = 0             # > 0: note_move() judges moves by a SearchJudge this deep (unless judge is set)
    JUDGED_PRIOR = 30.0         # moves at quality lift 0 that a pattern's judged lift is shrunk toward
    CROSS_GAME_MOVES = 300.0    # most moves other games' judged evidence counts as
    JUDGED_WEIGHT = 1.0         # weight of the judged lift against the win lift in value()
    SCALE_STEP = 1.05           # running median of nonzero losses: factor per move

    def __init__(self, db_path: str = DB_NAME, game_name: str = '',
                 import_cross_game: bool = True):
        self.db_path = db_path
        self.game_name = game_name
        self.import_cross_game = import_cross_game
        # Several trainers may share this database; wait for locks instead of failing
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.cursor = self.conn.cursor()
        self.values: Dict[str, float] = {}
        self.judge = None                   # a move judge (move_judges); see note_move
        self.pending: List[tuple] = []      # (features, judged loss) of moves noted since the last record_game
        self.last_loss = None               # the judged loss of the last move noted

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
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.JUDGED_TABLE} (
                features TEXT NOT NULL,
                game TEXT NOT NULL,
                moves REAL DEFAULT 0.0,         -- moves judged
                lift_sum REAL DEFAULT 0.0,      -- sum over them of (quality - the game's mean quality then)
                PRIMARY KEY (features, game)
            )
        ''')
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.JUDGE_SCALE_TABLE} (
                game TEXT PRIMARY KEY,
                median_loss REAL,               -- running median of the game's nonzero judged losses
                moves REAL DEFAULT 0.0,         -- moves judged
                quality_sum REAL DEFAULT 0.0
            )
        ''')
        self._import_legacy()
        self.conn.commit()
        self._load_values()
        self.refutations = RefutationLearner(self.conn, game_name, import_cross_game)

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
            if game == self.game_name:
                target = own
            elif self.import_cross_game:
                target = imported
            else:
                continue
            # the whole pattern, and each of its parts ('mob=restricts', ...) over all patterns that have it
            for key in [features] + features.split('|'):
                totals = target.setdefault(key, [0.0, 0.0])
                totals[0] += games
                totals[1] += lift_sum

        judged_own: Dict[str, List[float]] = {}
        judged_imported: Dict[str, List[float]] = {}
        for features, game, moves, lift_sum in self.cursor.execute(
                f"SELECT features, game, moves, lift_sum FROM {self.JUDGED_TABLE}"):
            if game == self.game_name:
                target = judged_own
            elif self.import_cross_game:
                target = judged_imported
            else:
                continue
            for key in [features] + features.split('|'):
                totals = target.setdefault(key, [0.0, 0.0])
                totals[0] += moves
                totals[1] += lift_sum

        def combined(own_totals, imported_totals, key, cap, prior_n):
            own_n, own_sum = own_totals.get(key, (0.0, 0.0))
            imp_n, imp_sum = imported_totals.get(key, (0.0, 0.0))
            k = min(imp_n, cap)
            prior = (imp_sum / imp_n) * k if imp_n > 0 else 0.0
            return (own_sum + prior) / (own_n + k + prior_n)

        judged = bool(judged_own or judged_imported)
        self.values = {}
        self.part_values = {}
        for key in own.keys() | imported.keys() | judged_own.keys() | judged_imported.keys():
            lift = combined(own, imported, key, self.CROSS_GAME_GAMES, win_lift.PRIOR_GAMES)
            if judged:
                judged_lift = combined(judged_own, judged_imported, key, self.CROSS_GAME_MOVES, self.JUDGED_PRIOR)
                lift = (lift + self.JUDGED_WEIGHT * judged_lift) / (1.0 + self.JUDGED_WEIGHT)
            (self.values if '|' in key else self.part_values)[key] = 50.0 * lift

    def part_value(self, part: str) -> float:
        """
        Learned value of one part of the general patterns (e.g. 'mob=restricts'), from the evidence of every
        pattern that has it, weighted between games as for whole patterns; same scale as value()
        """
        return self.part_values.get(part, 0.0)

    def value(self, features: str) -> float:
        """
        Learned value of a general pattern for this game, on the game-specific
        priority scale (50 * win lift, -50..+50, 0 = no evidence either way)
        """
        return self.values.get(features, 0.0)

    def note_move(self, adapter, state, color, move, features: str) -> str:
        """
        A trainer's move, before it is played, with its general pattern (returned unchanged). With a judge
        (self.judge, or a SearchJudge over adapter when JUDGE_DEPTH > 0) the move's judged loss is kept, and the
        next record_game() learns from it (except where the judge sees the position as lost whatever is
        played: last_loss is then None).
        """
        if self.judge is None and self.JUDGE_DEPTH > 0:
            from move_judges import SearchJudge
            self.judge = SearchJudge(adapter, self.JUDGE_DEPTH)
        if self.judge is not None and features is not None:
            self.last_loss = self.judge.move_loss(state, color, move)
            if self.last_loss is not None:          # (None: the judge sees the position as lost already)
                self.pending.append((features, self.last_loss))
                if self.LEARN_REFUTATIONS:
                    row = self.cursor.execute(f"SELECT median_loss FROM {self.JUDGE_SCALE_TABLE} WHERE game = ?",
                                              (self.game_name,)).fetchone()
                    if row and row[0] and self.last_loss >= BLUNDER_MEDIANS * row[0]:
                        self.learn_refutation(adapter, state, color, move)
        return features

    def learn_refutation(self, adapter, state, color, move):
        """The judge rated color's move a blunder: learn the patterns of the replies and of its best reply"""
        in_place = hasattr(adapter, 'push') and hasattr(adapter, 'pop')
        if in_place:
            adapter.push(state, move)
            after = state
        else:
            after = adapter.apply(state, move)
        try:
            opp = adapter.opponent(color)
            if not adapter.is_over(after) and adapter.to_move(after) == opp:
                replies = list(adapter.legal_moves(after, opp))
                if len(replies) > 1:
                    features = reply_features(adapter, after, opp, replies)
                    self.refutations.record(features, self.judge.best_reply(after, color, replies))
        finally:
            if in_place:
                adapter.pop(state)

    def analyze(self, adapter, state, color, candidates: Sequence, **kwargs) -> List[Dict]:
        """analyze_candidates(), with this game's refutation patterns when USE_REFUTATIONS is set"""
        if self.USE_REFUTATIONS and self.refutations.has_evidence():
            kwargs['refutations'] = self.refutations
        return analyze_candidates(adapter, state, color, candidates, **kwargs)

    def record_judged(self, judged: List[tuple], game_name: str = None):
        """Learn from judged moves of one game: [(features, judged loss)] (see the class docstring)"""
        game = self.game_name if game_name is None else game_name
        row = self.cursor.execute(
            f"SELECT median_loss, moves, quality_sum FROM {self.JUDGE_SCALE_TABLE} WHERE game = ?",
            (game,)).fetchone()
        median, moves, quality_sum = row if row else (None, 0.0, 0.0)
        totals: Dict[str, List[float]] = {}
        for features, loss in judged:
            if loss > EPSILON:
                if median is None:
                    median = loss
                else:
                    median = median * self.SCALE_STEP if loss > median else median / self.SCALE_STEP
            quality = 1.0 - min(2.0, loss / median) if median else 1.0
            mean = quality_sum / moves if moves else 0.0
            entry = totals.setdefault(features, [0.0, 0.0])
            entry[0] += 1.0
            entry[1] += quality - mean
            moves += 1.0
            quality_sum += quality
        self.cursor.executemany(f'''
            INSERT INTO {self.JUDGED_TABLE} (features, game, moves, lift_sum) VALUES (?, ?, ?, ?)
            ON CONFLICT(features, game) DO UPDATE SET
                moves = moves + excluded.moves, lift_sum = lift_sum + excluded.lift_sum
        ''', [(f, game, n, l) for f, (n, l) in totals.items()])
        self.cursor.execute(f"INSERT OR REPLACE INTO {self.JUDGE_SCALE_TABLE} VALUES (?, ?, ?, ?)",
                            (game, median, moves, quality_sum))
        self.conn.commit()

    def record_game(self, features_used: List[str], result: str, game_name: str = None):
        """
        Credit the general patterns used in one game of game_name (default: this learner's game), and learn
        from the moves judged since the last call (note_move)
        """
        game = self.game_name if game_name is None else game_name
        if self.pending:
            self.record_judged(self.pending, game)
            self.pending = []
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
