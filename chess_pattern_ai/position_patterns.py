#!/usr/bin/env python3
"""
Position Patterns - the value of a position, learned from what followed it

A position is described, from the side to move's point of view, by a
pattern for each piece that is attacked or attacks a piece of the other side:

    '<m: mine / t: theirs> <the piece's learned category (piece_knowledge)>
     <s/a/c/h> a<pieces of the other side it attacks, 0-2><'>' if one is worth more than it>'

    s: no piece of the other side attacks it; a: it is attacked, but defended and
    no cheaper piece attacks it; c: a cheaper piece attacks it; h: it is
    attacked and undefended

(a piece neither attacked nor attacking, 's a0', shows no pattern). Each
pattern counts once however many pieces show it, and once more (as
'<pattern> 2+') if two or more do, so a decisive pattern shown by one piece
(e.g. an undefended queen) weighs as much in learning as one every pawn shows.
Nothing else about the game is given; piece worth is the game's own scorer's
(GameScorer).

What material already covers is learned separately, by control features
(names starting with '='): a constant, the material balance, and which kinds
of piece (categories) each side has. They are fitted together with the
patterns but left out of a position's value, so a pattern's weight is what it
adds beyond the material on the board (e.g. "the side to move still has its
queen" is a control, not a pattern, and the search counts that material
itself).

The value of a position for the side to move is the sum of its patterns'
learned weights: the material that side is expected to gain over the next
HORIZON plies (checkmate counts as MATE_VALUE), plus LIFT_WEIGHT times how
much better than usual that side's game result was (as in sequence_patterns),
beyond what the controls predict. The weights are fitted to recorded games by
a normalized least-mean-squares update, one step per position.

Every position of every recorded game is learned from, whichever player is
to move, so the agent learns from its opponent's play (e.g. Stockfish's: what
the positions it reaches and lets the agent reach have led to) as much as
from its own.
"""

import sqlite3
from typing import Dict, List

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

from piece_knowledge import PieceKnowledge
from sequence_patterns import HORIZON, LIFT_WEIGHT, MATE_VALUE, material, worth

LEARNING_RATE = 0.1     # normalized LMS step size
CACHE_LIMIT = 200_000   # positions whose value is kept between searches
MATERIAL_SCALE = 50.0   # the material-balance control: balance / MATERIAL_SCALE (a rook is 1)
CONTROL = '='           # first character of a control feature's name


def is_control(name: str) -> bool:
    return name[0] == CONTROL


def features(board: 'chess.Board', pieces: PieceKnowledge) -> Dict[str, float]:
    """{pattern or control: its value} for the position, seen by the side to move"""
    mover = board.turn
    out: Dict[str, float] = {'=bias': 1.0, '=material': material(board, mover) / MATERIAL_SCALE}
    occupied = {chess.WHITE: board.occupied_co[chess.WHITE], chess.BLACK: board.occupied_co[chess.BLACK]}
    for sq, piece in board.piece_map().items():
        color = piece.color
        side = 'm' if color == mover else 't'
        category = pieces.category(piece.piece_type)
        has = f"={side} {category}"
        if has in out:
            out[has + ' 2+'] = 1.0
        else:
            out[has] = 1.0
        value = worth(piece.piece_type)
        attackers = board.attackers_mask(not color, sq)
        if not attackers:
            safety = 's'
        elif min(worth(board.piece_type_at(a)) for a in chess.scan_forward(attackers)) < value:
            safety = 'c'
        elif board.attackers_mask(color, sq):
            safety = 'a'
        else:
            safety = 'h'
        targets = board.attacks_mask(sq) & occupied[not color]
        count = chess.popcount(targets)
        richer = '>' if count and any(worth(board.piece_type_at(t)) > value
                                      for t in chess.scan_forward(targets)) else ''
        if safety == 's' and not count:
            continue
        key = f"{side} {category} {safety} a{min(count, 2)}{richer}"
        if key in out:
            out[key + ' 2+'] = 1.0
        else:
            out[key] = 1.0
    return out


class PositionPatternLearner:
    """Learns the value of positions from the patterns in them (see the module docstring)"""

    def __init__(self, db_path: str = 'rule_discovery.db', pieces: PieceKnowledge = None):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_patterns (
                pattern TEXT PRIMARY KEY,
                weight REAL DEFAULT 0.0,            -- material expected over the horizon, per occurrence
                seen REAL DEFAULT 0.0               -- positions it occurred in
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_baseline (
                mover TEXT PRIMARY KEY,             -- 'ai' or 'opponent'
                games REAL DEFAULT 0.0,
                result_sum REAL DEFAULT 0.0         -- sum of results (+1 / 0 / -1)
            )
        ''')
        self.conn.commit()
        self.own_pieces = pieces is None             # (closed here only if made here)
        self.pieces = pieces if pieces is not None else PieceKnowledge(db_path)
        self.weights: Dict[str, List[float]] = {
            p: [w, s] for p, w, s in self.cursor.execute('SELECT pattern, weight, seen FROM position_patterns')}
        self.baseline = {m: [g, r] for m, g, r in
                         self.cursor.execute('SELECT mover, games, result_sum FROM position_baseline')}
        self.cache: Dict = {}

    # ------------------------------------------------------------------ use

    def features(self, board: 'chess.Board') -> Dict[str, float]:
        return features(board, self.pieces)

    def value(self, board: 'chess.Board') -> float:
        """
        Material the side to move is expected to gain over the next plies (with the result lift) from the
        position's patterns, beyond what its material predicts (the controls are left out)
        """
        key = board._transposition_key()
        v = self.cache.get(key)
        if v is None:
            weights = self.weights
            v = 0.0
            for f, n in self.features(board).items():
                entry = weights.get(f)
                if entry is not None and not is_control(f):
                    v += entry[0] * n
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.clear()
            self.cache[key] = v
        return v

    # -------------------------------------------------------------- learning

    def record_game(self, moves: List['chess.Move'], ai_color: 'chess.Color', result: str, judge=None):
        """
        Learn from one finished game: every position, with either player to move
        Args:
            moves: the game's moves from the starting position (board.move_stack)
            ai_color: the agent's colour
            result: 'win', 'loss' or 'draw' for the agent
            judge: a chess judge (move_judges.StockfishJudge): if given, a position's target is instead the
                judge's evaluation of it for the side to move, in material units (a pawn is 10, at most
                +-MATE_VALUE), minus that side's material balance: what the judge sees beyond material
        """
        board = chess.Board()
        positions, balance = [], []                 # per ply: features before it, White's material before it
        judged = []                                 # per ply, with a judge: its evaluation for White
        for move in moves:
            positions.append(self.features(board))
            balance.append(material(board, chess.WHITE))
            if judge is not None:
                judged.append(max(-MATE_VALUE, min(MATE_VALUE, judge.evaluate(board) / 10.0)))
            board.push(move)
        balance.append(material(board, chess.WHITE))
        n = len(moves)
        mated = board.is_checkmate()
        ai_result = {'win': 1.0, 'loss': -1.0}.get(result, 0.0)

        def results(color):
            return ai_result if color == ai_color else -ai_result

        def lift(color):
            games, total = self.baseline.get('ai' if color == ai_color else 'opponent', (0.0, 0.0))
            return results(color) - (total / games if games else 0.0)

        lifts = {c: lift(c) for c in (chess.WHITE, chess.BLACK)}
        for i, feats in enumerate(positions):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            end = min(i + HORIZON, n)
            sign = 1.0 if color == chess.WHITE else -1.0
            if judge is not None:
                target = sign * (judged[i] - balance[i])
            else:
                target = sign * (balance[end] - balance[i]) + LIFT_WEIGHT * lifts[color]
                if mated and n - i <= HORIZON:
                    target += MATE_VALUE if (n - 1 - i) % 2 == 0 else -MATE_VALUE
            entries = [(self.weights.setdefault(f, [0.0, 0.0]), c) for f, c in feats.items()]
            error = target - sum(e[0] * c for e, c in entries)
            step = LEARNING_RATE * error / sum(c * c for _, c in entries)
            for e, c in entries:
                e[0] += step * c
                e[1] += 1.0

        for color in (chess.WHITE, chess.BLACK):
            mover = 'ai' if color == ai_color else 'opponent'
            games, total = self.baseline.get(mover, (0.0, 0.0))
            self.baseline[mover] = [games + 1.0, total + results(color)]
        self.cache.clear()
        self._save()

    def _save(self):
        self.cursor.executemany('INSERT OR REPLACE INTO position_patterns (pattern, weight, seen) VALUES (?, ?, ?)',
                                [(p, w, s) for p, (w, s) in self.weights.items()])
        self.cursor.executemany('INSERT OR REPLACE INTO position_baseline (mover, games, result_sum) VALUES (?, ?, ?)',
                                [(m, g, r) for m, (g, r) in self.baseline.items()])
        self.conn.commit()

    def top_patterns(self, limit: int = 10, min_seen: float = 20.0, worst: bool = False, controls: bool = False):
        """(pattern, weight, positions seen) with the highest (worst: lowest) weights (controls: of the controls)"""
        rows = [(p, w, s) for p, (w, s) in self.weights.items() if s >= min_seen and is_control(p) == controls]
        rows.sort(key=lambda r: r[1], reverse=not worst)
        return rows[:limit]

    def close(self):
        if self.own_pieces:
            self.pieces.close()
        self.conn.close()
