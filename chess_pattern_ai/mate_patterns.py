#!/usr/bin/env python3
"""
Mate Patterns - what the surroundings of a king look like before it is mated

A game is won by checkmate, so the agent learns which surroundings of a king
are followed by its being mated. A king's surroundings are described from
its own side's point of view by its neighbouring squares:

    'f<free> b<blocked> a<attacked> e<off the board> <+ if in check>'

    free:      squares it could move to (empty or holding an undefended
               piece of the other side, and not attacked)
    blocked:   squares holding a piece of its own side
    attacked:  squares the other side attacks (with the king removed, so a
               square behind the king on a checking line counts as attacked)
    off board: squares beyond the edge

e.g. a king mated on the back rank behind its own pawns: 'f0 b3 a2 e3 +'.

For every position of every recorded game, both kings' surroundings are
counted (seen), and counted as followed by mate (mated) when that king is
checkmated within MATE_PLIES plies, so the agent learns from the mates the
opponent gives it as much as from the mates it gives. The mate rate of a
description is mated / (seen + PRIOR_SEEN); the value of a position for the
side to move is MATE_VALUE times (the other king's mate rate - its own
king's), in material units like the other learned values.
"""

import sqlite3
from typing import Dict, List

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

from sequence_patterns import MATE_VALUE

MATE_PLIES = 6          # a king mated within this many plies counts as mated after a position
PRIOR_SEEN = 5.0        # pseudo-observations without a mate: rarely seen surroundings stay near 0
CACHE_LIMIT = 200_000


def surroundings(board: 'chess.Board', color: 'chess.Color') -> str:
    """Description of the surroundings of color's king (board unchanged on return)"""
    king = board.king(color)
    if king is None:
        return 'none'
    enemy = not color
    free = blocked = attacked = 0
    rank, file = chess.square_rank(king), chess.square_file(king)
    edge = sum(1 for dr in (-1, 0, 1) for df in (-1, 0, 1) if (dr or df)
               and not (0 <= rank + dr < 8 and 0 <= file + df < 8))
    board._remove_piece_at(king)        # (the public remove_piece_at would clear the move stack)
    try:
        for sq in chess.scan_forward(chess.BB_KING_ATTACKS[king]):
            if board.color_at(sq) == color:
                blocked += 1
            elif board.is_attacked_by(enemy, sq):
                attacked += 1
            else:
                free += 1
        check = board.is_attacked_by(enemy, king)
    finally:
        board._set_piece_at(king, chess.KING, color)
    return f"f{free} b{blocked} a{attacked} e{edge}{' +' if check else ''}"


class MatePatternLearner:
    """Learns how often each surroundings of a king are followed by its being mated"""

    def __init__(self, db_path: str = 'rule_discovery.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mate_patterns (
                pattern TEXT PRIMARY KEY,           -- a king's surroundings (see surroundings())
                seen REAL DEFAULT 0.0,              -- positions it occurred in
                mated REAL DEFAULT 0.0              -- ... followed by that king's being mated within MATE_PLIES
            )
        ''')
        self.conn.commit()
        self.stats: Dict[str, List[float]] = {
            p: [s, m] for p, s, m in self.cursor.execute('SELECT pattern, seen, mated FROM mate_patterns')}
        self.cache: Dict = {}

    def rate(self, pattern: str) -> float:
        seen, mated = self.stats.get(pattern, (0.0, 0.0))
        return mated / (seen + PRIOR_SEEN)

    def value(self, board: 'chess.Board') -> float:
        """MATE_VALUE * (the other king's mate rate - the side to move's king's), in material units"""
        key = board._transposition_key()
        v = self.cache.get(key)
        if v is None:
            v = MATE_VALUE * (self.rate(surroundings(board, not board.turn)) -
                              self.rate(surroundings(board, board.turn)))
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.clear()
            self.cache[key] = v
        return v

    def record_game(self, moves: List['chess.Move']):
        """Learn from one finished game: both kings' surroundings in every position"""
        board = chess.Board()
        positions = []
        for move in moves:
            positions.append((surroundings(board, chess.WHITE), surroundings(board, chess.BLACK)))
            board.push(move)
        positions.append((surroundings(board, chess.WHITE), surroundings(board, chess.BLACK)))
        n = len(moves)
        mated_color = board.turn if board.is_checkmate() else None
        for i, pair in enumerate(positions):
            for color, pattern in zip((chess.WHITE, chess.BLACK), pair):
                entry = self.stats.setdefault(pattern, [0.0, 0.0])
                entry[0] += 1.0
                if color == mated_color and n - i <= MATE_PLIES:
                    entry[1] += 1.0
        self.cache.clear()
        self.cursor.executemany('INSERT OR REPLACE INTO mate_patterns (pattern, seen, mated) VALUES (?, ?, ?)',
                                [(p, s, m) for p, (s, m) in self.stats.items()])
        self.conn.commit()

    def top_patterns(self, limit: int = 10, min_seen: float = 5.0):
        """(pattern, mate rate, positions seen) with the highest mate rates"""
        rows = [(p, self.rate(p), s) for p, (s, m) in self.stats.items() if s >= min_seen]
        rows.sort(key=lambda r: -r[1])
        return rows[:limit]

    def close(self):
        self.conn.close()
