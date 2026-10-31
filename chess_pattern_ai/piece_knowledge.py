#!/usr/bin/env python3
"""
Piece Knowledge - piece categories learned by watching moves being played

Nothing about how pieces move is given: every move played (by either player)
is observed as a displacement of the piece that moved, seen from the mover's
side (rows counted toward the opponent, so both colours' pieces of a kind are
one kind of piece). From these observations each kind of piece is described
by how it has been seen to move:

- geometry: o (along a row or column), d (along a diagonal), j (any other
            displacement: a jump), for its ordinary moves; and for its
            captures, when they differ (':' then the capture geometry)
- range:    L if it has been seen to move 3 or more squares, else S
- f:        it has only been seen to move forward

e.g. 'oL' (moves along rows/columns, long range), 'jS' (jumps, short range),
'o:dSf' (moves along a column, captures diagonally, short range, forward
only). Kinds of piece seen fewer than MIN_OBSERVED times are 'u' (unknown).
The category, not the piece's identity, is what sequence patterns record, so
pieces that behave alike share patterns, and a piece's category is refined as
more of its moves are seen.
"""

import json
import sqlite3
from typing import Dict

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

MIN_OBSERVED = 10       # moves of a kind of piece seen before it is categorized
LONG_RANGE = 3          # squares moved that make a piece long range


def displacement_class(dr: int, dc: int) -> str:
    if dr == 0 or dc == 0:
        return 'o'
    if abs(dr) == abs(dc):
        return 'd'
    return 'j'


class PieceKnowledge:
    """How each kind of piece has been seen to move, and the category that describes it"""

    def __init__(self, db_path: str = 'rule_discovery.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS piece_observations (
                piece INTEGER PRIMARY KEY,          -- the kind of piece as the board shows it
                data TEXT NOT NULL                  -- JSON: what its observed moves had in common
            )
        ''')
        self.conn.commit()
        self.seen: Dict[int, Dict] = {p: json.loads(d) for p, d in
                                      self.cursor.execute('SELECT piece, data FROM piece_observations')}
        self._categories: Dict[int, str] = {}

    @staticmethod
    def _empty() -> Dict:
        return {'moves': 0, 'move_geometry': [], 'capture_geometry': [], 'longest': 0, 'not_forward': False,
                'captures': 0, 'checks': 0}

    def observe(self, board: 'chess.Board', move: 'chess.Move'):
        """Watch one move being played (board before the move)"""
        piece = board.piece_at(move.from_square)
        if piece is None:
            return
        sign = 1 if piece.color == chess.WHITE else -1
        dr = sign * (chess.square_rank(move.to_square) - chess.square_rank(move.from_square))
        dc = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        capture = board.is_capture(move)
        s = self.seen.setdefault(piece.piece_type, self._empty())
        s['moves'] += 1
        geometry = s['capture_geometry' if capture else 'move_geometry']
        g = displacement_class(dr, dc)
        if g not in geometry:
            geometry.append(g)
        s['longest'] = max(s['longest'], max(abs(dr), abs(dc)))
        s['not_forward'] = s['not_forward'] or dr <= 0
        s['captures'] += capture
        s['checks'] += board.gives_check(move)
        self._categories.pop(piece.piece_type, None)

    def category(self, piece_type: int) -> str:
        """The learned category of a kind of piece ('u' until it has been seen MIN_OBSERVED times)"""
        c = self._categories.get(piece_type)
        if c is None:
            s = self.seen.get(piece_type)
            if s is None or s['moves'] < MIN_OBSERVED:
                c = 'u'
            else:
                moves = ''.join(sorted(s['move_geometry']))
                caps = ''.join(sorted(s['capture_geometry']))
                c = moves + (':' + caps if caps and caps != moves else '')
                c += 'L' if s['longest'] >= LONG_RANGE and set(s['move_geometry'] + s['capture_geometry']) - {'j'} else 'S'
                c += '' if s['not_forward'] else 'f'
            self._categories[piece_type] = c
        return c

    def save(self):
        self.cursor.executemany('INSERT OR REPLACE INTO piece_observations (piece, data) VALUES (?, ?)',
                                [(p, json.dumps(s)) for p, s in self.seen.items()])
        self.conn.commit()

    def describe(self) -> Dict[str, Dict]:
        """{piece symbol: {category, moves seen, share of captures, share of checks}}"""
        out = {}
        for p, s in sorted(self.seen.items()):
            out[chess.piece_symbol(p).upper()] = {
                'category': self.category(p), 'moves': s['moves'],
                'captures': round(s['captures'] / s['moves'], 2), 'checks': round(s['checks'] / s['moves'], 2)}
        return out

    def close(self):
        self.conn.close()
