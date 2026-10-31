#!/usr/bin/env python3
"""
Sequence Patterns - multi-move patterns learned from what follows them

A single-move pattern (learnable_move_prioritizer) describes one move. A
sequence pattern describes a run of consecutive plies, by both players,
ending with a move, e.g. "my knight move attacking two pieces, their king
move, my knight captures a rook". Each pattern is valued by what happened
after it in recorded games, from the point of view of the player who made
its last move:

- delta: that player's material change from just before the last move to
         HORIZON plies later (a checkmate within the horizon counts as
         +-MATE_VALUE), so a quiet move that sets up a later gain is credited
         with the gain
- lift:  the game result for that player (+1 / 0 / -1) minus that player's
         average result, so patterns are not credited just for being played
         by the stronger player

Both players' moves are recorded, so patterns the opponent uses to win
material are learned too (and avoided or used by the agent).

A ply is described only by observable relations on the board after it, from
the mover's side (see token()): the piece moved, what it captured, whether it
gives check, how many opponent pieces it then attacks, and whether it stands
attacked and undefended. No square or opening knowledge.

Evidence decays as in win_lift: after each recorded game all statistics are
multiplied by DECAY.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

from game_scorer import GameScorer
import win_lift

MAX_LENGTH = 3          # plies in the longest pattern
HORIZON = 4             # plies after a pattern's start over which its material change is measured
MATE_VALUE = 200.0      # a checkmate within the horizon, in material units (a pawn is 10)
LIFT_WEIGHT = 50.0      # material units per unit of result lift (+1: always won where the player usually lost)
PRIOR_SEEN = 5.0        # pseudo-occurrences at value 0: rarely seen patterns stay near 0
MIN_SEEN = 3.0          # (decayed) occurrences before a longer pattern is preferred to a shorter one

_LETTERS = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K'} if CHESS_AVAILABLE else {}


def token(board: 'chess.Board', move: 'chess.Move') -> str:
    """
    Observable description of a move from the mover's side (board before the move; unchanged on return):
    '<piece moved><x captured piece or -><+ if check> a<opponent pieces it attacks, 0-2> <s/a/h>'
    s: the moved piece is not attacked; a: attacked but defended; h: attacked and undefended
    """
    mover = board.turn
    piece = board.piece_at(move.from_square)
    if board.is_en_passant(move):
        captured = 'P'
    else:
        target = board.piece_at(move.to_square)
        captured = _LETTERS[target.piece_type] if target else None
    board.push(move)
    try:
        to = move.to_square
        check = '+' if board.is_check() else ''
        attacked = sum(1 for sq in board.attacks(to)
                       if board.piece_at(sq) is not None and board.piece_at(sq).color != mover)
        if not board.is_attacked_by(not mover, to):
            safety = 's'
        elif board.is_attacked_by(mover, to):
            safety = 'a'
        else:
            safety = 'h'
    finally:
        board.pop()
    moved = _LETTERS[move.promotion] if move.promotion else _LETTERS[piece.piece_type]
    return f"{moved}{'x' + captured if captured else '-'}{check} a{min(attacked, 2)} {safety}"


def material(board: 'chess.Board', color: 'chess.Color') -> float:
    """color's material minus the opponent's (GameScorer's piece values)"""
    values = GameScorer.PIECE_VALUES
    total = 0.0
    for sq, piece in board.piece_map().items():
        v = values[piece.symbol()]
        total += v if piece.color == color else -v
    return total


class SequencePatternLearner:
    """Learns the value of multi-move patterns (see the module docstring)"""

    def __init__(self, db_path: str = 'rule_discovery.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequence_patterns (
                pattern TEXT PRIMARY KEY,           -- tokens joined by ' | ', oldest first
                length INTEGER NOT NULL,            -- plies in the pattern
                seen REAL DEFAULT 0.0,              -- decayed occurrences
                delta_sum REAL DEFAULT 0.0,         -- decayed sum of material changes over the horizon
                lift_sum REAL DEFAULT 0.0           -- decayed sum of result lifts
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequence_baseline (
                mover TEXT PRIMARY KEY,             -- 'ai' or 'opponent'
                games REAL DEFAULT 0.0,
                result_sum REAL DEFAULT 0.0         -- decayed sum of results (+1 / 0 / -1)
            )
        ''')
        self.conn.commit()
        self.stats: Dict[str, List[float]] = {
            p: [s, d, l] for p, s, d, l in
            self.cursor.execute('SELECT pattern, seen, delta_sum, lift_sum FROM sequence_patterns')}
        self.baseline = {m: [g, r] for m, g, r in
                         self.cursor.execute('SELECT mover, games, result_sum FROM sequence_baseline')}

    # ------------------------------------------------------------------ use

    @staticmethod
    def context(board: 'chess.Board', plies: int = MAX_LENGTH - 1) -> List[str]:
        """Tokens of the last plies played on board (oldest first; board unchanged on return)"""
        undone = []
        while board.move_stack and len(undone) < plies:
            undone.append(board.pop())
        tokens = []
        for move in reversed(undone):
            tokens.append(token(board, move))
            board.push(move)
        return tokens

    def value(self, board: 'chess.Board', move: 'chess.Move', context: Optional[List[str]] = None) -> float:
        """
        Expected material gain (with the result lift) for the side to move from playing move, by the
        longest pattern ending with it seen at least MIN_SEEN times (shorter patterns otherwise)
        """
        if context is None:
            context = self.context(board)
        tokens = context + [token(board, move)]
        best = 0.0
        for length in range(1, min(MAX_LENGTH, len(tokens)) + 1):
            entry = self.stats.get(' | '.join(tokens[-length:]))
            if entry is None:
                break
            seen, delta, lift = entry
            if length > 1 and seen < MIN_SEEN:
                break
            best = (delta + LIFT_WEIGHT * lift) / (seen + PRIOR_SEEN)
        return best

    # -------------------------------------------------------------- learning

    def record_game(self, moves: List['chess.Move'], ai_color: 'chess.Color', result: str):
        """
        Learn from one finished game: every ply by either player ends patterns of length 1..MAX_LENGTH
        Args:
            moves: the game's moves from the starting position (board.move_stack)
            ai_color: the agent's colour (results are compared with each player's own average)
            result: 'win', 'loss' or 'draw' for the agent
        """
        board = chess.Board()
        tokens, balance = [], []                    # per ply: token, White's material before it
        for move in moves:
            balance.append(material(board, chess.WHITE))
            tokens.append(token(board, move))
            board.push(move)
        balance.append(material(board, chess.WHITE))
        n = len(moves)
        mated = board.is_checkmate()                # the side to move at the end was mated
        ai_result = {'win': 1.0, 'loss': -1.0}.get(result, 0.0)

        def results(color):
            return ai_result if color == ai_color else -ai_result

        def lift(color):
            mover = 'ai' if color == ai_color else 'opponent'
            games, total = self.baseline.get(mover, (0.0, 0.0))
            return results(color) - (total / games if games else 0.0)

        # forgetting
        for entry in self.stats.values():
            entry[0] *= win_lift.DECAY
            entry[1] *= win_lift.DECAY
            entry[2] *= win_lift.DECAY
        lifts = {c: lift(c) for c in (chess.WHITE, chess.BLACK)}

        for i in range(n):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            end = min(i + HORIZON, n)
            sign = 1.0 if color == chess.WHITE else -1.0
            delta = sign * (balance[end] - balance[i])
            if mated and n - i <= HORIZON:
                # the side that made the last move gave mate
                delta += MATE_VALUE if (n - 1 - i) % 2 == 0 else -MATE_VALUE
            for length in range(1, min(MAX_LENGTH, i + 1) + 1):
                key = ' | '.join(tokens[i - length + 1:i + 1])
                entry = self.stats.setdefault(key, [0.0, 0.0, 0.0])
                entry[0] += 1.0
                entry[1] += delta
                entry[2] += lifts[color]

        for color in (chess.WHITE, chess.BLACK):
            mover = 'ai' if color == ai_color else 'opponent'
            games, total = self.baseline.get(mover, (0.0, 0.0))
            self.baseline[mover] = [games * win_lift.DECAY + 1.0, total * win_lift.DECAY + results(color)]
        self._save()

    def _save(self):
        self.cursor.executemany(
            'INSERT OR REPLACE INTO sequence_patterns (pattern, length, seen, delta_sum, lift_sum) '
            'VALUES (?, ?, ?, ?, ?)',
            [(p, p.count(' | ') + 1, s, d, l) for p, (s, d, l) in self.stats.items()])
        self.cursor.executemany(
            'INSERT OR REPLACE INTO sequence_baseline (mover, games, result_sum) VALUES (?, ?, ?)',
            [(m, g, r) for m, (g, r) in self.baseline.items()])
        self.conn.commit()

    def top_patterns(self, limit: int = 10, min_length: int = 2, avoid: bool = False) -> List[Tuple[str, float, float]]:
        """(pattern, value, occurrences) of the most valuable (avoid: least) patterns of min_length plies or more"""
        rows = [(p, (d + LIFT_WEIGHT * l) / (s + PRIOR_SEEN), s) for p, (s, d, l) in self.stats.items()
                if p.count(' | ') + 1 >= min_length and s >= MIN_SEEN]
        rows.sort(key=lambda r: r[1], reverse=not avoid)
        return rows[:limit]

    def close(self):
        self.conn.close()
