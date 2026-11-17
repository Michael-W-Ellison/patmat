#!/usr/bin/env python3
"""
Look-ahead Patterns - learning which positions hide a tactic, and reading them further

The chess search (pattern_search) reads a few plies, then only captures. A
tactic whose gain comes after the end of a line is missed: in 10 games
against Stockfish (blunder_analysis), 12 of the agent's 13 blunders lost their
material for good only from the third ply of the punishing line on (forks,
discovered attacks, exchanges that go wrong), and in 12 of 13 the search had
read the punishing reply, but not far enough.

A player learns which positions call for calculating further. The program
learns it from its own games, in hindsight: after each game, at SAMPLE of its
positions (both sides to move), it compares what a glance shows (the capture
search alone, as at the end of a line) with what a deeper look shows (the
same search DEEP plies deep). Where the two differ by SWING or more (material
units; a pawn is 10), the position hid something a glance did not see. Each
position is described by its tension (signature):

    '<c|-> m<n><stake> t<n> p<n> l<n> o<n>'
    c        the side to move is in check
    m<n>     how many of its pieces the other side can win at once (threatened():
             attacked and undefended, or attacked by a cheaper piece; 0, 1, 2 = 2 or more),
             and the largest stake: P (under 15), M (15 to 39: about a minor piece), R (40 to 69), Q
    t<n>     how many of the other side's pieces it can win at once (0, 1, 2)
    p<n>     its pieces pinned to its king (0, 1 = 1 or more)
    l<n>     its loose pieces: knights, bishops, rooks and queens it does not defend (0, 1, 2)
    o<n>     the other side's loose pieces (0, 1, 2)

Learned: per signature, positions seen and positions that hid a swing. A
signature's rate is hidden / seen, shrunk toward the rate over all positions
by PRIOR positions.

Use (PatternSearch.lookahead): at the end of a line (depth 0), a position whose
signature's rate is at least EXTEND_RATE is read EXTENSION plies further
before the capture search, once per line. The deeper look of the lessons does
not use these extensions, so what it teaches does not depend on what was
learned.

Results (evaluate_lookahead; against Stockfish skill 0, each agent move judged
at depth 12): learned in 70 games (830 lessons; 35% of positions hid a swing;
rates by part: in check 0.64 against 0.32, two pieces threatened with a rook
at stake 0.65, a piece of the other side to win 0.45 against 0.31); EXTEND_RATE
chosen on those lessons (18% of them read further). Then 100 games per arm from
the same learned database:
- plain:     18 won, 82 lost; mean centipawn loss 67.5; blunders 3.2% (118 of
             3,657 moves); search depth reached 2.86, 1,201 positions per move
- lookahead: 11 won, 89 lost; 68.7; 3.4% (143 of 4,170); depth reached 2.37,
             1,900 positions per move, 40 read further per move
Reading further did not cut blunders, and it cost the search time: within the
0.5 s per move the 3-ply search finished less often, so more lines were read
less deep. Hence use_lookahead is off by default (headless_trainer).
"""

import random
import sqlite3
from typing import Dict, List, Optional

import chess

SAMPLE = 12             # positions of each game looked at again after it
DEEP = 4                # plies of the deeper look
LOOK_SECONDS = 3.0      # time limit of one deeper look (positions it does not finish are not lessons)
SWING = 20              # a deeper look differing from the glance by this much (a pawn is 10) hid something
PRIOR = 20.0            # pseudo-positions at the overall rate a signature's rate is shrunk toward
EXTEND_RATE = 0.38      # the search reads a position further when its signature's rate is at least this
EXTENSION = 2           # plies it reads it further


def _stake(value: float) -> str:
    return 'Q' if value >= 70 else 'R' if value >= 40 else 'M' if value >= 15 else 'P'


def _loose(board: chess.Board, color: chess.Color) -> int:
    pieces = board.occupied_co[color] & (board.knights | board.bishops | board.rooks | board.queens)
    return sum(1 for sq in chess.scan_forward(pieces) if not board.attackers_mask(color, sq))


def signature(board: chess.Board) -> str:
    """The tension of a position for the side to move (module docstring)"""
    from pattern_search import threatened
    side = board.turn
    mine = threatened(board, side)
    theirs = threatened(board, not side)
    pinned = any(board.is_pinned(side, sq)
                 for sq in chess.scan_forward(board.occupied_co[side] & ~board.kings))
    return (f"{'c' if board.is_check() else '-'} m{min(2, len(mine))}{_stake(mine[0][1]) if mine else ''} "
            f"t{min(2, len(theirs))} p{int(pinned)} l{min(2, _loose(board, side))} "
            f"o{min(2, _loose(board, not side))}")


class LookaheadLearner:
    """Per signature: positions seen, and positions where a deeper look differed from a glance by SWING"""

    def __init__(self, db_path: str, seed: Optional[int] = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS lookahead_patterns '
                          '(signature TEXT PRIMARY KEY, seen INTEGER, hidden INTEGER)')
        self.conn.commit()
        self.counts: Dict[str, List[int]] = {
            s: [n, h] for s, n, h in self.conn.execute('SELECT signature, seen, hidden FROM lookahead_patterns')}
        self.rng = random.Random(seed)
        self._rates: Optional[Dict[str, float]] = None
        self.lessons = 0

    # -------------------------------------------------------------- what was learned

    def overall(self) -> float:
        seen = sum(n for n, _ in self.counts.values())
        return sum(h for _, h in self.counts.values()) / seen if seen else 0.0

    def rate(self, sig: str) -> float:
        """How often positions of this signature hid a swing (shrunk toward the overall rate)"""
        if self._rates is None:
            self._rates = {}
        r = self._rates.get(sig)
        if r is None:
            n, h = self.counts.get(sig, (0, 0))
            r = self._rates[sig] = (h + PRIOR * self.overall()) / (n + PRIOR)
        return r

    def extend(self, board: chess.Board) -> bool:
        """Whether the search should read this position further"""
        return bool(self.counts) and self.rate(signature(board)) >= EXTEND_RATE

    # -------------------------------------------------------------- lessons

    def lesson(self, board: chess.Board, search) -> Optional[bool]:
        """Glance and deeper look at one position: whether it hid a swing (None: the look did not finish)"""
        glance = search.glance(board)
        deep = search.look(board, DEEP, LOOK_SECONDS)
        if deep is None:
            return None
        hidden = abs(deep - glance) >= SWING
        entry = self.counts.setdefault(signature(board), [0, 0])
        entry[0] += 1
        entry[1] += int(hidden)
        self.lessons += 1
        return hidden

    def record_game(self, moves: List[chess.Move], search) -> int:
        """After a game (its moves from the starting position): lessons at SAMPLE of its positions; how many"""
        board = chess.Board()
        positions = []
        for move in moves:
            board.push(move)
            if not board.is_game_over():
                positions.append(len(board.move_stack))
        chosen = set(self.rng.sample(positions, min(SAMPLE, len(positions))))
        board = chess.Board()
        done = 0
        for move in moves:
            board.push(move)
            if len(board.move_stack) in chosen and self.lesson(board, search) is not None:
                done += 1
        self._rates = None
        self.save()
        return done

    def save(self) -> None:
        self.conn.executemany('INSERT OR REPLACE INTO lookahead_patterns VALUES (?, ?, ?)',
                              [(s, n, h) for s, (n, h) in self.counts.items()])
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
