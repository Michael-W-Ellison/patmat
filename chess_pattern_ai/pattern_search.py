#!/usr/bin/env python3
"""
Pattern Search - a look-ahead guided by learned multi-move patterns

An alpha-beta search (iterative deepening within a time limit, then captures
only until the position is quiet) whose move order comes from the learned
sequence patterns (sequence_patterns): at every position the side to move's
moves are tried best pattern value first, and below the root only the
BREADTH best are searched, so the patterns decide which lines are read. The
value of a line is the value of the position at its end: its material (the
game's scorer's piece worth; checkmate is decisive) plus, if position patterns
are given, the material its learned position patterns (position_patterns)
expect the side to move to gain; among the agent's own moves, the pattern value is
added (PATTERN_WEIGHT), so the patterns choose between moves the search
finds equal.
"""

import time
from typing import List, Optional, Tuple

import chess

from position_patterns import PositionPatternLearner
from sequence_patterns import SequencePatternLearner, material, worth

MATE = 100_000.0
BREADTH = 8             # moves searched at each position below the root (best pattern value first)
QUIESCENCE_DEPTH = 6    # captures followed beyond the search depth
MAX_DEPTH = 8
PATTERN_WEIGHT = 0.5    # root moves: search value + PATTERN_WEIGHT * pattern value (both material units)


class OutOfTime(Exception):
    pass


class PatternSearch:

    def __init__(self, sequences: SequencePatternLearner, seconds: float = 0.5, use_patterns: bool = True,
                 positions: Optional[PositionPatternLearner] = None):
        self.sequences = sequences
        self.positions = positions          # None: positions are valued by material only
        self.seconds = seconds
        self.use_patterns = use_patterns    # False: order by capture value only (for comparison)
        self.nodes = 0
        self.depth_reached = 0

    # ------------------------------------------------------------- ordering

    def ordered(self, board: chess.Board, context: List[str], captures_only: bool = False) -> List[Tuple]:
        """(move, pattern value, token) for the side to move, best first"""
        moves = [m for m in board.legal_moves if not captures_only or board.is_capture(m)]
        out = []
        for m in moves:
            if self.use_patterns:
                tok = self.sequences.token(board, m)
                out.append((m, self.sequences.value(board, m, context, token=tok), tok))
            else:
                target = board.piece_at(m.to_square)
                out.append((m, worth(target.piece_type) if target else 0.0, None))
        out.sort(key=lambda e: -e[1])
        return out

    def evaluate(self, board: chess.Board) -> float:
        """Value of a (non-final) position for the side to move"""
        value = material(board, board.turn)
        if self.positions is not None:
            value += self.positions.value(board)
        return value

    # --------------------------------------------------------------- search

    def choose(self, board: chess.Board) -> Optional[chess.Move]:
        """The move for the side to move"""
        self.deadline = time.time() + self.seconds
        self.nodes = 0
        context = self.sequences.context(board) if self.use_patterns else []
        root = self.ordered(board, context)
        if not root:
            return None
        best_move = root[0][0]
        for depth in range(1, MAX_DEPTH + 1):
            try:
                scored = []
                best_total = -float('inf')
                for move, pattern, tok in root:
                    bonus = PATTERN_WEIGHT * pattern
                    board.push(move)
                    try:
                        # (only whether the move beats best_total matters: a null window below it)
                        value = -self.negamax(board, depth - 1, -float('inf'), -(best_total - bonus),
                                              self.next_context(context, tok), 1)
                    finally:
                        board.pop()
                    total = value + bonus
                    scored.append((total, move, pattern, tok))
                    if total > best_total:
                        best_total, best_move_now = total, move
                best_move = best_move_now
                self.depth_reached = depth
                # next iteration: this iteration's best first
                root = [(m, p, t) for _, m, p, t in sorted(scored, key=lambda e: -e[0])]
            except OutOfTime:
                break
        return best_move

    @staticmethod
    def next_context(context: List[str], tok: Optional[str]) -> List[str]:
        return (context + [tok])[-2:] if tok is not None else context

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, context: List[str], ply: int) -> float:
        self.nodes += 1
        if time.time() > self.deadline:
            raise OutOfTime
        if depth <= 0:
            return self.quiesce(board, alpha, beta, context, ply, QUIESCENCE_DEPTH)
        moves = self.ordered(board, context)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0.0
        best = -float('inf')
        for move, _, tok in moves[:BREADTH]:
            board.push(move)
            try:
                value = -self.negamax(board, depth - 1, -beta, -alpha, self.next_context(context, tok), ply + 1)
            finally:
                board.pop()
            if value > best:
                best = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best

    def quiesce(self, board: chess.Board, alpha: float, beta: float, context: List[str], ply: int, left: int) -> float:
        self.nodes += 1
        if time.time() > self.deadline:
            raise OutOfTime
        if board.is_checkmate():
            return -(MATE - ply)
        stand = self.evaluate(board)
        if stand >= beta or left == 0:
            return stand
        alpha = max(alpha, stand)
        for move, _, tok in self.ordered(board, context, captures_only=True):
            board.push(move)
            try:
                value = -self.quiesce(board, -beta, -alpha, self.next_context(context, tok), ply + 1, left - 1)
            finally:
                board.pop()
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break
        return alpha
