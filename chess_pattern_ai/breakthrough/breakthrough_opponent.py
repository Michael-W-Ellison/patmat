#!/usr/bin/env python3
"""
A fixed (not learning) Breakthrough opponent stronger than random play, for measurements

greedy(game, color), in order:
1. a move that wins at once (reaches the goal row, or takes the last piece);
2. otherwise, among moves after which the opponent cannot win at once (all
   moves if there are none), the one with the best material and advancement
   (BreakthroughScorer's position score, color minus opponent), ties broken
   at random.
"""

import random

from breakthrough.breakthrough_adapter import BreakthroughAdapter
from general_move_patterns import winning_moves

_adapter = BreakthroughAdapter()


def greedy(game, color, rng=random):
    moves = _adapter.legal_moves(game, color)
    if not moves:
        return None
    wins = winning_moves(_adapter, game, color, moves)
    if wins:
        return rng.choice(wins)
    other = _adapter.opponent(color)
    scored = []
    for move in moves:
        after = _adapter.apply(game, move)
        replies = _adapter.legal_moves(after, other)
        safe = not winning_moves(_adapter, after, other, replies)
        scored.append((safe, _adapter.advantage(after, color), move))
    if any(s for s, _, _ in scored):
        scored = [x for x in scored if x[0]]
    best = max(v for _, v, _ in scored)
    return rng.choice([m for _, v, m in scored if v == best])
