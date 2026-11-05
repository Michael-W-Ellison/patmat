#!/usr/bin/env python3
"""
Tactical Patterns - what a move threatens after the opponent's answer, in every game

Forks, pins, skewers, discovered attacks and overloaded defenders are, in
chess, geometry: lines and squares. A game-independent adapter (see
general_move_patterns) has no lines or squares, only moves, positions and an
advantage measure. What these tactics have in common, seen through that
interface, is their effect: after the move and the opponent's best answer,
the mover still has more ways to gain than before (one answer cannot parry
two threats, a pinned or overloaded defender cannot cover what it covered),
or a bigger gain than any available now. This module measures that effect.

For each candidate move, in the position after it and after the opponent
reply that the one-ply look (analyze_candidates) rates best for the opponent,
the mover's own moves (follow-ups) are each played and scored by the value
they leave (evaluate(), so a won game is worth WIN_VALUE) minus the value in
the position now, before the candidate: what the mover nets over the move,
the answer and the follow-up. (Measured from the position after the answer
instead, escaping a check the answer gave would count as a gain in chess,
whose advantage measure values giving check.) Compared with the mover's moves
in the position now, each scored by its immediate gain:

- tac: wins   one of those follow-ups wins the game (the answer did not stop it)
       multi  at least 2 follow-ups gain, and a larger share of the follow-ups
              gain than of the moves now (a double threat that survived the answer)
       more   a larger share of the follow-ups gain than of the moves now
       same   the same share
       fewer  a smaller share
       none   the move ends the game, or the answer ends it, or no follow-up
- top: up     the best follow-up gains more than the best move available now gains
       same   as much
       down   less

A move that keeps the turn (extra_turn) is measured in the position right
after it. Shares, not counts, are compared because the moves examined are
sampled above max_moves.

GeneralPatternLearner (TACTICS) appends 'tac=<t>|top=<u>' to a move's general
pattern and learns the two halves as separate patterns: the four-part
pattern as before, and the tactical pattern, whose value is added.
"""

import random
from typing import Optional, Tuple

EPSILON = 1e-9


def gains(adapter, state, color, play, unplay, rng=random, max_moves=float('inf'),
          now: Optional[float] = None) -> Tuple[list, bool]:
    """
    Value (evaluate()) after each of color's moves in state minus now (default: the value of state itself),
    sampled above max_moves, and whether one of them wins
    """
    from general_move_patterns import WIN_VALUE, evaluate
    moves = adapter.legal_moves(state, color)
    if len(moves) > max_moves:
        moves = rng.sample(list(moves), int(max_moves))
    if now is None:
        now = evaluate(adapter, state, color)
    out = []
    for move in list(moves):
        after = play(state, move)
        out.append(evaluate(adapter, after, color) - now)
        unplay(after)
    return out, any(now + g >= WIN_VALUE / 2 for g in out)


def summary(values: list) -> Tuple[float, float]:
    """(share of moves that gain, best gain); (0, 0) without moves"""
    if not values:
        return 0.0, 0.0
    return sum(v > EPSILON for v in values) / len(values), max(values)


def classify(now: Tuple[float, float], later: Optional[Tuple[float, float, int, bool]]) -> str:
    """
    'tac=<t>|top=<u>' from now = summary() of the mover's moves now and later = (share, best, number gaining,
    wins) of its follow-ups after the answer, or None where nothing follows (see the module docstring)
    """
    if later is None:
        return 'tac=none|top=same'
    share_now, best_now = now
    share, best, count, wins = later
    if wins:
        tac = 'wins'
    elif share > share_now + EPSILON:
        tac = 'multi' if count >= 2 else 'more'
    elif share < share_now - EPSILON:
        tac = 'fewer'
    else:
        tac = 'same'
    if best > best_now + EPSILON:
        top = 'up'
    elif best < best_now - EPSILON:
        top = 'down'
    else:
        top = 'same'
    return f'tac={tac}|top={top}'
