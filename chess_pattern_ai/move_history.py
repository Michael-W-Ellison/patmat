#!/usr/bin/env python3
"""
Move History - killer moves and history scores, for every game

Two move-ordering patterns of alpha-beta search, learned during the search
itself and kept between searches:

- killer moves: at each ply (distance from the root of a search), the last
  KILLERS moves that caused a cutoff there. A move that refuted one line at a
  ply often refutes its sibling lines too, since those differ from it by one
  earlier move.
- history scores: for each side and move, the sum of depth * depth over the
  times that move caused a cutoff or was the best move of a position, in any
  position. A move that has been good in many positions is likely good in the
  next. Scores are multiplied by DECAY at the start of each search
  (new_search), so that recent searches count most.

A move is identified by itself where it is hashable (chess.Move, tuples),
else by its repr(): the trainers' move objects are dataclasses whose repr
lists their contents (piece, squares), so equal moves in different positions
share one entry. Positions are not part of the key: that is what lets a
move's record carry over to the next position.

Used by (each off by default):
- move_judges.SearchJudge (move_history set): moves tried killers first,
  then by history score, so the same search cuts off more and the node limit
  stops it later (the values it finds at a completed depth are unchanged);
- general_move_patterns.analyze_candidates (move_history given): where the
  opponent's replies are sampled (above max_replies), the killer replies
  (the replies that were the opponent's best against earlier candidates in
  the same position) and the best-scored replies are always examined, and
  the rest sampled;
- pattern_search.PatternSearch (move_history set): below the root, the
  killers of the ply are searched besides the usual moves, and non-forcing
  moves are tried killers first, then by history score.
"""

from typing import Dict, Hashable, List, Sequence

KILLERS = 2             # killer moves kept per ply
DECAY = 0.5             # history scores are multiplied by this at the start of each search
FORGOTTEN = 0.01        # history scores below this are dropped (a move recorded once at depth 1: after 7 searches)
REPLIES_KEPT = 1 / 3    # analyze_candidates: share of the sampled replies taken by killers and history first


def move_key(move) -> Hashable:
    try:
        hash(move)
        return move
    except TypeError:
        return repr(move)


class MoveHistory:
    """Killer moves per ply and history scores per (side, move); see the module docstring"""

    def __init__(self):
        self.killers: Dict[int, List[Hashable]] = {}
        self.scores: Dict[tuple, float] = {}

    def new_search(self):
        """Start of a search: killers are forgotten (plies count from a new root), history scores decay"""
        self.killers = {}
        self.scores = {k: v * DECAY for k, v in self.scores.items() if v * DECAY >= FORGOTTEN}

    def record(self, ply: int, side, move, depth: int, cutoff: bool = True):
        """move was side's best at a position ply plies from the root, searched depth plies further"""
        key = move_key(move)
        if cutoff:
            row = self.killers.setdefault(ply, [])
            if key in row:
                row.remove(key)
            row.insert(0, key)
            del row[KILLERS:]
        self.scores[(side, key)] = self.scores.get((side, key), 0.0) + max(1, depth) ** 2

    def is_killer(self, ply: int, move) -> bool:
        return move_key(move) in self.killers.get(ply, ())

    def score(self, side, move) -> float:
        return self.scores.get((side, move_key(move)), 0.0)

    def order(self, moves: Sequence, ply: int, side) -> list:
        """moves, killers of the ply first, then by history score (highest first); ties keep their order"""
        killers = self.killers.get(ply, ())
        keyed = [(m, move_key(m)) for m in moves]
        return [m for m, k in sorted(keyed, key=lambda e: (e[1] not in killers,
                                                           -self.scores.get((side, e[1]), 0.0)))]
