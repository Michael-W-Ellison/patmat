#!/usr/bin/env python3
"""
Grid Weights - the evaluation weights of chess (evaluation_weights), for every board game

As in chess (and as the Tetris agent does, object_agent), a position is
valued by a weighted sum of features, the weights learned from the game's
results by the cross-entropy method (evaluation_weights.WeightTuner: weights
drawn per game from a normal distribution, which moves toward the weights of
the best-scoring games). Here the features are game-independent: each is
worked out on the board as the game's adapter draws it (adapter.to_grid,
grid_vocabulary; the side's cells are adapter.grid_values), so the same
feature means the same thing in every game, and each game learns its own
weights (table grid_weights_<game> of its training database): which of
them matter, and which way, in that game.

Features of the position after a candidate move, for the side that moved
(its count minus the opponent's):
    pieces          cells holding its pieces
    largest_group   cells in its largest connected group (grid neighbours)
    groups          its connected groups
    longest_line    its longest straight run of cells
    edge            its pieces on the board's border
    liberties       empty cells next to its pieces
    centre          its pieces in the middle of the board (within a quarter
                    of the rows and of the columns of its centre)

Use (in each board game's trainer, use_weights): the agent's score for each
candidate move (its learned priority, the decisive part of the one-ply look
ahead and the general patterns' value) gets value(state, move, color) =
w . features(position after the move). A game's score for the cross-entropy
method is its result (1 won, 0.5 drawn, 0 lost) plus less than 0.2 for its
score margin (0.2 * m / (|m| + 1): a larger margin ranks a game higher among
games of the same result, as in chess the material balance does).
Weights start at 0 with spread START_STD (priority points per unit of a
feature); after each game the next game's weights are drawn.

Results: evaluate_grid_weights. Each game: 160 games of learning (20
generations of the cross-entropy method) against the trainer's usual random
opponent, then 300 games without weights ('plain') and 300 with the learned
mean ('weights'), each arm from a copy of the learned databases. Win rates
(p: two-sided test of two proportions):

    game          plain    weights    p
    arimaa        97.7%    57.0%      0.000   worse
    breakthrough  96.3%    99.0%      0.031   better
    checkers      97.7%    94.3%      0.037   worse (13 draws against 2)
    connect4      96.0%    93.7%      0.197
    dots_boxes   100.0%   100.0%      1.000
    go            97.0%    93.3%      0.036   worse
    gomoku       100.0%   100.0%      1.000
    hex           75.0%    80.3%      0.117   (rhombus layout; see below)
    loa           98.7%    99.3%      0.412
    morris        96.3%    93.0%      0.069
    othello       73.3%    72.7%      0.854
    pentago       91.7%    85.7%      0.020   worse

Better in one game (breakthrough), worse in four, no significant
difference in seven. Against a random opponent the plain trainers already win
75-100% of games in all but othello, so there is little room to gain. Why the
weights hurt in four games was not measured; one untested possibility is
that weights from only 20 generations, with a start spread of 3 priority
points, are large enough to outweigh better learned priorities. The
weights also cost time: 1.1x to 2.7x the seconds per game.
use_weights is on in the breakthrough trainer and off in the others.

Hex was first measured with offset rows (HexBoard's neighbours then
depended on row parity): plain 87.3%, weights 66.7%, p 0.000. The row
above is from a new run after the change to the standard rhombus.
"""

from typing import Dict, List, Optional, Tuple

from evaluation_weights import WeightTuner
from grid_vocabulary import Grid, adjacent_cells, components, lines

FEATURES = ('pieces', 'largest_group', 'groups', 'longest_line', 'edge', 'liberties', 'centre')
START_STD = 3.0
OWN, OPP, EMPTY, OTHER = 1, 2, 0, 3


def side_grid(adapter, state, color) -> Grid:
    """The board with color's cells OWN, the opponent's OPP, empty cells EMPTY and anything else OTHER"""
    grid = adapter.to_grid(state)
    own, opp = adapter.grid_values(color), adapter.grid_values(adapter.opponent(color))
    cells = [[OWN if v in own else OPP if v in opp else EMPTY if v == grid.empty else OTHER
              for v in row] for row in grid.cells.tolist()]
    return Grid(cells, grid.topology, empty=EMPTY, valid=grid.valid, adjacency=grid.adjacency,
                line_sets=grid.line_sets)


_shapes: Dict[Tuple, Tuple[frozenset, frozenset]] = {}


def _border_and_centre(grid: Grid) -> Tuple[frozenset, frozenset]:
    key = (grid.shape, grid.topology, grid.valid.tobytes())
    hit = _shapes.get(key)
    if hit is None:
        rows, cols = grid.shape
        centre = frozenset(p for p in grid.positions()
                           if abs(p[0] - (rows - 1) / 2) <= rows / 4 and abs(p[1] - (cols - 1) / 2) <= cols / 4)
        hit = _shapes[key] = (frozenset(grid.border_cells()), centre)
    return hit


def features(adapter, state, color) -> List[float]:
    """FEATURES of a position for color (its count minus the opponent's)"""
    g = side_grid(adapter, state, color)
    border, centre = _border_and_centre(g)
    cells = {OWN: [], OPP: []}
    for p in g.positions():
        v = g[p]
        if v in cells:
            cells[v].append(p)
    groups = {OWN: [], OPP: []}
    for region in components(g, values=(OWN, OPP)):
        groups[region.value].append(region.size)
    longest = {OWN: 0, OPP: 0}
    for line in lines(g, min_length=1):
        if line.value in longest and line.length > longest[line.value]:
            longest[line.value] = line.length

    def side(v):
        mine = cells[v]
        return [len(mine), max(groups[v], default=0), len(groups[v]), longest[v],
                sum(1 for p in mine if p in border), len(adjacent_cells(g, mine, value=EMPTY)),
                sum(1 for p in mine if p in centre)]
    return [a - b for a, b in zip(side(OWN), side(OPP))]


def game_score(result: str, margin: float) -> float:
    base = {'win': 1.0, 'draw': 0.5}.get(result, 0.0)
    return base + 0.2 * margin / (abs(margin) + 1.0)


class GridWeights:
    """A game's learned weights over FEATURES, and the value they give a candidate move"""

    def __init__(self, db_path: str, game: str, adapter, tune: bool = True, seed: Optional[int] = None):
        self.adapter = adapter
        self.tune = tune
        self.tuner = WeightTuner(db_path, seed=seed, table=f'grid_weights_{game}',
                                 start_mean=[0.0] * len(FEATURES), start_std=[START_STD] * len(FEATURES))
        self.current = self._next()

    def _next(self) -> List[float]:
        return self.tuner.draw() if self.tune else self.tuner.mean()

    def value(self, state, move, color) -> float:
        """w . features of the position after move (by color), in the trainer's priority points"""
        if hasattr(self.adapter, 'push') and hasattr(self.adapter, 'pop'):
            self.adapter.push(state, move)
            try:
                f = features(self.adapter, state, color)
            finally:
                self.adapter.pop(state)
        else:
            f = features(self.adapter, self.adapter.apply(state, move), color)
        return sum(w * x for w, x in zip(self.current, f))

    def end_game(self, result: str, margin: float) -> None:
        """A game's result and score margin (tuning: recorded with its weights); the next game's weights drawn"""
        if self.tune:
            self.tuner.record(self.current, game_score(result, margin))
        self.current = self._next()

    def close(self) -> None:
        self.tuner.close()
