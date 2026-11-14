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
feature), except in a game listed in LEARNED, whose new databases start
from the mean and spread learned there (as chess's new databases start
from its learned weights); after each game the next game's weights are
drawn.

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
use_weights is off in every trainer. It was on for a time in the
breakthrough trainer (tune_weights off: the learned mean, LEARNED), and
turned off after the measurements below against a stronger opponent.
Breakthrough with the weights on (the learned mean, not tuned), from new databases (both arms learning
everything else as they play), 1000 games each against the random
opponent: plain 95.1% (951 won, mean margin +1336), weights 99.3% (993 won,
+1481), p 1e-8; wins per 250 games plain 236, 242, 235, 238, weights 249,
248, 247, 249; 0.09 against 0.22 s per game.
The same against a stronger, fixed opponent (breakthrough_opponent.greedy:
wins at once if it can, avoids moves that let the agent win at once,
otherwise the best material and advancement; it beat random moves in 39 of
40 games), 1000 games each from new databases: plain 24.5% (245 won, mean
margin -712), weights 20.6% (206 won, -922), p 0.037; wins per 250 games
plain 63, 78, 47, 57, weights 45, 65, 46, 50; 0.17 against 0.39 s per game.
The weights were learned against random moves; against this opponent they
did worse than none.
Learned against that opponent instead (evaluate_grid_weights learn
--opponent greedy --start zero: 160 games, 20 generations from 0, spread
3; it won 4, 11, 13, 9 of each 40 learning games), then 1000 games each
against it from copies of the learned databases: plain 64.8% (648 won,
mean margin +362), weights 27.9% (279 won, -460), p 2e-61; wins per 250
games plain 123, 180, 178, 167, weights 44, 67, 82, 86. Learned weights:
pieces +1.99, largest_group -0.13, groups -2.43, longest_line -0.55, edge
+0.27, liberties +0.08, centre +2.78 (final spread 0.20-0.22).
Why plain play won 64.8% here and 24.5% from new databases (both against
greedy; plain play, 4 runs of 400 games per condition unless stated):
    A  new databases                                  26.9% (runs 22.2, 8.8, 25.5, 51.2)
    B  copies of the learned databases (both files)   39.5% (37.8, 45.8, 33.5, 41.0)
    C  the learned move priorities only               28.9% (13.8, 21.2, 63.0, 17.8)
    D  the learned general patterns only              47.0% (44.2, 47.5, 48.8, 47.5)
    E  three new learning runs (160 games, drawn
       weights), 400 plain games after each           35.8%, 11.0%, 0.2%
    F1 D keeping only the threat=allows_loss patterns  5.3%
    F2 D without the threat=allows_loss patterns      46.2%
Most of the gap is run-to-run variation: from new databases the runs end
anywhere from 9% to 51%, and the same learned databases gave 34-46%
rather than 64.8%. What the learning period passed on is in the general
patterns (universal_move_patterns.db): D against A p 7e-32, C against A
p 0.21; with them, play is steady from the first 100 games (41-47 wins).
Of the general patterns, it is those of moves that let the opponent win
nothing at once (F2 against D p 0.67; F1 alone is worse than none, p 4e-62,
not explained). The learned move priorities lowered it (B against D p
2e-5). Learning does not reliably reach such patterns (E): a run that
seldom wins learns almost no differences between patterns (E3 won none of
its 160 learning games; every pattern's win lift -0.01 to -0.03) and stays
there.

Hex was first measured with offset rows (HexBoard's neighbours then
depended on row parity): plain 87.3%, weights 66.7%, p 0.000. The row
above is from a new run after the change to the standard rhombus.
"""

from typing import Dict, List, Optional, Tuple

from evaluation_weights import WeightTuner
from grid_vocabulary import Grid, adjacent_cells, components, lines

FEATURES = ('pieces', 'largest_group', 'groups', 'longest_line', 'edge', 'liberties', 'centre')
START_STD = 3.0
# A new database's start (mean, spread) in these games: the weights learned by evaluate_grid_weights
# (160 games, 20 generations, against the trainer's usual opponent)
LEARNED = {
    'breakthrough': ([4.26, -1.29, 1.05, 5.38, 1.03, 1.37, 1.12], [0.33, 0.20, 0.20, 0.20, 0.20, 0.23, 0.36]),
}
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
        mean, std = LEARNED.get(game, ([0.0] * len(FEATURES), [START_STD] * len(FEATURES)))
        self.tuner = WeightTuner(db_path, seed=seed, table=f'grid_weights_{game}',
                                 start_mean=list(mean), start_std=list(std))
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
