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

Results: evaluate_grid_weights, against the trainer's usual random
opponent. Each game, 6 replicates; each replicate: 160 games of learning
from 0 (20 generations of the cross-entropy method, its own seed), then 150
games without weights ('plain') and 150 with the learned mean ('weights'),
each arm from a copy of that replicate's learned databases. Win rates
(mean of the replicates), the weights arm minus the plain arm in each
replicate (points), and p for those 6 differences (paired t test;
Wilcoxon signed-rank, whose smallest possible two-sided p with 6 pairs is
0.031):

    game          plain  weights  differences per replicate        t p    wilcoxon p
    arimaa        70.9%   51.0%   -30.0 +4.0 -9.3 -20.7 -26.7 -36.7  0.022  0.062  worse
    breakthrough  96.0%   98.6%   +2.0 +5.3 +3.3 +4.0 -0.7 +1.3      0.032  0.062  better
    checkers      96.7%   95.7%   0.0 -6.0 +2.7 +4.0 -6.0 -0.7       0.588  0.562
    connect4      96.1%   96.7%   +3.3 +7.3 +2.0 -7.3 -2.7 +0.7      0.799  0.750
    dots_boxes    99.2%   97.1%   0.0 +2.0 -14.0 0.0 0.0 -0.7        0.420  0.750
    go            96.2%   98.6%   +8.0 0.0 +1.3 +2.0 +1.3 +1.3       0.101  0.062
    gomoku       100.0%  100.0%   0 in all six                       -      -
    hex           81.8%   84.9%   +12.7 -2.0 +10.0 -5.3 +2.0 +1.3    0.322  0.469
    loa           99.1%   99.1%   -0.7 -0.7 -1.3 0.0 -0.7 +3.3       1.000  0.625
    morris        93.4%   89.8%   -14.0 -2.0 -6.7 +0.7 +6.0 -6.0     0.248  0.250
    othello       80.9%   66.8%   -3.3 -16.0 -23.3 -8.0 -24.7 -9.3   0.010  0.031  worse
    pentago       91.1%   76.8%   -9.3 -14.0 -35.3 -16.0 -3.3 -8.0   0.026  0.031  worse

Worse in othello and pentago (all 6 replicates lower), and in arimaa (5 of
6 lower; t p 0.022, Wilcoxon p 0.062); better in breakthrough (5 of 6
higher; t p 0.032, Wilcoxon p 0.062); no significant difference in the
other eight. An earlier version of this table (one replicate of 300 games
per arm, two-proportion tests over the games) also had checkers, go and
pentago worse at p < 0.05; games of one arm are not independent (the
arm's learning carries from game to game), so those tests gave p values
too small. Against a random opponent the plain trainers already win
70-100% of games, so there is little room to gain. The weights also cost
time: 1.1x to 2.7x the seconds per game (from the earlier version).
Why the weights hurt in othello and pentago (arimaa not measured):
- Their size. Over each decision's candidates (20 games per replicate with
  the learned mean), the median spread of the weights' value was 8-67
  priority points in othello and 6-24 in pentago, of the rest of the score
  2-6 and 0.4-4; the weights changed the chosen move in 20-87% and 35-64%
  of decisions. That alone does not hurt: in breakthrough and go they
  changed 68-89% of moves, and the weights were better there. Only in
  checkers and dots_boxes was the rest of the score usually the larger.
- The learned weights were no better than random ones. 150 games per
  replicate from the same learned databases with weights drawn once from
  N(0, 3) per feature (learning's start spread): othello 61.2% (learned
  66.8%, plain 80.9%), pentago 77.1% (learned 76.8%, plain 91.1%); random
  minus learned: t p 0.38 and 0.97. In breakthrough and go the same random
  draws did worse than the learned weights (87.6% against 98.6%, and
  75.8% against 98.6%; t p 0.090 and 0.017): there learning found weights
  that help, in othello and pentago it did not in 160 games. Each
  feature's learned weight changes sign across the replicates in every
  game, and in pentago the win rate fell from the first 80 learning games
  to the last 80 in 5 of 6 replicates.
- The learned mean times 0.1 lost less: othello 75.1% (plain minus 5.8,
  t p 0.013; 8.3 above the full weights, t p 0.065), pentago 85.8% (plain
  minus 5.3, t p 0.054; 9.0 above the full weights, t p 0.086).
So in these two games, weights the learning did not improve beyond random
decided most moves and replaced the plain trainers' better choices.
Why learning found nothing there: 16 weight vectors drawn from N(0, 3) and
zero weights (plain), 100 games each from copies of replicate 1's learned
databases (standard error of a vector's win rate about 4.5 points):
    game          plain  random vectors' win rates         above plain
    othello        70%   29-71%, mean 54.7%                1 (71%)
    pentago        85%   49-89%, mean 76.9%                3 (86-89%)
    go             94%   36-100%, mean 75.9%               5 (97-100%)
    breakthrough   94%   61-100%, mean 87.6%               8 (95-100%)
- Little to find: in othello and pentago no drawn vector clearly beat plain
  play (those above it are within the standard error); in go and
  breakthrough several won 97-100%.
- A weaker signal: of the variance of one game's result, the vector
  accounted for 0.02 in othello and 0.06 in pentago, 0.27 in go and 0.15
  in breakthrough. One generation of the cross-entropy method (8 of the
  vectors, one recorded game each, the 3 best game scores kept; simulated)
  raised the true win rate of the kept vectors over the 8 by 2.1 points in
  othello and 3.2 in pentago (selecting by true win rate: 7.5 and 8.7),
  against 7.9 in go and 7.1 in breakthrough.
- The search stops regardless of signal: after the 20 generations every
  spread was at or near the floor (CE_MIN_STD 0.2; at most 0.6) in all four
  games and every replicate. With little signal the mean stops wherever
  the noise led it, which fits the signs of the learned weights changing
  across replicates.
One replicate's databases only; 16 vectors.
use_weights is on in the breakthrough trainer (tune_weights off: the
learned mean, LEARNED) and off in every other trainer. It was turned off
for a time after earlier measurements against a stronger opponent, and
turned on again after the re-measurement below.

Breakthrough from new databases (both arms learning everything else as
they play; the weights arm with the mean in LEARNED, not tuned), 8 runs of
400 games per arm; each run's win rate; p: Mann-Whitney between the runs
of the two arms:
    against the random opponent
        plain    93.2 95.5 96.0 97.2 97.2 97.2 98.2 98.2   mean 96.6
        weights  98.5 99.0 99.2 99.5 99.5 99.5 99.8 99.8   mean 99.3   p 0.0009
    against a stronger, fixed opponent (breakthrough_opponent.greedy: wins
    at once if it can, avoids moves that let the agent win at once,
    otherwise the best material and advancement; it beat random moves in
    39 of 40 games)
        plain     0.2  1.8  3.0  3.2  5.2  8.5 10.5 24.2   mean  7.1
        weights  16.0 16.2 17.2 17.5 17.5 19.8 20.0 22.0   mean 18.3   p 0.014
The weights (learned against random moves) were better against both
opponents. Against greedy the plain runs spread from 0.2% to 24%, the
weights runs 16-22%. An earlier measurement (4 blocks of 250 games of one
run per arm) had plain 24.5%, weights 20.6% against greedy, p 0.037 from
pooled games; the runs here do not reproduce it.
Weights learned against greedy instead (evaluate_grid_weights learn
--opponent greedy --start zero: 160 games, 20 generations from 0, spread
3), 8 replicates, each followed by 400 games per arm against greedy from
copies of its learned databases:
        plain    48.2  5.5  8.2  8.8  5.8  0.2 16.2  2.2   mean 11.9
        weights  19.2  0.2  8.5  9.0  0.8  0.0  7.5 20.2   mean  8.2
        differences -29.0 -5.2 +0.2 +0.2 -5.0 -0.2 -8.8 +18.0: t p 0.45,
        Wilcoxon p 0.38
No significant difference. An earlier single replicate had plain 64.8%,
weights 27.9% (p 2e-61 from pooled games); its learned weights were
pieces +1.99, largest_group -0.13, groups -2.43, longest_line -0.55, edge
+0.27, liberties +0.08, centre +2.78. In 7 of the 8 replicates here plain
play won 0.2-16%, so that replicate's 64.8% was not typical.

Why plain play won 64.8% in that replicate and 24.5% from new databases
(both against greedy; plain play, runs of 400 games; each run's win rate;
conditions B-D start from copies of that replicate's databases):
    A  new databases (8 runs)                 0.5 8.8 10.3 15.5 22.3 24.0 25.5 51.3
    B  copies of the learned databases (both files)    33.5 37.8 41.0 45.8
    C  the learned move priorities only                13.8 17.8 21.2 63.0
    D  the learned general patterns only               44.3 47.5 47.5 48.8
    E  three new learning runs (160 games, drawn
       weights), 400 plain games after each            0.2 11.0 35.8
    F1 D keeping only the threat=allows_loss patterns
       (8 runs)                               0.0 2.8 3.0 3.8 11.8 25.3 52.3 64.5
    F2 D without the threat=allows_loss patterns       37.5 44.3 49.0 54.3
    G1 the learned win-rate baseline only, no patterns
       (8 runs)                               5.5 7.5 8.3 15.0 47.0 49.3 49.5 56.8
    G2 F1 without the baseline (8 runs)       0.3 0.8 0.8 1.0 1.3 2.0 3.5 35.5
Tests compare runs (Mann-Whitney on the run win rates), not games: the
games of one run are not independent (each run's learning carries from
game to game), and tests over pooled games gave p values far too small
(an earlier version of this note had D against A p 7e-32 and F1 against A
p 4e-62 from pooled games).
Most of the gap is run-to-run variation: from new databases the runs end
anywhere from 0.5% to 51%, and the same learned databases gave 34-46%
rather than 64.8%. The general patterns learned (universal_move_patterns.db)
made play steady: D's four runs 44-49%, steady from the first 100 games
(41-47 wins); D against A p 0.050. The move priorities alone did not (C
against A p 0.68). F2 against D p 1.0. F1 is not worse than new databases
(p 0.72): four more runs gave 0, 25, 52 and 65%. G2 (the allows_loss
patterns with no baseline) was low in 7 of 8 runs (p 0.052 against A;
not explained, not significant at 0.05). G1 against A p 0.80. Learning
does not reliably reach good general patterns (E): a run that seldom wins
learns almost no differences between patterns (E3 won none of its 160
learning games; every pattern's win lift -0.01 to -0.03) and stays there.

Hex was first measured with offset rows (HexBoard's neighbours then
depended on row parity): plain 87.3%, weights 66.7% (one replicate). The
row above is from after the change to the standard rhombus.
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
