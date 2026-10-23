#!/usr/bin/env python3
"""
Relation Learning - which grid relations go with winning, learned from games

Plan step 2. Positions of every game are seen as grids (adapter.to_grid)
and described with the shared grid vocabulary. For one side, each relation
is that side's measure minus the opponent's:

- pieces:          cells held
- largest_region:  size of the largest connected group of own cells
- regions:         number of separate groups of own cells
- longest_line:    longest straight run of own cells
- freedom:         empty cells next to own cells
- enclosed_empty:  empty cells that own cells wall off from the border
- capture_threat:  most opponent cells one placement could bracket
- span:            largest extent (rows or columns) of one own group
- sides_touched:   most grid sides touched by one own group

A move is described by the direction it changes each relation for the
side that made it ('+', '0', '-'). Games are observed with random moves
for both sides, so no strategy is built in; the only signal is which side
eventually won. For each game and each (relation, change):

    lift = P(mover wins | move made that change) - P(mover wins)

over moves of decisive games, with a 95% interval from resampling whole
games (moves of one game are not independent). Nothing game-specific is
used beyond the adapter interface (to_grid, grid_values, legal moves).

Run:  python3 relation_learning.py --games all --n 200 --db relations.db
"""

import argparse
import importlib
import random
import sqlite3
import time
from typing import Dict, List, Optional, Sequence, Tuple

from grid_vocabulary import (Grid, adjacent_cells, bracketed, cells_of, components,
                             enclosed, longest_line, touches)

RELATIONS = ('pieces', 'largest_region', 'regions', 'longest_line', 'freedom',
             'enclosed_empty', 'capture_threat', 'span', 'sides_touched')
CHANGES = ('+', '0', '-')
EPSILON = 1e-9

# Constructors only: everything else goes through the adapter interface
GAMES = {
    'chess': ('chess', 'Board', 'chess_adapter', 'ChessAdapter'),
    'checkers': ('checkers.checkers_game', 'CheckersGame', 'checkers.checkers_adapter', 'CheckersAdapter'),
    'othello': ('othello.othello_game', 'OthelloGame', 'othello.othello_adapter', 'OthelloAdapter'),
    'connect4': ('connect4.connect4_game', 'Connect4Game', 'connect4.connect4_adapter', 'Connect4Adapter'),
    'gomoku': ('gomoku.gomoku_game', 'GomokuGame', 'gomoku.gomoku_adapter', 'GomokuAdapter'),
    'hex': ('hex.hex_game', 'HexGame', 'hex.hex_adapter', 'HexAdapter'),
    'dots_boxes': ('dots_boxes.dots_boxes_game', 'DotsBoxesGame', 'dots_boxes.dots_boxes_adapter', 'DotsBoxesAdapter'),
    'breakthrough': ('breakthrough.breakthrough_game', 'BreakthroughGame', 'breakthrough.breakthrough_adapter', 'BreakthroughAdapter'),
    'go': ('go.go_game', 'GoGame', 'go.go_adapter', 'GoAdapter'),
    'morris': ('morris.morris_game', 'MorrisGame', 'morris.morris_adapter', 'MorrisAdapter'),
    'pentago': ('pentago.pentago_game', 'PentagoGame', 'pentago.pentago_adapter', 'PentagoAdapter'),
    'loa': ('loa.loa_game', 'LOAGame', 'loa.loa_adapter', 'LOAAdapter'),
    'arimaa': ('arimaa.arimaa_game', 'ArimaaGame', 'arimaa.arimaa_adapter', 'ArimaaAdapter'),
}


def make_game(name: str):
    """(adapter, initial state, [first colour, second colour])"""
    state_mod, state_cls, adapter_mod, adapter_cls = GAMES[name]
    state = getattr(importlib.import_module(state_mod), state_cls)()
    adapter = getattr(importlib.import_module(adapter_mod), adapter_cls)()
    first = adapter.to_move(state)
    return adapter, state, [first, adapter.opponent(first)]


# ------------------------------------------------------------- relations

def _side_grid(grid: Grid, own: set, opp: set) -> Grid:
    """Same board with own cells 1, opponent cells 2, everything else 0"""
    cells = [[1 if v in own else 2 if v in opp else 0 for v in row] for row in grid.cells.tolist()]
    return Grid(cells, grid.topology, empty=0, valid=grid.valid,
                adjacency=grid.adjacency, line_sets=grid.line_sets)


def _side_measures(side: Grid, me: int, them: int, empty_cells: set) -> Dict[str, float]:
    mine = cells_of(side, [me])
    if not mine:
        return dict.fromkeys(RELATIONS, 0.0)
    groups = components(side, values=[me])
    rows, cols = side.shape
    out = {
        'pieces': float(len(mine)),
        'largest_region': float(max(g.size for g in groups)),
        'regions': float(len(groups)),
        'longest_line': float(longest_line(side, me)),
        'freedom': float(len(adjacent_cells(side, mine) & empty_cells)),
        'span': float(max(max(g.bbox[2] - g.bbox[0], g.bbox[3] - g.bbox[1]) + 1 for g in groups)),
        'sides_touched': float(max(sum(touches(side, g.cells, s) for s in ('top', 'bottom', 'left', 'right'))
                                   for g in groups)),
    }
    if side.topology == 'graph':
        # No border or straight directions on a graph board
        out['enclosed_empty'] = 0.0
        out['capture_threat'] = 0.0
    else:
        out['enclosed_empty'] = float(sum(len(r.cells & empty_cells) for r in enclosed(side, walls=[me])))
        out['capture_threat'] = float(max((len(bracketed(side, e, own=me, other=them)) for e in empty_cells),
                                          default=0))
    return out


def relation_profile(grid: Grid, own_values: Sequence[int], opp_values: Sequence[int]) -> Dict[str, float]:
    """Each relation as own measure minus opponent measure"""
    side = _side_grid(grid, set(own_values), set(opp_values))
    empty_cells = set(cells_of(side, [0]))
    mine = _side_measures(side, 1, 2, empty_cells)
    theirs = _side_measures(side, 2, 1, empty_cells)
    return {r: mine[r] - theirs[r] for r in RELATIONS}


def move_changes(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, str]:
    """Direction each relation changed ('+', '0', '-')"""
    out = {}
    for r in RELATIONS:
        d = after[r] - before[r]
        out[r] = '+' if d > EPSILON else '-' if d < -EPSILON else '0'
    return out


# ------------------------------------------------------------ observation

def observe_random_games(adapter, initial_state, colors, n_games: int, max_plies: int,
                         rng: random.Random) -> List[Tuple[Optional[object], List[Tuple[int, Dict[str, str]]]]]:
    """
    Play n_games with random legal moves for both sides.

    Returns:
        per game (winner colour or None, [(mover index 0/1, changes), ...])
    """
    values = [adapter.grid_values(colors[0]), adapter.grid_values(colors[1])]
    games = []
    for _ in range(n_games):
        state = initial_state
        profile = relation_profile(adapter.to_grid(state), values[0], values[1])
        moves = []
        winner = None
        for _ in range(max_plies):
            if adapter.is_over(state):
                winner = adapter.winner(state)
                break
            mover = adapter.to_move(state)
            legal = adapter.legal_moves(state, mover)
            if not legal:
                break
            state = adapter.apply(state, rng.choice(list(legal)))
            after = relation_profile(adapter.to_grid(state), values[0], values[1])
            idx = 0 if mover == colors[0] else 1
            sign = 1 if idx == 0 else -1     # relations are antisymmetric
            moves.append((idx, move_changes({r: sign * v for r, v in profile.items()},
                                            {r: sign * v for r, v in after.items()})))
            profile = after
        else:
            if adapter.is_over(state):
                winner = adapter.winner(state)
        games.append((None if winner is None else (0 if winner == colors[0] else 1), moves))
    return games


def move_lifts(games, rng: random.Random, bootstrap: int = 400) -> Dict[Tuple[str, str], Dict]:
    """Lift of every (relation, change) with a 95% interval from resampling games"""
    decisive = [(w, mv) for w, mv in games if w is not None and mv]
    # Per game: total moves/wins and per (relation, change) moves/wins
    per_game = []
    for winner, moves in decisive:
        totals = [0, 0]
        cells = {}
        for idx, changes in moves:
            won = 1 if idx == winner else 0
            totals[0] += 1
            totals[1] += won
            for r, c in changes.items():
                acc = cells.setdefault((r, c), [0, 0])
                acc[0] += 1
                acc[1] += won
        per_game.append((totals, cells))

    def lifts(sample):
        n = sum(t[0] for t, _ in sample)
        w = sum(t[1] for t, _ in sample)
        base = w / n if n else 0.0
        out = {}
        for r in RELATIONS:
            for c in CHANGES:
                m = sum(cells.get((r, c), (0, 0))[0] for _, cells in sample)
                k = sum(cells.get((r, c), (0, 0))[1] for _, cells in sample)
                out[(r, c)] = (k / m - base) if m else None
        return out, n

    point, total_moves = lifts(per_game)
    draws = {key: [] for key in point}
    for _ in range(bootstrap if per_game else 0):
        sample = [per_game[rng.randrange(len(per_game))] for _ in per_game]
        res, _ = lifts(sample)
        for key, v in res.items():
            if v is not None:
                draws[key].append(v)

    result = {}
    for (r, c), lift in point.items():
        m = sum(cells.get((r, c), (0, 0))[0] for _, cells in per_game)
        d = sorted(draws[(r, c)])
        lo = d[int(0.025 * (len(d) - 1))] if d else None
        hi = d[int(0.975 * (len(d) - 1))] if d else None
        result[(r, c)] = {'moves': m, 'share': m / total_moves if total_moves else 0.0,
                          'lift': lift, 'ci_low': lo, 'ci_high': hi}
    return result


# ---------------------------------------------------------------- storage

class RelationKnowledge:
    """Stores per-game relation lifts and ranks relation changes across games"""

    def __init__(self, db_path: str = 'relations.db'):
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS relation_lift (
                game TEXT NOT NULL,
                relation TEXT NOT NULL,
                change TEXT NOT NULL,          -- '+', '0', '-'
                moves INTEGER,                 -- moves of decisive games with this change
                share REAL,                    -- fraction of all such moves
                lift REAL,                     -- P(mover wins | change) - P(mover wins)
                ci_low REAL,
                ci_high REAL,
                games INTEGER,                 -- decisive games observed
                PRIMARY KEY (game, relation, change)
            )
        ''')
        self.conn.commit()

    def save(self, game: str, lifts: Dict, decisive_games: int):
        self.conn.executemany('''
            INSERT OR REPLACE INTO relation_lift VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(game, r, c, v['moves'], v['share'], v['lift'], v['ci_low'], v['ci_high'], decisive_games)
              for (r, c), v in lifts.items()])
        self.conn.commit()

    def ranking(self) -> List[Dict]:
        """
        Per (relation, change): in how many games it goes significantly with
        winning (interval above 0) or losing (below 0), and the mean lift
        """
        rows = self.conn.execute(
            "SELECT relation, change, lift, ci_low, ci_high FROM relation_lift WHERE lift IS NOT NULL").fetchall()
        table = {}
        for r, c, lift, lo, hi in rows:
            e = table.setdefault((r, c), {'relation': r, 'change': c, 'games': 0, 'helps': 0,
                                          'hurts': 0, 'lift_sum': 0.0})
            e['games'] += 1
            e['lift_sum'] += lift
            e['helps'] += lo is not None and lo > 0
            e['hurts'] += hi is not None and hi < 0
        out = []
        for e in table.values():
            e['mean_lift'] = e.pop('lift_sum') / e['games']
            out.append(e)
        out.sort(key=lambda e: (e['helps'] - e['hurts'], e['mean_lift']), reverse=True)
        return out

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Learn which grid relations go with winning')
    parser.add_argument('--games', default='all', help="comma-separated game names, or 'all'")
    parser.add_argument('--n', type=int, default=200, help='random games per game type')
    parser.add_argument('--max-plies', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--db', default='relations.db')
    args = parser.parse_args()

    names = list(GAMES) if args.games == 'all' else args.games.split(',')
    knowledge = RelationKnowledge(args.db)
    for name in names:
        t0 = time.time()
        rng = random.Random(f'{args.seed}-{name}')
        adapter, state, colors = make_game(name)
        games = observe_random_games(adapter, state, colors, args.n, args.max_plies, rng)
        decisive = sum(1 for w, mv in games if w is not None and mv)
        knowledge.save(name, move_lifts(games, rng), decisive)
        print(f"{name}: {len(games)} games, {decisive} decisive, {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'relation':<16}{'change':>7}{'games':>7}{'helps':>7}{'hurts':>7}{'mean lift':>11}")
    for e in knowledge.ranking():
        print(f"{e['relation']:<16}{e['change']:>7}{e['games']:>7}{e['helps']:>7}{e['hurts']:>7}{e['mean_lift']:>+11.3f}")
    knowledge.close()


if __name__ == '__main__':
    main()
