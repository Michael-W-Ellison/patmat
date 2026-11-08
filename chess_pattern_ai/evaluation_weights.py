#!/usr/bin/env python3
"""
Evaluation Weights - the Tetris agent's way of valuing positions, for chess

The Tetris agent (object_agent, "Score and drift") values the places its
piece can come to rest by a weighted sum of features of the position (the
columns' heights, covered cells, roughness) and learns the weights from its
score by the cross-entropy method: each episode plays with weights drawn
from a normal distribution, and after every CE_POPULATION episodes the
distribution's mean and spread move (by CE_SMOOTH) to those of the CE_ELITE
episodes that scored best, the spread kept at least CE_MIN_STD.

Here the same, for the chess search's evaluation (PatternSearch.weights): a
position's value for the side to move gets w . features(board), each feature
the side to move's count minus the other side's:
    mobility      squares its knights, bishops, rooks and queens attack (/ 5)
    king_danger   squares next to its king the other side attacks
    king_shield   its own pawns on the three squares in front of its king
    loose         its knights, bishops, rooks and queens that are undefended
    pawn_weakness its doubled and isolated pawns
    passed        its passed pawns, each counted by how far it has advanced
                  (1 on its starting rank ... 6 one step from promotion;
                  / 3)
    centre        its attacks on d4, e4, d5 and e5, and its pieces there
    undeveloped   its knights and bishops on their starting squares
(weights in material units per unit of a feature; a pawn is 10). A game is
an episode; its score is its result for the agent (1 won, 0.5 drawn, 0 lost)
plus 0.0001 times the agent's mean material balance over the game (the
program's own measure: losing less material ranks a lost game higher, as the
Tetris agent ranks episodes by rows cleared, then by actions survived; a
balance never exceeds the material on the board, 0.0001 times it less than
the 0.5 between results).

The distribution is kept in the training database (table evaluation_weights).
For playing without learning, mean() gives the distribution's mean.

Results (evaluate_lookahead, arms weights and weights_depth; against
Stockfish skill 0, each agent move judged at depth 12): learned in 160 games
(20 generations) from the database of the episodic_memory results; the mean
weights then: mobility 2.5, king_danger -2.7, king_shield -1.2, loose -2.5,
pawn_weakness -4.9, passed 1.5, centre 2.3, undeveloped 0.8 (spreads 0.2 to
1.1). Wins while learning, per 40 games: 6, 7, 6, 5. Then 100 games per arm
from the database as learning left it, the four arms at once:
- plain:         15 won, 85 lost; blunders 2.8% (127 of 4,467 moves); mean
                 centipawn loss 61.5; depth reached 2.89; 0.24 s per move
- weights:       17 won, 83 lost; 2.9% (134 of 4,676); 58.2; depth 2.75;
                 0.29 s per move
- plain_depth:   14 won, 86 lost; 3.6% (151 of 4,207); 71.2; 0.30 s per move
- weights_depth: 23 won, 77 lost; 2.6% (123 of 4,814); 61.7; 0.42 s per move
At 0.5 s per move the weights changed nothing measurable. With the 3-ply
search always finished they won 23 against 14 (a difference of that size
arises by chance about one time in ten at 100 games each) and blundered
less, but the earlier plain_depth run (episodic_memory) blundered 2.5% with
14 wins, so this plain_depth run may have been a poor one; plain runs at 0.5
s have won between 10 and 18 of 100. Hence use_weights is off by default.
"""

import json
import random
import sqlite3
from typing import List, Optional

import chess

CE_POPULATION, CE_ELITE, CE_MIN_STD, CE_SMOOTH = 8, 3, 0.2, 0.7     # as the Tetris agent's (object_agent)
START_STD = 3.0         # the starting spread of each weight (material units per unit of the feature)

FEATURES = ('mobility', 'king_danger', 'king_shield', 'loose', 'pawn_weakness', 'passed', 'centre', 'undeveloped')
CENTRE = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])
PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
HOME = {chess.WHITE: chess.SquareSet([chess.B1, chess.C1, chess.F1, chess.G1]),
        chess.BLACK: chess.SquareSet([chess.B8, chess.C8, chess.F8, chess.G8])}


def _side(board: chess.Board, color: chess.Color) -> List[float]:
    """The features of one side (before the difference)"""
    other = not color
    own = board.occupied_co[color]
    pieces = own & (board.knights | board.bishops | board.rooks | board.queens)
    mobility = 0
    loose = 0
    for sq in chess.scan_forward(pieces):
        mobility += chess.popcount(board.attacks_mask(sq) & ~own)
        if not board.attackers_mask(color, sq):
            loose += 1
    king = board.king(color)
    danger = shield = 0
    if king is not None:
        for sq in chess.scan_forward(chess.BB_KING_ATTACKS[king]):
            if board.attackers_mask(other, sq):
                danger += 1
        step = 8 if color == chess.WHITE else -8
        front = king + step
        if 0 <= front < 64:
            f = chess.square_file(king)
            for df in (-1, 0, 1):
                if 0 <= f + df < 8:
                    sq = front + df
                    if board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == color:
                        shield += 1
    pawns = board.pawns & own
    their_pawns = board.pawns & board.occupied_co[other]
    files = [chess.popcount(pawns & chess.BB_FILES[f]) for f in range(8)]
    weak = sum(n - 1 for n in files if n > 1)
    weak += sum(n for f, n in enumerate(files)
                if n and not (f > 0 and files[f - 1]) and not (f < 7 and files[f + 1]))
    passed = 0
    for sq in chess.scan_forward(pawns):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
        block = 0
        for ff in (f - 1, f, f + 1):
            if 0 <= ff < 8:
                for rr in ahead:
                    block |= chess.BB_SQUARES[chess.square(ff, rr)]
        if not their_pawns & block:
            passed += (r if color == chess.WHITE else 7 - r)
    centre = sum(1 for sq in CENTRE if board.attackers_mask(color, sq)) + chess.popcount(own & int(CENTRE))
    undeveloped = chess.popcount((board.knights | board.bishops) & own & int(HOME[color]))
    return [mobility / 5.0, danger, shield, loose, weak, passed / 3.0, centre, undeveloped]


def features(board: chess.Board) -> List[float]:
    """The features (FEATURES) of a position: the side to move's minus the other side's"""
    mine, theirs = _side(board, board.turn), _side(board, not board.turn)
    return [a - b for a, b in zip(mine, theirs)]


def value(board: chess.Board, weights: List[float]) -> float:
    return sum(w * f for w, f in zip(weights, features(board)))


def episode_score(result: str, balances: List[float]) -> float:
    """A game's score for the cross-entropy method (module docstring)"""
    base = {'win': 1.0, 'draw': 0.5}.get(result, 0.0)
    return base + 0.0001 * (sum(balances) / len(balances) if balances else 0.0)


class WeightTuner:
    """A normal distribution over the weights, moved toward the best-scoring games' weights (cross-entropy)"""

    def __init__(self, db_path: str, seed: Optional[int] = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS evaluation_weights (id INTEGER PRIMARY KEY, state TEXT)')
        self.conn.commit()
        row = self.conn.execute('SELECT state FROM evaluation_weights WHERE id = 1').fetchone()
        state = json.loads(row[0]) if row else {}
        n = len(FEATURES)
        self.mu: List[float] = state.get('mean', [0.0] * n)
        self.std: List[float] = state.get('std', [START_STD] * n)
        self.done: List = state.get('done', [])         # (score, weights) of this generation's episodes
        self.generations: int = state.get('generations', 0)
        self.episodes: int = state.get('episodes', 0)
        self.rng = random.Random(seed)

    def mean(self) -> List[float]:
        return list(self.mu)

    def draw(self) -> List[float]:
        """Weights for an episode"""
        return [self.rng.gauss(m, s) for m, s in zip(self.mu, self.std)]

    def record(self, weights: List[float], score: float) -> None:
        """An episode's weights and score; after CE_POPULATION of them, the distribution moves to the elite"""
        self.done.append((score, list(weights)))
        self.episodes += 1
        if len(self.done) >= CE_POPULATION:
            elite = [w for _, w in sorted(self.done, key=lambda t: -t[0])[:CE_ELITE]]
            for i in range(len(self.mu)):
                col = [w[i] for w in elite]
                m = sum(col) / len(col)
                s = (sum((x - m) ** 2 for x in col) / len(col)) ** 0.5
                self.mu[i] = CE_SMOOTH * m + (1 - CE_SMOOTH) * self.mu[i]
                self.std[i] = max(CE_SMOOTH * s + (1 - CE_SMOOTH) * self.std[i], CE_MIN_STD)
            self.done = []
            self.generations += 1
        self.save()

    def save(self) -> None:
        state = {'mean': self.mu, 'std': self.std, 'done': self.done, 'generations': self.generations,
                 'episodes': self.episodes}
        self.conn.execute('INSERT OR REPLACE INTO evaluation_weights VALUES (1, ?)', (json.dumps(state),))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
