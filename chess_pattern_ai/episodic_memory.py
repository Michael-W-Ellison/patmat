#!/usr/bin/env python3
"""
Episodic Memory - remembering the moves that were punished, and what punished them

Chess. The program's other memories are tallies: what kinds of move or position
were followed by what, counted over many games; once a game is counted, the
game itself is gone. This memory keeps episodes: particular moments of
particular games, what happened, and what it cost. It is fed by the program's
own hindsight; no outside judge.

Episodes (record_game, after each game): where a side's material fell by
SWING or more (a pawn is 10) within WINDOW plies of one of its moves in the
game as it was played, the program looks again, more deeply, at that move
(pattern_search.look, DEEP plies): the move was the cause if the position
after it is worth at least SWING less to the mover than the position before
it. The episode is then kept: the position after the move, the move, the
reply the opponent played (the punishment), the line that followed (LINE
plies), the material lost, and whose move it was (the agent's own blunders
and the opponent's).

Recall (recall): a move matches an episode when the same kind of piece moves
from the same square to the same square, seen from the side that moves (for
black the board is turned round), as the punishing reply of an episode did. A
remembered punishment is a cue: in the search (PatternSearch.episodes), a reply
to the agent's move that matches one is always read, and EXTENSION plies
deeper; so is a root move of the agent's that matches one (a trick it has seen
work).

Semantic memory (consolidate): each episode is also recorded, game-
independently, in the refutation patterns (refutation_patterns.RefutationLearner,
shared with the other games' trainers): every legal reply in the episode's
position described by its reply pattern (gain and mobility classes), and which
of them was the punishment. Over many episodes this gives what kinds of reply
punish a move, in general; the search reads the kinds that most often did
(PatternSearch.refutations).

Results (evaluate_lookahead; against Stockfish skill 0, each agent move judged
at depth 12, 0.5 s per move): learned in 70 games: 267 episodes (177 of the
agent's moves punished, mean 53 material units lost; 90 of the opponent's,
mean 37). Semantic memory: replies that gain most and leave the other side
fewest moves punished 10.6 times as often as an average reply; replies that
gain nothing, 0.3 times. Then 100 games per arm from the same database:
- plain:        12 won, 88 lost; blunders 3.2% (127 of 4,022 moves); mean
                centipawn loss 66.7; search depth reached 2.84
- memory:        1 won, 99 lost; 4.4% (150 of 3,389); 73.4; depth 1.34
                (37 remembered replies read further per move: the search no
                longer finished, and replies at once took pieces: 51 of its
                allowed blunders lost material from the first ply, against 13)
- memory_first: 13 won, 87 lost; 3.5% (150 of 4,303); 65.6; depth 2.62
                (remembered replies read first, not further; 186 matched per
                move, counted by recalls)
Neither helped. The cue is too broad: a piece's jump from one square to
another, from 592 episodes, matches most moves of most positions, so recall
does not single out the situation remembered; and what the semantic memory
learned (the replies that gain most punish) is what the search already reads
first (forcing moves). Hence use_memory is off by default (headless_trainer).
"""

import sqlite3
from typing import Dict, List, Optional, Tuple

import chess

from sequence_patterns import material

WINDOW = 6              # plies after a move within which a fall in material is looked into
SWING = 20              # a fall (and a deeper look's difference) of this much counts (a pawn is 10)
DEEP = 4                # plies of the deeper look
LOOK_SECONDS = 3.0      # its time limit (a look not finished is no evidence)
LINE = 6                # plies of the line kept with an episode
EXTENSION = 2           # plies a recalled move is read further


def key(board: chess.Board, move: chess.Move) -> Optional[Tuple[int, int, int]]:
    """A move as recall sees it: (piece type, from square, to square), seen from the side moving"""
    piece = board.piece_type_at(move.from_square)
    if piece is None:
        return None
    if board.turn == chess.BLACK:
        return piece, chess.square_mirror(move.from_square), chess.square_mirror(move.to_square)
    return piece, move.from_square, move.to_square


class EpisodicMemory:
    """Episodes kept in the training database; recall by key; consolidation into the refutation patterns"""

    def __init__(self, db_path: str, semantic=None, adapter=None):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fen TEXT,               -- the position after the punished move (the punisher to move)
            move TEXT,              -- the punished move
            reply TEXT,             -- the punishment: the reply played
            line TEXT,              -- the line that followed, from the reply (UCI, spaces)
            lost REAL,              -- material the mover lost within the window
            deeper REAL,            -- how much worse the deeper look found the position after the move
            blunderer TEXT,         -- 'agent' or 'opponent'
            game INTEGER,
            ply INTEGER)''')
        self.conn.commit()
        self.semantic = semantic        # a RefutationLearner (or None)
        self.adapter = adapter          # the game's adapter (for the reply patterns)
        self.keys: Dict[Tuple[int, int, int], int] = {}
        for fen, reply in self.conn.execute('SELECT fen, reply FROM episodes'):
            k = key(chess.Board(fen), chess.Move.from_uci(reply))
            if k is not None:
                self.keys[k] = self.keys.get(k, 0) + 1
        self.games = (self.conn.execute('SELECT MAX(game) FROM episodes').fetchone()[0] or 0)

    def __len__(self) -> int:
        return sum(self.keys.values())

    def recall(self, board: chess.Board, move: chess.Move) -> bool:
        """Does move (by the side to move on board) match the punishment of a remembered episode?"""
        return bool(self.keys) and key(board, move) in self.keys

    # -------------------------------------------------------------- episodes

    def record_game(self, moves: List[chess.Move], agent: chess.Color, search) -> List[Dict]:
        """After a game (its moves from the starting position): the episodes found in it, kept and consolidated"""
        self.games += 1
        boards = [chess.Board()]
        for m in moves:
            b = boards[-1].copy(stack=False)
            b.push(m)
            boards.append(b)
        found = []
        looks: Dict[int, Optional[float]] = {}

        def look(i):            # the deeper look at the position before move i, for the side to move there
            if i not in looks:
                b = chess.Board()
                for m in moves[:i]:
                    b.push(m)
                looks[i] = None if b.is_game_over() else search.look(b, DEEP, LOOK_SECONDS)
            return looks[i]
        i = 0
        while i < len(moves):
            mover = boards[i].turn
            now = material(boards[i], mover)
            later = [material(boards[j], mover) for j in range(i + 1, min(len(boards), i + 1 + WINDOW))]
            if later and now - min(later) >= SWING and i + 1 < len(moves):
                before, after = look(i), look(i + 1)
                if before is not None and after is not None and before - (-after) >= SWING:
                    found.append(self._keep(boards, moves, i, now - min(later), before + after,
                                            'agent' if mover == agent else 'opponent'))
                    i += 2              # the reply belongs to this episode
                    continue
            i += 1
        return found

    def _keep(self, boards, moves, i, lost, deeper, blunderer) -> Dict:
        after = boards[i + 1]
        reply = moves[i + 1]
        line = ' '.join(m.uci() for m in moves[i + 1:i + 1 + LINE])
        self.conn.execute('INSERT INTO episodes (fen, move, reply, line, lost, deeper, blunderer, game, ply) '
                          'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                          (after.fen(), moves[i].uci(), reply.uci(), line, lost, deeper, blunderer, self.games, i))
        self.conn.commit()
        k = key(after, reply)
        if k is not None:
            self.keys[k] = self.keys.get(k, 0) + 1
        self.consolidate(after, reply)
        return {'ply': i, 'move': moves[i].uci(), 'reply': reply.uci(), 'lost': lost, 'blunderer': blunderer}

    def consolidate(self, after: chess.Board, reply: chess.Move) -> None:
        """Record the episode in the semantic memory: every legal reply's pattern, and which was the punishment"""
        if self.semantic is None or self.adapter is None:
            return
        from refutation_patterns import reply_features
        replies = list(after.legal_moves)
        if reply not in replies:
            return
        board = after.copy()
        self.semantic.record(reply_features(self.adapter, board, board.turn, replies), replies.index(reply))

    def close(self) -> None:
        self.conn.close()
