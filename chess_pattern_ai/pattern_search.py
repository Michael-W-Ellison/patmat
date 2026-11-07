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
expect the side to move to gain, plus, if mate patterns are given, the value
of how often each king's surroundings have been followed by its being mated
(mate_patterns); among the agent's own moves, the pattern value is
added (PATTERN_WEIGHT, limited to +-bonus_cap if one is given), so the patterns choose
between moves the search finds equal.

Selective search (selective=True), closer to how a person reads a position:
below the root, instead of the BREADTH best moves, a side considers its
forcing moves (captures, checks, promotions, and moves after which the moved
piece attacks two pieces or one worth more than it, as the move's sequence
pattern token describes it) and only its QUIET_MOVES best other moves by
pattern value. Captures and threats count only if the moving piece cannot
then be won at once or the capture takes at least its worth, and the same
rule limits the captures followed after the search depth. So the replies
that punish a move (a capture, a fork, an attack on a queen) are always
read, and quiet lines are read little.

A position that already occurred in the game is valued as a draw (0): going
back to it starts a repetition, and repeating ends the game drawn, so a side
that is ahead avoids it and a side that is behind may seek it.

Threat recognition (threats=True), what a person asks before moving: "what
does my opponent threaten?" A side's piece is threatened when the other side
attacks it and it is undefended (the whole piece is at stake) or attacked by
a cheaper piece (its worth minus that piece's is at stake), as the pieces'
movement shows it (threatened()). Then
- in the selective search, the side to move also considers, for each of its
  threatened pieces, that piece's best move (by pattern value) to a square
  where it cannot be won at once, so escaping a threat is read even when it
  is not among the QUIET_MOVES best quiet moves;
- where the search stops (the stand-pat value of the capture search), a side
  can answer only one threat with its move, so the value of a position
  subtracts what is at stake in its second-largest threat (e.g. a fork).

Refutation patterns (refutations set to a refutation_patterns.RefutationLearner,
with adapter set to the game's adapter): in the selective search, the replies
to each root move also include the REPLIES_EXTENDED replies whose
game-independent reply patterns most often refuted the agent's blunders
(learned from a judge), so a quiet reply that punishes a move is read even
when it is not among the QUIET_MOVES best quiet moves.

Learned look-ahead (lookahead set to a lookahead_patterns.LookaheadLearner): at
the end of a line, a position of a kind that in the agent's own games often
hid a tactic (a deeper look found a swing a glance did not) is read
lookahead_patterns.EXTENSION plies further before the capture search, once per
line. glance() and look() are the two views the lessons compare.

Episodic memory (episodes set to an episodic_memory.EpisodicMemory): a reply to
a root move that matches the punishment of a remembered episode is always
read, first, and episodic_memory.EXTENSION plies further; a root move that
matches one is read further too. Or (memory_mode 'root') only the
ROOT_CHECKED best root moves are checked, after the search (check()).

Killer moves and history scores (move_history set to a move_history.MoveHistory):
below the root, the killer moves of the ply (the last moves that caused a
cutoff there) are searched besides the usual moves, and the moves searched are
tried forcing moves first (in pattern order), then killers, then the others by
history score. Cutoffs and best moves are recorded as the search goes.
"""

import time
from typing import Dict, List, Optional, Tuple

import chess

from mate_patterns import MatePatternLearner
from position_patterns import PositionPatternLearner
from sequence_patterns import SequencePatternLearner, material, worth
from refutation_patterns import REPLIES_EXTENDED, reply_features

MATE = 100_000.0
BREADTH = 8             # moves searched at each position below the root (best pattern value first)
QUIESCENCE_DEPTH = 6    # captures followed beyond the search depth
MAX_DEPTH = 8
PATTERN_WEIGHT = 0.5    # root moves: search value + PATTERN_WEIGHT * pattern value (both material units)
QUIET_MOVES = 2         # selective search: non-forcing moves considered at each position below the root
ROOT_CHECKED = 3        # memory at the root: the search's best moves checked for remembered situations
ROOT_SECONDS = 0.5      # ... and the time for searching them again


def forcing(move: chess.Move, token: Optional[str], board: chess.Board) -> bool:
    """
    Is move a sound forcing move: a check, a promotion, a capture that does not give up more than it takes,
    or a threat (the moved piece then attacks two pieces, or one worth more than it) by a piece that cannot
    then be taken for less? From its sequence pattern token '<cat> <x cat|-><+> a<n><>> <safety>' (safety
    s/a: the moved piece cannot be won by the opponent at once); without a token, captures and checks
    """
    if move.promotion:
        return True
    if token is None:
        return board.is_capture(move) or board.gives_check(move)
    _, taken, attacks, safety = token.split(' ')
    if taken[-1] == '+':
        return True
    safe = safety in ('s', 'a')
    if taken[0] == 'x':
        return safe or sound_capture(board, move)
    return safe and (attacks[-1] == '>' or attacks[1] == '2')


def sound_capture(board: chess.Board, move: chess.Move) -> bool:
    """Does the capture take a piece worth at least as much as the capturing piece?"""
    target = board.piece_at(move.to_square)
    if target is None:          # en passant
        return True
    return worth(target.piece_type) >= worth(board.piece_type_at(move.from_square))


def threatened(board: chess.Board, color: chess.Color) -> List[Tuple[int, float]]:
    """
    (square, material at stake) for each of color's pieces (not the king) the other side can win at once,
    largest first: attacked and undefended, its worth; attacked by a cheaper piece, its worth minus the
    cheapest attacker's
    """
    out = []
    enemy = not color
    for sq in chess.scan_forward(board.occupied_co[color] & ~board.kings):
        attackers = board.attackers_mask(enemy, sq)
        if not attackers:
            continue
        value = worth(board.piece_type_at(sq))
        cheapest = min(worth(board.piece_type_at(a)) for a in chess.scan_forward(attackers))
        if not board.attackers_mask(color, sq):
            out.append((sq, value))
        elif cheapest < value:
            out.append((sq, value - cheapest))
    out.sort(key=lambda e: -e[1])
    return out


class OutOfTime(Exception):
    pass


class PatternSearch:

    def __init__(self, sequences: SequencePatternLearner, seconds: float = 0.5, use_patterns: bool = True,
                 positions: Optional[PositionPatternLearner] = None, selective: bool = False,
                 max_depth: Optional[int] = None, bonus_cap: Optional[float] = None,
                 mates: Optional[MatePatternLearner] = None, threats: bool = False):
        self.sequences = sequences
        self.threat_escapes = threats       # recognise threats: escapes from them are considered ...
        self.threat_cost = threats          # ... and a second threat costs its stake where the search stops
        self.mates = mates                  # None: kings' surroundings are not valued
        self.selective = selective          # below the root: forcing moves + QUIET_MOVES others (else BREADTH best)
        self.max_depth = max_depth          # None: MAX_DEPTH
        self.bonus_cap = bonus_cap          # None: the root pattern bonus is not limited
        self.positions = positions          # None: positions are valued by material only
        self.seconds = seconds
        self.use_patterns = use_patterns    # False: order by capture value only (for comparison)
        self.nodes = 0
        self.depth_reached = 0
        self.refutations = None             # a RefutationLearner: replies to root moves likeliest to refute them
        self.adapter = None                 # (the game's adapter, for the reply patterns)
        self.refutation_cache: Dict = {}
        self.move_history = None            # a MoveHistory: killers and history scores order the moves below the root
        self.lookahead = None               # a LookaheadLearner: positions to read further at a line's end
        self.episodes = None                # an EpisodicMemory: remembered punishments, read further
        self.recalls = 0
        # where the episodes are used: 'lines' (in every line, at its first reply and at the root) or 'root'
        # (only the ROOT_CHECKED best root moves, after the search; check())
        self.memory_mode = 'lines'
        self.root_flagged = 0               # root mode: searches whose best moves met a remembered situation
        self.extend_cache: Dict = {}
        self.extensions = 0

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

    def candidates(self, board: chess.Board, context: List[str], ply: int = 0) -> List[Tuple]:
        """The moves searched at a position below the root (ply: its distance from the root)"""
        moves = self.ordered(board, context)
        if not self.selective:
            return moves[:BREADTH]
        out, quiet = [], 0
        for entry in moves:
            if forcing(entry[0], entry[2], board):
                out.append(entry)
            elif quiet < QUIET_MOVES:
                out.append(entry)
                quiet += 1
        if self.threat_escapes:
            chosen = {e[0] for e in out}
            for sq, _ in threatened(board, board.turn):
                for entry in moves:         # best first: the threatened piece's best move to safety
                    move, _, tok = entry
                    if move.from_square == sq and (tok is None or tok.split(' ')[3] in ('s', 'a')):
                        if move not in chosen:
                            out.append(entry)
                            chosen.add(move)
                        break
        if self.refutations is not None and ply == 1 and self.refutations.has_evidence():
            key = board._transposition_key()
            extra = self.refutation_cache.get(key)
            if extra is None:
                features = reply_features(self.adapter, board, board.turn, [e[0] for e in moves])
                chosen = {e[0] for e in out}
                extra = [moves[i] for i in self.refutations.ranked(
                    features, REPLIES_EXTENDED, exclude={i for i, e in enumerate(moves) if e[0] in chosen})]
                self.refutation_cache[key] = extra
            out = out + extra
        if self.episodes is not None and self.memory_mode == 'lines' and ply == 1 and len(self.episodes):
            # remembered punishments: always read, and first
            recalled = [e for e in moves if self.episodes.recall(board, e[0])]
            if recalled:
                first = {e[0] for e in recalled}
                out = recalled + [e for e in out if e[0] not in first]
        history = self.move_history
        if history is not None:
            chosen = {e[0] for e in out}
            out = out + [e for e in moves if e[0] not in chosen and history.is_killer(ply, e[0])]
            side = board.turn

            def rank(i, entry):
                if forcing(entry[0], entry[2], board):
                    return 0, i, 0.0
                return 1, 0 if history.is_killer(ply, entry[0]) else 1, -history.score(side, entry[0]), i

            out = [e for _, e in sorted(((rank(i, e), e) for i, e in enumerate(out)), key=lambda r: r[0])]
        return out

    def bonus(self, pattern: float) -> float:
        b = PATTERN_WEIGHT * pattern
        if self.bonus_cap is not None:
            b = max(-self.bonus_cap, min(self.bonus_cap, b))
        return b

    def evaluate(self, board: chess.Board) -> float:
        """Value of a (non-final) position for the side to move"""
        value = material(board, board.turn)
        if self.positions is not None:
            value += self.positions.value(board)
        if self.mates is not None:
            value += self.mates.value(board)
        return value

    # --------------------------------------------------------------- search

    def choose(self, board: chess.Board, extra: Optional[Dict[chess.Move, float]] = None) -> Optional[chess.Move]:
        """The move for the side to move (extra: a further bonus for root moves, in material units)"""
        self.deadline = time.time() + self.seconds
        self.nodes = 0
        self.history = self.game_positions(board)
        self.refutation_cache = {}
        self.extend_cache = {}
        self.extensions = 0
        self.recalls = 0
        if self.move_history is not None:
            self.move_history.new_search()
        context = self.sequences.context(board) if self.use_patterns else []
        root = self.ordered(board, context)
        if not root:
            return None
        best_move = root[0][0]
        for depth in range(1, (self.max_depth if self.max_depth is not None else MAX_DEPTH) + 1):
            try:
                scored = []
                best_total = -float('inf')
                for move, pattern, tok in root:
                    bonus = self.bonus(pattern) + (extra.get(move, 0.0) if extra else 0.0)
                    # only whether the move's value beats this matters: the search below cuts off at it, and
                    # a move cut off returns it exactly, so the comparison is on the value, not on
                    # value + bonus (whose rounding could make a cut-off move look better than the best)
                    threshold = best_total - bonus
                    more = self.recalled(board, move)       # a trick remembered: read further
                    board.push(move)
                    try:
                        value = -self.negamax(board, depth - 1 + more, -float('inf'), -threshold,
                                              self.next_context(context, tok), 1, more > 0)
                    finally:
                        board.pop()
                    scored.append((value + bonus, move, pattern, tok))
                    if value > threshold:
                        best_total, best_move_now = value + bonus, move
                best_move = best_move_now
                self.depth_reached = depth
                # next iteration: this iteration's best first
                root = [(m, p, t) for _, m, p, t in sorted(scored, key=lambda e: -e[0])]
            except OutOfTime:
                break
        if self.memory_mode == 'root' and self.episodes is not None and len(self.episodes):
            best_move = self.check(board, root, context, extra, best_move)
        return best_move

    def check(self, board: chess.Board, root: List[Tuple], context: List[str],
              extra: Optional[Dict[chess.Move, float]], best_move: chess.Move) -> chess.Move:
        """
        Memory at the root only: after the search, if a reply to one of its ROOT_CHECKED best moves makes a
        situation remembered from an episode, those moves are searched again (full window, ROOT_SECONDS more):
        the ones met by a remembered situation EXTENSION plies deeper; the best of them is played
        """
        from episodic_memory import EXTENSION
        top = root[:ROOT_CHECKED]
        flagged = []
        for move, _, _ in top:
            board.push(move)
            try:
                flagged.append(any(self.episodes.recall(board, r) for r in board.legal_moves))
            finally:
                board.pop()
        if not any(flagged):
            return best_move
        self.root_flagged += 1
        self.recalls += sum(flagged)
        self.deadline = time.time() + ROOT_SECONDS
        depth = max(1, self.depth_reached)
        best, chosen = -float('inf'), best_move
        try:
            for (move, pattern, tok), hit in zip(top, flagged):
                bonus = self.bonus(pattern) + (extra.get(move, 0.0) if extra else 0.0)
                more = EXTENSION if hit else 0
                board.push(move)
                try:
                    value = -self.negamax(board, depth - 1 + more, -float('inf'), float('inf'),
                                          self.next_context(context, tok), 1, True)
                finally:
                    board.pop()
                if value + bonus > best:
                    best, chosen = value + bonus, move
        except OutOfTime:
            return best_move            # not all checked in time: the search's own choice
        return chosen

    def glance(self, board: chess.Board) -> float:
        """The value of a position for the side to move as a line's end sees it: the capture search alone"""
        self.deadline = float('inf')
        context = self.sequences.context(board) if self.use_patterns else []
        return self.quiesce(board, -float('inf'), float('inf'), context, 0, QUIESCENCE_DEPTH)

    def look(self, board: chess.Board, depth: int, seconds: float) -> Optional[float]:
        """The value of a position for the side to move searched depth plies (without look-ahead extensions,
        episodes or refutation patterns: what it shows does not depend on what was learned from it); None if it
        takes longer than seconds"""
        saved = self.lookahead, self.episodes, self.refutations
        self.lookahead = self.episodes = self.refutations = None
        self.deadline = time.time() + seconds
        self.history = self.game_positions(board) - {board._transposition_key()}
        self.refutation_cache = {}
        context = self.sequences.context(board) if self.use_patterns else []
        try:
            return self.negamax(board, depth, -float('inf'), float('inf'), context, 0)
        except OutOfTime:
            return None
        finally:
            self.lookahead, self.episodes, self.refutations = saved

    def recalled(self, board: chess.Board, move: chess.Move) -> int:
        """Plies to read a move further because it matches a remembered episode's punishment (episodic memory;
        counted in recalls whether or not EXTENSION is 0)"""
        if self.episodes is None or self.memory_mode != 'lines' or not self.episodes.recall(board, move):
            return 0
        from episodic_memory import EXTENSION
        self.recalls += 1
        return EXTENSION

    def read_further(self, board: chess.Board) -> bool:
        """At a line's end: whether the learned look-ahead patterns call for reading this position further"""
        key = board._transposition_key()
        hit = self.extend_cache.get(key)
        if hit is None:
            hit = self.extend_cache[key] = not board.is_game_over() and self.lookahead.extend(board)
        return hit

    @staticmethod
    def game_positions(board: chess.Board) -> set:
        """The positions (transposition keys) the game has been in, the current one included"""
        keys, b = set(), board.copy()
        keys.add(b._transposition_key())
        while b.move_stack:
            b.pop()
            keys.add(b._transposition_key())
        return keys

    @staticmethod
    def next_context(context: List[str], tok: Optional[str]) -> List[str]:
        return (context + [tok])[-2:] if tok is not None else context

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, context: List[str], ply: int,
                extended: bool = False) -> float:
        self.nodes += 1
        if time.time() > self.deadline:
            raise OutOfTime
        if board._transposition_key() in self.history:
            return 0.0                      # back to a position of the game: a repetition, valued as a draw
        if depth <= 0:
            if self.lookahead is not None and not extended and self.read_further(board):
                self.extensions += 1        # a position whose kind often hides a tactic: read further, once
                from lookahead_patterns import EXTENSION
                return self.negamax(board, EXTENSION, alpha, beta, context, ply, True)
            return self.quiesce(board, alpha, beta, context, ply, QUIESCENCE_DEPTH)
        moves = self.candidates(board, context, ply)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0.0
        best, best_move, cutoff, raised = -float('inf'), None, False, False
        for move, _, tok in moves:
            more = self.recalled(board, move) if ply == 1 and not extended else 0   # a punishment remembered
            board.push(move)
            try:
                value = -self.negamax(board, depth - 1 + more, -beta, -alpha, self.next_context(context, tok),
                                      ply + 1, extended or more > 0)
            finally:
                board.pop()
            if value > best:
                best, best_move = value, move
            if value > alpha:
                alpha, raised = value, True
            if alpha >= beta:
                cutoff = True
                break
        if self.move_history is not None and raised:
            self.move_history.record(ply, board.turn, best_move, depth, cutoff)
        return best

    def quiesce(self, board: chess.Board, alpha: float, beta: float, context: List[str], ply: int, left: int) -> float:
        self.nodes += 1
        if time.time() > self.deadline:
            raise OutOfTime
        if board.is_checkmate():
            return -(MATE - ply)
        stand = self.evaluate(board)
        if self.threat_cost:
            at_stake = threatened(board, board.turn)
            if len(at_stake) >= 2:
                stand -= at_stake[1][1]     # one threat can be answered, the second costs what is at stake
        if stand >= beta or left == 0:
            return stand
        alpha = max(alpha, stand)
        for move, _, tok in self.ordered(board, context, captures_only=True):
            if self.selective and not forcing(move, tok, board):
                continue
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
