#!/usr/bin/env python3
"""
Move judges - how much worse than the best move a move was, by a deeper look

A judge gives a teacher signal for every move, instead of one game result
per game: a move's loss is the value of the best move minus the value of the
move played, both as the judge sees them (0 for a move the judge rates best,
never negative). Learners use it with game results (see
general_move_patterns.GeneralPatternLearner.note_move).

Two judges share one interface, move_loss(state, color, move) -> float, or
None where the judge sees the position as already lost whatever is played:
there every move's loss would be 0, rating a move that loses at once as
highly as any other.

- SearchJudge works for every game with an adapter (see
  general_move_patterns): an alpha-beta search over the game's own
  evaluate() (its winning condition, else its advantage measure), deeper
  than the one-ply analysis the trainers use to choose moves. It knows
  nothing the game's adapter does not, so what it teaches is what that
  deeper look sees. Losses are in the adapter's advantage units, except
  that a game won or lost is worth WIN_VALUE.

- StockfishJudge is for chess only: Stockfish at a fixed depth. Losses are
  in centipawns, with mate scores and evaluations limited to +-MATE_CP.
"""

from typing import Dict, Optional

from general_move_patterns import WIN_VALUE, evaluate

INF = float('inf')


class SearchJudge:
    """
    Alpha-beta over an adapter (general_move_patterns), to depth plies below
    the position judged; a search stopped by the node limit falls back to the
    deepest depth it completed (the first ply is always completed)
    """

    def __init__(self, adapter, depth: int = 3, max_nodes: int = 20_000):
        if getattr(adapter, 'hidden_information', False):
            raise ValueError('looking ahead with apply() would reveal hidden information')
        self.adapter = adapter
        self.depth = depth
        self.max_nodes = max_nodes
        self.in_place = hasattr(adapter, 'push') and hasattr(adapter, 'pop')
        self.nodes = 0

    def _play(self, state, move):
        if self.in_place:
            self.adapter.push(state, move)
            return state
        return self.adapter.apply(state, move)

    def _unplay(self, state):
        if self.in_place:
            self.adapter.pop(state)

    def _search(self, state, color, depth: int, alpha: float, beta: float, limited: bool) -> float:
        """Value of state for color, depth plies deep (the side to move maximizes its own value)"""
        adapter = self.adapter
        self.nodes += 1
        if depth == 0 or adapter.is_over(state):
            return evaluate(adapter, state, color)
        if limited and self.nodes > self.max_nodes:
            raise _OutOfNodes
        mover = adapter.to_move(state)
        moves = adapter.legal_moves(state, mover)
        if not moves:
            return evaluate(adapter, state, color)
        maximize = mover == color
        best = -INF if maximize else INF
        for move in moves:
            after = self._play(state, move)
            try:
                v = self._search(after, color, depth - 1, alpha, beta, limited)
            finally:
                self._unplay(after)
            if maximize:
                best = max(best, v)
                alpha = max(alpha, v)
            else:
                best = min(best, v)
                beta = min(beta, v)
            if alpha >= beta:
                break
        return best

    def move_values(self, state, color, depth: int, limited: bool = True) -> list:
        """[(move, value for color after it)] for every legal move of color, searched depth - 1 plies further"""
        out = []
        for move in self.adapter.legal_moves(state, color):
            after = self._play(state, move)
            try:
                out.append((move, self._search(after, color, depth - 1, -INF, INF, limited)))
            finally:
                self._unplay(after)
        return out

    def move_loss(self, state, color, move) -> Optional[float]:
        self.nodes = 0
        values = self.move_values(state, color, 1, limited=False)
        completed = 1
        for depth in range(2, self.depth + 1):
            try:
                values = self.move_values(state, color, depth)
            except _OutOfNodes:
                break
            completed = depth
        best = max(v for _, v in values)
        if best <= -WIN_VALUE / 2:
            return None
        # the move played, searched as deep as the others (searched again: games' move objects need not compare
        # equal to those legal_moves() makes)
        after = self._play(state, move)
        try:
            chosen = self._search(after, color, completed - 1, -INF, INF, False)
        finally:
            self._unplay(after)
        return max(0.0, best - chosen)


class _OutOfNodes(Exception):
    pass


MATE_CP = 1000      # StockfishJudge: mate scores and evaluations are limited to +-MATE_CP centipawns


class StockfishJudge:
    """Chess: Stockfish at a fixed depth; state is a chess.Board, losses in centipawns"""

    def __init__(self, path: str = '/usr/games/stockfish', depth: int = 12):
        import chess.engine
        self.limit = chess.engine.Limit(depth=depth)
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.cache: Dict[str, int] = {}

    def evaluate(self, board) -> int:
        """Centipawns for White (+-MATE_CP at most); finished games by their result"""
        import chess
        key = board.fen()
        v = self.cache.get(key)
        if v is None:
            outcome = board.outcome()
            if outcome is not None:
                v = 0 if outcome.winner is None else (MATE_CP if outcome.winner == chess.WHITE else -MATE_CP)
            else:
                # a forced mate scores beyond the limit, so every forced mate is exactly +-MATE_CP (with
                # mate_score=MATE_CP, being mated in 3 would score -997 and the position would not count as lost)
                score = self.engine.analyse(board, self.limit)['score'].white().score(mate_score=100 * MATE_CP)
                v = max(-MATE_CP, min(MATE_CP, score))
            if len(self.cache) > 100_000:
                self.cache.clear()
            self.cache[key] = v
        return v

    def value(self, board, color) -> int:
        import chess
        v = self.evaluate(board)
        return v if color == chess.WHITE else -v

    def move_loss(self, board, color, move) -> Optional[float]:
        before = self.value(board, color)
        if before <= -MATE_CP:
            return None
        board.push(move)
        try:
            after = self.value(board, color)
        finally:
            board.pop()
        return float(max(0, before - after))

    def close(self):
        self.engine.quit()
