#!/usr/bin/env python3
"""
Refutation Patterns - which kinds of reply punish a blunder, in every game

When a judge (move_judges) rates one of the agent's moves a blunder (a loss
of at least BLUNDER_MEDIANS times the game's running median loss), its best
reply to that move is the refutation. Every legal reply in that position is
described by a game-independent reply pattern (reply_features), and the
learner counts how often replies of each pattern were seen there and how
often one was the refutation. A pattern's refutation rate is how many times
more often than an average reply it was the refutation (1: as often as
average), shrunk toward 1 by PRIOR_REPLIES; other games' counts enter as a
prior worth at most CROSS_GAME_REPLIES replies, as for the general patterns.

Use: the replies whose patterns refute most often are the ones worth a
deeper look before playing a move. analyze_candidates (general_move_patterns)
searches the REPLIES_EXTENDED best-rated replies to each candidate
EXTENSION_DEPTH plies further, and the chess search (pattern_search) reads
them below its root moves besides its usual quiet moves. A refutation that
these searches see only after its first ply (a quiet reply that wins later)
is what they otherwise miss.

Reply pattern 'gain=<g>|mob=<m>' for a reply by the side to move:
- gain: change in that side's advantage measure (adapter.advantage) the reply
        causes, compared with the other replies (best / positive / zero / negative)
- mob:  the other side's number of moves afterwards compared with the median
        over the replies (restricts / neutral / frees), or terminal / extra_turn
"""

import statistics
from typing import Dict, List, Optional, Sequence

BLUNDER_MEDIANS = 2.0       # a blunder: judged loss >= this many times the game's running median loss
PRIOR_REPLIES = 20.0        # pseudo-replies at rate 1 that a pattern's rate is shrunk toward
CROSS_GAME_REPLIES = 200.0  # most replies other games' evidence counts as
REPLIES_EXTENDED = 2        # best-rated replies searched further per candidate
EXTENSION_DEPTH = 2         # plies those replies are searched further (analyze_candidates)
MIN_RATE = 1.0              # only replies rated above this are searched further
EPSILON = 1e-9


def reply_features(adapter, state, mover, moves: Sequence) -> List[str]:
    """The reply pattern of each of mover's moves in state (see the module docstring), in the same order"""
    in_place = hasattr(adapter, 'push') and hasattr(adapter, 'pop')
    other = adapter.opponent(mover)
    now = adapter.advantage(state, mover)
    gains, counts, ends = [], [], []
    for move in moves:
        if in_place:
            adapter.push(state, move)
            after = state
        else:
            after = adapter.apply(state, move)
        if adapter.is_over(after):
            ends.append('terminal')
            gains.append(0.0)
            counts.append(None)
        else:
            gains.append(adapter.advantage(after, mover) - now)
            if adapter.to_move(after) == mover:
                ends.append('extra_turn')
                counts.append(None)
            else:
                ends.append(None)
                counts.append(len(adapter.legal_moves(after, other)))
        if in_place:
            adapter.pop(state)
    best = max(gains, default=0.0)
    known = [c for c in counts if c is not None]
    median = statistics.median(known) if known else 0
    out = []
    for gain, count, end in zip(gains, counts, ends):
        if gain > EPSILON:
            g = 'best' if gain >= best - EPSILON else 'positive'
        else:
            g = 'zero' if gain >= -EPSILON else 'negative'
        if end is not None:
            m = end
        elif count < median:
            m = 'restricts'
        elif count > median:
            m = 'frees'
        else:
            m = 'neutral'
        out.append(f'gain={g}|mob={m}')
    return out


class RefutationLearner:
    """Counts, per game, how often replies of each pattern refuted the agent's blunders (module docstring)"""

    TABLE = 'general_refutations'

    def __init__(self, conn, game_name: str, import_cross_game: bool = True):
        self.conn = conn
        self.game_name = game_name
        self.import_cross_game = import_cross_game
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                features TEXT NOT NULL,
                game TEXT NOT NULL,
                seen REAL DEFAULT 0.0,          -- replies of this pattern in positions after a blunder
                refuted REAL DEFAULT 0.0,       -- of those, the judge's best reply
                PRIMARY KEY (features, game)
            )
        ''')
        conn.commit()
        self._load()

    def _load(self):
        own: Dict[str, List[float]] = {}
        imported: Dict[str, List[float]] = {}
        for features, game, seen, refuted in self.conn.execute(
                f'SELECT features, game, seen, refuted FROM {self.TABLE}'):
            if game == self.game_name:
                target = own
            elif self.import_cross_game:
                target = imported
            else:
                continue
            entry = target.setdefault(features, [0.0, 0.0])
            entry[0] += seen
            entry[1] += refuted

        def base(counts):
            seen = sum(s for s, _ in counts.values())
            return sum(r for _, r in counts.values()) / seen if seen else 0.0

        own_base, imported_base = base(own), base(imported)
        self.rates: Dict[str, float] = {}
        for features in own.keys() | imported.keys():
            # each side's evidence as a ratio to its own base rate, so games with many or few replies per
            # position compare
            own_seen, own_refuted = own.get(features, (0.0, 0.0))
            imp_seen, imp_refuted = imported.get(features, (0.0, 0.0))
            k = min(imp_seen, CROSS_GAME_REPLIES)
            imp_ratio = (imp_refuted / imp_seen) / imported_base if imp_seen and imported_base else 1.0
            # own refutations in units of the base rate: own_seen if the pattern refutes as often as average
            own_units = own_refuted / own_base if own_base else 0.0
            own_n = own_seen if own_base else 0.0
            self.rates[features] = (own_units + imp_ratio * k + PRIOR_REPLIES) / (own_n + k + PRIOR_REPLIES)

    def rate(self, features: str) -> float:
        """How many times more often than an average reply replies of this pattern refuted a blunder (1: no evidence)"""
        return self.rates.get(features, 1.0)

    def has_evidence(self) -> bool:
        return bool(self.rates)

    def record(self, features: List[str], refutation: int):
        """One blunder: the reply patterns of all replies, and the index of the judge's best reply among them"""
        counts: Dict[str, List[float]] = {}
        for i, f in enumerate(features):
            entry = counts.setdefault(f, [0.0, 0.0])
            entry[0] += 1.0
            entry[1] += 1.0 if i == refutation else 0.0
        self.conn.executemany(f'''
            INSERT INTO {self.TABLE} (features, game, seen, refuted) VALUES (?, ?, ?, ?)
            ON CONFLICT(features, game) DO UPDATE SET
                seen = seen + excluded.seen, refuted = refuted + excluded.refuted
        ''', [(f, self.game_name, s, r) for f, (s, r) in counts.items()])
        self.conn.commit()
        self._load()

    def ranked(self, features: List[str], limit: int = REPLIES_EXTENDED,
               exclude: Optional[set] = None) -> List[int]:
        """Indices of the (at most limit) replies with the highest refutation rates above MIN_RATE"""
        order = sorted((i for i, f in enumerate(features)
                        if self.rate(f) > MIN_RATE and (exclude is None or i not in exclude)),
                       key=lambda i: -self.rate(features[i]))
        return order[:limit]
