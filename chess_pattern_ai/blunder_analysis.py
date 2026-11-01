#!/usr/bin/env python3
"""
Blunder Analysis - what the agent's blunders against Stockfish consist of

The agent plays games against Stockfish; a full-strength Stockfish (the
judge) scores every agent move by how many centipawns it loses. For each
blunder (a loss of BLUNDER centipawns or more) it records:

- kind:    'allowed' (the opponent can now gain) or 'missed' (the agent had a
           winning move, e.g. a mate, and did not play it, without falling into
           anything)
- motif:   what the opponent's best reply (by the judge) does: 'mate in N',
           'takes undefended piece', 'takes with cheaper piece', 'fork', 'pin',
           'skewer', 'discovered attack', 'attacks a piece', 'check', 'exchange'
           or 'quiet move'; for 'missed', what the agent's best move was
           ('mate in N', 'capture', 'fork', ... by the same tests)
- lost:    which of the agent's pieces is lost first in the judge's line: 'the
           piece moved' (the blunder put it where it is taken) or 'another piece'
- loss ply: the ply of the judge's line from which the agent is at least
           LOSS_UNITS of material down for good (None: no lasting material loss
           within the line, e.g. a mate or a positional loss)
- rank:    where the agent's own move ordering placed the opponent's best reply
           (0 = first)
- read:    whether that reply is among the moves the agent's search considers
           below the root (if not, the search never read it)
- depth:   how deep the agent's search went for the move

usage: blunder_analysis.py <db> <games> [--level N] [--out file.jsonl]
"""

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List, Optional

import chess
import chess.engine

from game_scorer import GameScorer

BLUNDER = 300           # centipawns lost that make a move a blunder
JUDGE_DEPTH = 12
LINE_PLIES = 10         # plies of the judge's line examined
LOSS_UNITS = 20         # material units (a pawn is 10) that count as a lasting loss
KING_WORTH = 1000       # a king's worth for "attacks a more valuable piece" (the scorer gives it 0)


def worth(piece_type: int) -> int:
    if piece_type == chess.KING:
        return KING_WORTH
    return GameScorer.PIECE_VALUES[chess.piece_symbol(piece_type).upper()]


def material(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece in board.piece_map().values():
        v = GameScorer.PIECE_VALUES[piece.symbol()]
        total += v if piece.color == color else -v
    return total


def _vulnerable(board: chess.Board, sq: int, attacker_worth: int) -> bool:
    """Is the piece on sq (attacked) worth more than the attacker, or undefended?"""
    piece = board.piece_at(sq)
    return worth(piece.piece_type) > attacker_worth or not board.is_attacked_by(piece.color, sq)


def motif(board: chess.Board, move: chess.Move) -> str:
    """What move (by the side to move on board) does; board unchanged on return"""
    mover = board.turn
    victim = not mover
    piece = board.piece_at(move.from_square)
    mover_worth = worth((move.promotion or piece.piece_type))
    if board.is_capture(move):
        target = board.piece_at(move.to_square)
        taken = chess.PAWN if target is None else target.piece_type     # (en passant)
        if target is not None and not board.is_attacked_by(victim, move.to_square):
            capture = 'takes undefended piece'
        elif worth(taken) > worth(piece.piece_type):
            capture = 'takes with cheaper piece'
        else:
            capture = 'exchange'
    else:
        capture = None

    # attacks the moved piece and the other pieces make once the move is played
    attacked_before = {sq: board.attackers_mask(mover, sq) for sq in chess.scan_forward(board.occupied_co[victim])}
    board.push(move)
    try:
        if board.is_checkmate():
            return 'mate in 1'
        to = move.to_square
        check = board.is_check()
        targets = [sq for sq in chess.scan_forward(board.attacks_mask(to) & board.occupied_co[victim])
                   if board.piece_type_at(sq) == chess.KING or _vulnerable(board, sq, mover_worth)]
        fork = len(targets) >= 2
        line = _line_motif(board, to, victim) if board.piece_type_at(to) in (chess.BISHOP, chess.ROOK, chess.QUEEN) else None
        discovered = False
        for sq in chess.scan_forward(board.occupied_co[victim]):
            new = board.attackers_mask(mover, sq) & ~attacked_before.get(sq, 0) & ~chess.BB_SQUARES[to]
            if new and (board.piece_type_at(sq) == chess.KING or
                        _vulnerable(board, sq, min(worth(board.piece_type_at(a)) for a in chess.scan_forward(new)))):
                discovered = True
                break
    finally:
        board.pop()
    if capture in ('takes undefended piece', 'takes with cheaper piece'):
        return capture
    if fork:
        return 'fork'
    if line:
        return line
    if discovered:
        return 'discovered attack'
    if capture:
        return capture
    if any(board.piece_type_at(t) != chess.KING for t in targets):
        return 'attacks a piece'
    if check:
        return 'check'
    return 'quiet move'


def _line_motif(board: chess.Board, sq: int, victim: chess.Color) -> Optional[str]:
    """For the slider on sq (just moved): 'pin' / 'skewer' along a line through one of victim's pieces"""
    slider_worth = worth(board.piece_type_at(sq))
    for front in chess.scan_forward(board.attacks_mask(sq) & board.occupied_co[victim]):
        # squares beyond front on the line from sq, nearest first
        beyond = [s for s in chess.scan_forward(chess.ray(sq, front)) if chess.BB_SQUARES[front] & chess.between(sq, s)]
        beyond.sort(key=lambda s: chess.square_distance(s, sq))
        behind = next((s for s in beyond if board.piece_at(s) is not None), None)
        if behind is None or board.color_at(behind) != victim:
            continue
        front_worth, behind_worth = worth(board.piece_type_at(front)), worth(board.piece_type_at(behind))
        if behind_worth > front_worth and behind_worth > slider_worth:
            return 'pin'
        if front_worth > behind_worth and front_worth >= slider_worth and _vulnerable(board, behind, slider_worth):
            return 'skewer'
    return None


def line_material(board: chess.Board, line: List[chess.Move], color: chess.Color) -> List[int]:
    """color's material along line (index 0: before it); board unchanged on return"""
    out = [material(board, color)]
    played = 0
    try:
        for move in line[:LINE_PLIES]:
            if move not in board.legal_moves:
                break
            board.push(move)
            played += 1
            out.append(material(board, color))
    finally:
        for _ in range(played):
            board.pop()
    return out


def loss_ply(values: List[int]) -> Optional[int]:
    """First ply from which material stays LOSS_UNITS or more below its start"""
    for k in range(1, len(values)):
        if all(v <= values[0] - LOSS_UNITS for v in values[k:]):
            return k
    return None


def first_lost(board: chess.Board, line: List[chess.Move], color: chess.Color, moved_to: int) -> Optional[str]:
    """Which of color's pieces the line captures first: 'the piece moved' (on moved_to) or 'another piece'"""
    squares = {moved_to}            # where the moved piece is, followed if it moves again
    played = 0
    try:
        for move in line[:LINE_PLIES]:
            if move not in board.legal_moves:
                break
            if board.turn != color and board.is_capture(move):
                return 'the piece moved' if move.to_square in squares else 'another piece'
            if board.turn == color and move.from_square in squares:
                squares = {move.to_square}
            board.push(move)
            played += 1
    finally:
        for _ in range(played):
            board.pop()
    return None


def score_of(info, color: chess.Color) -> int:
    return info['score'].pov(color).score(mate_score=10000)


def mate_in(info, color: chess.Color) -> Optional[int]:
    m = info['score'].pov(color).mate()
    return m if m is not None and m > 0 else None


def analyse_move(judge, board: chess.Board, move: chess.Move, agent, color: chess.Color) -> Optional[Dict]:
    """Judge the agent's move on board (agent to move; board unchanged on return): a record if it is a blunder"""
    before = judge.analyse(board, chess.engine.Limit(depth=JUDGE_DEPTH))
    board.push(move)
    try:
        after = judge.analyse(board, chess.engine.Limit(depth=JUDGE_DEPTH))
        b, a = score_of(before, color), score_of(after, color)
        loss = max(0, min(b, 1000) - max(a, -1000))
        if loss < BLUNDER:
            return None
        rec = {'fen': None, 'move': move.uci(), 'before': b, 'after': a, 'loss': loss,
               'depth': getattr(agent, 'depth_reached', None), 'ply': len(board.move_stack)}
        line = after.get('pv', [])
        opp_mate = mate_in(after, not color)
        if b >= 200 and a >= -100 and opp_mate is None:
            rec['kind'] = 'missed'
            board.pop()
            try:
                best = before.get('pv', [None])[0]
                own_mate = mate_in(before, color)
                rec['motif'] = f'mate in {own_mate}' if own_mate else (motif(board, best) if best else 'unknown')
                rec['fen'] = board.fen()
            finally:
                board.push(move)
            rec['lost'], rec['loss_ply'], rec['rank'] = None, None, None
            return rec
        rec['kind'] = 'allowed'
        board.pop()
        rec['fen'] = board.fen()
        board.push(move)
        if not line:
            rec['motif'], rec['lost'], rec['loss_ply'], rec['rank'] = 'unknown', None, None, None
            return rec
        reply = line[0]
        rec['reply'] = reply.uci()
        rec['motif'] = f'mate in {opp_mate}' if opp_mate else motif(board, reply)
        rec['lost'] = first_lost(board, line, color, move.to_square)
        rec['loss_ply'] = loss_ply(line_material(board, line, color))
        rec['rank'] = reply_rank(agent, board, reply)
        rec['read'] = reply_read(agent, board, reply)
        return rec
    finally:
        board.pop()


def reply_rank(agent, board: chess.Board, reply: chess.Move) -> Optional[int]:
    """Where the agent's move ordering puts reply among the opponent's moves (board: opponent to move)"""
    if agent is None:
        return None
    context = agent.sequences.context(board) if agent.use_patterns else []
    ordered = [m for m, _, _ in agent.ordered(board, context)]
    return ordered.index(reply) if reply in ordered else None


def reply_read(agent, board: chess.Board, reply: chess.Move) -> Optional[bool]:
    """Is reply among the moves the agent's search considers below the root (board: opponent to move)?"""
    if agent is None:
        return None
    context = agent.sequences.context(board) if agent.use_patterns else []
    return any(m == reply for m, _, _ in agent.candidates(board, context))


# ------------------------------------------------------------------- report

def report(records: List[Dict], moves_judged: int) -> str:
    out = [f'{moves_judged} agent moves judged, {len(records)} blunders '
           f'({len(records) / max(1, moves_judged) * 100:.1f}%)']
    for kind in ('allowed', 'missed'):
        rs = [r for r in records if r['kind'] == kind]
        if not rs:
            continue
        out.append(f'\n{kind}: {len(rs)}')
        for m, n in Counter(r['motif'] if not r['motif'].startswith('mate') else
                            ('mate in 1' if r['motif'] == 'mate in 1' else 'mate in 2+') for r in rs).most_common():
            out.append(f'  {m:26s} {n:4d}  {n / len(rs) * 100:5.1f}%')
    allowed = [r for r in records if r['kind'] == 'allowed']
    if allowed:
        out.append('\nallowed: piece lost first')
        for m, n in Counter(str(r['lost']) for r in allowed).most_common():
            out.append(f'  {m:26s} {n:4d}')
        out.append('allowed: ply of the line from which the material loss lasts (None: no lasting loss)')
        for m, n in sorted(Counter(r['loss_ply'] for r in allowed).items(), key=lambda e: (e[0] is None, e[0] or 0)):
            out.append(f'  {str(m):26s} {n:4d}')
        read = [r for r in allowed if r.get('read') is not None]
        if read:
            out.append('allowed: opponent\'s best reply not among the moves the agent\'s search considers '
                       f'below the root: {sum(not r["read"] for r in read)} of {len(read)}')
        out.append('allowed: agent already lost (judge score before the move <= -500): '
                   f'{sum(r["before"] <= -500 for r in allowed)} of {len(allowed)}')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('db')
    parser.add_argument('games', type=int)
    parser.add_argument('--level', type=int, default=0, help='Stockfish skill level of the opponent')
    parser.add_argument('--stockfish-path', default='/usr/games/stockfish')
    parser.add_argument('--out', default=None, help='write each blunder as a JSON line')
    args = parser.parse_args()

    import logging
    logging.disable(logging.INFO)
    from headless_trainer import HeadlessTrainer

    trainer = HeadlessTrainer(args.db, opponent='stockfish', stockfish_level=args.level,
                              stockfish_path=args.stockfish_path)
    judge = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    records, judged = [], [0]
    out = open(args.out, 'w') if args.out else None
    play = trainer.play_ai_move

    def judged_move(board, color):
        move = play(board, color)
        judged[0] += 1
        rec = analyse_move(judge, board, move, trainer.search if trainer.use_search else None, color)
        if rec is not None:
            records.append(rec)
            if out:
                out.write(json.dumps(rec) + '\n')
                out.flush()
        return move

    trainer.play_ai_move = judged_move
    results = Counter()
    for i in range(args.games):
        result, _, _ = trainer.play_game(chess.WHITE if i % 2 == 0 else chess.BLACK)
        results[result] += 1
        print(f'game {i + 1}: {result} ({len(records)} blunders so far)', file=sys.stderr, flush=True)
    print(f'results: {dict(results)}')
    print(report(records, judged[0]))
    judge.quit()
    trainer.close()
    if out:
        out.close()


if __name__ == '__main__':
    main()
