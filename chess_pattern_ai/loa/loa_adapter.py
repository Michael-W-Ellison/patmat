#!/usr/bin/env python3
"""Lines of Action adapter for game-independent pattern learning (see general_move_patterns.py)"""

from grid_vocabulary import Grid

from .loa_board import Color
from .loa_game import LOAGame
from .loa_scorer import LOAScorer


class LOAAdapter:
    """State is a LOAGame; color is Color.BLACK / Color.WHITE"""

    def __init__(self):
        self.scorer = LOAScorer()

    def legal_moves(self, game, color):
        if game.current_player != color:
            game = game.copy()
            game.current_player = color
        return game.get_legal_moves()

    def apply(self, game, move):
        """
        Same result as LOAGame.make_move (which works on a copy), but the
        "opponent has no legal moves" rule only looks for one legal move
        instead of generating all of them: make_move generates the full
        list and throws it away, which made feature extraction several
        times slower than the rest of move selection.
        """
        new_game = game.copy()
        if not new_game.board.move_piece(move.from_pos, move.to_pos):
            raise ValueError(f"Invalid move: {move}")

        # Connected exactly when there is at most one group; the group counts
        # are kept on the new state so advantage() need not flood-fill again
        mover = new_game.current_player
        groups = new_game._adapter_groups = {}
        groups[mover] = new_game.board.count_groups(mover)
        if groups[mover] <= 1:
            new_game.winner = mover
            return new_game
        groups[mover.opposite()] = new_game.board.count_groups(mover.opposite())
        if groups[mover.opposite()] <= 1:
            new_game.winner = mover.opposite()
            return new_game

        new_game.current_player = mover.opposite()
        if new_game.is_game_over() or not self._has_legal_move(new_game):
            new_game.winner = mover
        return new_game

    @staticmethod
    def _has_legal_move(game):
        """Whether game.get_legal_moves() would be non-empty (stops at the first move)"""
        board = game.board
        for piece in board.get_pieces(game.current_player):
            from_pos = piece.position
            for dr, dc in LOAGame.DIRECTIONS:
                count = board.count_pieces_in_line(from_pos, (dr, dc))
                to_pos = (from_pos[0] + dr * count, from_pos[1] + dc * count)
                if board._is_valid_position(to_pos) and board.can_jump_to(from_pos, to_pos):
                    return True
        return False

    def advantage(self, game, color):
        """
        Scorer's position value (connection, groups, material, centre) of color
        minus the opponent's, i.e. scorer._evaluate_color(board, color) -
        scorer._evaluate_color(board, opponent).
        """
        known = getattr(game, '_adapter_groups', {})
        return (self._evaluate_color(game.board, color, known.get(color))
                - self._evaluate_color(game.board, self.opponent(color),
                                       known.get(self.opponent(color))))

    def _evaluate_color(self, board, color, groups=None):
        """
        Same value as LOAScorer._evaluate_color with at most one flood fill
        instead of two: all pieces are connected exactly when there is at most
        one group.
        """
        s = self.scorer
        if groups is None:
            groups = board.count_groups(color)
        if groups <= 1:
            return s.CONNECTED_BONUS
        pieces = board.get_pieces(color)
        score = 0.0
        score += groups * s.GROUP_PENALTY
        score += len(pieces) * s.PIECE_VALUE
        for piece in pieces:
            score += s._position_value(piece.position)
        return score

    def to_grid(self, game):
        """
        8x8 square Grid in the board's own coordinates (row 0 is the top
        row, black's starting row). Empty 0; black 1, white 2. Groups are
        components(..., diagonal=True) (8-neighbour connectivity).
        """
        return Grid([[0 if p is None else (1 if p.color == Color.BLACK else 2) for p in row]
                     for row in game.board.board], 'square')

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.current_player

    def opponent(self, color):
        return color.opposite()
