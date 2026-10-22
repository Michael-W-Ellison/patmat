#!/usr/bin/env python3
"""Pentago adapter for game-independent pattern learning (see general_move_patterns.py)"""

from .pentago_board import PentagoBoard, Color
from .pentago_game import PentagoGame
from .pentago_scorer import PentagoScorer


class _ShallowCopyBoard(PentagoBoard):
    """
    PentagoBoard whose copy() copies the rows but shares the Marble objects.
    Marbles are never changed in place (placing and rotating store new
    Marble objects), so this is equivalent to PentagoBoard.copy()'s deepcopy
    and much cheaper. PentagoGame.copy() calls board.copy(), so games made
    from this board keep using it through PentagoGame.make_move.
    """

    def copy(self):
        new_board = _ShallowCopyBoard.__new__(_ShallowCopyBoard)
        new_board.board = [row[:] for row in self.board]
        new_board.marble_count = self.marble_count
        return new_board


def _windows():
    """The 32 five-cell lines that PentagoBoard.check_five_in_row / count_threats scan"""
    lines = []
    for row in range(6):
        for col in range(2):
            lines.append([(row, col + i) for i in range(5)])
    for col in range(6):
        for row in range(2):
            lines.append([(row + i, col) for i in range(5)])
    for row in range(2):
        for col in range(2):
            lines.append([(row + i, col + i) for i in range(5)])
    for row in range(2):
        for col in range(4, 6):
            lines.append([(row + i, col - i) for i in range(5)])
    return lines


_WINDOWS = _windows()


class PentagoAdapter:
    """State is a PentagoGame; color is Color.WHITE / Color.BLACK"""

    def __init__(self):
        self.scorer = PentagoScorer()

    def legal_moves(self, game, color):
        if game.current_player != color:
            game = game.copy()
            game.current_player = color
        return game.get_legal_moves()

    def apply(self, game, move):
        # PentagoGame.make_move works on a copy and returns the new game
        if not isinstance(game.board, _ShallowCopyBoard):
            board = _ShallowCopyBoard.__new__(_ShallowCopyBoard)
            board.board = [row[:] for row in game.board.board]
            board.marble_count = game.board.marble_count
            start = PentagoGame(board=board, current_player=game.current_player)
            start.winner = game.winner
            start.is_draw = game.is_draw
            game = start
        return game.make_move(move)

    def advantage(self, game, color):
        """
        Scorer's position value (threats, centre) of color minus the opponent's,
        i.e. scorer._evaluate_color(board, color) - scorer._evaluate_color(board, opponent).

        Computed in one pass over the board's 32 lines instead of the eight
        passes the scorer makes (five-in-row check and 2/3/4 threat counts per
        color), with the scorer's own weights; gives the same values.
        """
        return self._evaluate_both(game.board)[color]

    def _evaluate_both(self, board):
        s = self.scorer
        cells = board.board
        value = {Color.WHITE: 0.0, Color.BLACK: 0.0}
        five = {Color.WHITE: False, Color.BLACK: False}
        for line in _WINDOWS:
            white = black = 0
            for r, c in line:
                marble = cells[r][c]
                if marble is not None:
                    if marble.color == Color.WHITE:
                        white += 1
                    else:
                        black += 1
            for mine, theirs, col in ((white, black, Color.WHITE), (black, white, Color.BLACK)):
                if theirs == 0 and mine >= 2:
                    if mine == 5:
                        five[col] = True
                    value[col] += (s.THREAT_2 + (s.THREAT_3 if mine >= 3 else 0)
                                   + (s.THREAT_4 if mine >= 4 else 0))
        for row in cells:
            for marble in row:
                if marble is not None:
                    value[marble.color] += s._position_value(marble.position)
        for col in five:
            if five[col]:
                value[col] = s.THREAT_5
        return {Color.WHITE: value[Color.WHITE] - value[Color.BLACK],
                Color.BLACK: value[Color.BLACK] - value[Color.WHITE]}

    def is_over(self, game):
        return game.is_game_over()

    def winner(self, game):
        return game.get_winner()

    def to_move(self, game):
        return game.current_player

    def opponent(self, color):
        return color.opposite()
