#!/usr/bin/env python3
"""
Game Scorer - Score-Based Learning System

Instead of binary win/loss, calculate nuanced game scores:

Scoring Rules:
1. Own king value: 100 (constant)
2. Opponent king value: 100 - rounds_played (decays each round)
3. Material values: P=100, N=320, B=330, R=500, Q=900

Final Score Calculation:
- If AI wins (checkmates opponent):
    score = own_material + (100 - rounds_played)

    Examples:
    - Round 10 checkmate: material + 90 = BEST
    - Round 50 checkmate: material + 50 = GOOD but slower

- If AI loses (own king checkmated):
    score = -100 (WORST OUTCOME)

- If draw:
    score = 0 (neutral)

This naturally teaches:
- Quick checkmate is valuable (opponent king worth more early)
- Material advantage matters
- Losing is catastrophic (-100)
- Winning quickly > winning slowly
"""

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

from typing import Tuple


class GameScorer:
    """
    Calculates game scores based on observable outcomes

    NO hardcoded chess knowledge - just observable material and rounds
    """

    # Standard piece values (centipawns)
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
        'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0
    }

    def __init__(self):
        pass

    def calculate_final_score(self, board: 'chess.Board', ai_color: 'chess.Color',
                             rounds_played: int) -> Tuple[float, str]:
        """
        Calculate final score for a completed game

        Args:
            board: Final board position
            ai_color: Color AI played
            rounds_played: Number of full rounds (move pairs)

        Returns:
            (final_score, result_type)

        Score Breakdown:
        - Win:  own_material + (100 - rounds_played)
        - Loss: -100
        - Draw: 0
        """
        if not CHESS_AVAILABLE:
            return 0.0, 'unknown'

        # Determine game result
        if board.is_checkmate():
            if board.turn != ai_color:
                # AI delivered checkmate
                result_type = 'win'
                own_material = self._calculate_material(board, ai_color)
                opponent_king_value = max(0, 100 - rounds_played)  # Decays with time
                final_score = own_material + opponent_king_value

            else:
                # AI was checkmated
                result_type = 'loss'
                final_score = -100.0  # Catastrophic

        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_fifty_moves() or board.is_repetition():
            # Draw
            result_type = 'draw'
            final_score = 0.0

        else:
            # Game not finished (shouldn't happen)
            result_type = 'unfinished'
            final_score = 0.0

        return final_score, result_type

    def _calculate_material(self, board: 'chess.Board', color: 'chess.Color') -> float:
        """
        Calculate total material value for a side

        Observable: just count pieces and sum their values
        """
        if not CHESS_AVAILABLE:
            return 0.0

        total_material = 0.0

        # Count all pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                piece_symbol = piece.symbol()
                total_material += self.PIECE_VALUES.get(piece_symbol, 0)

        return total_material

    def calculate_material_delta(self, fen_before: str, fen_after: str,
                                ai_color: 'chess.Color') -> float:
        """
        Calculate material change from one move

        Positive = AI gained material
        Negative = AI lost material

        Args:
            fen_before: Position before move
            fen_after: Position after move
            ai_color: AI's color

        Returns:
            Material delta (positive = gain)
        """
        if not CHESS_AVAILABLE:
            return 0.0

        try:
            board_before = chess.Board(fen_before)
            board_after = chess.Board(fen_after)

            material_before = self._calculate_material(board_before, ai_color)
            material_after = self._calculate_material(board_after, ai_color)

            return material_after - material_before

        except:
            return 0.0

    def get_move_count_from_fen(self, fen: str) -> int:
        """
        Extract move number from FEN string

        FEN format: ... <fullmove_number>
        """
        try:
            parts = fen.split()
            if len(parts) >= 6:
                return int(parts[5])
            return 1
        except:
            return 1


def test_game_scorer():
    """Test the game scorer"""
    if not CHESS_AVAILABLE:
        print("python-chess not available, skipping test")
        return

    print("=" * 70)
    print("GAME SCORER TEST")
    print("=" * 70)

    scorer = GameScorer()

    # Test 1: Quick checkmate (round 10)
    print("\nTest 1: Scholar's Mate (round 4)")
    board = chess.Board()
    moves = ['e4', 'e5', 'Bc4', 'Nc6', 'Qh5', 'Nf6', 'Qxf7#']
    for move_san in moves:
        board.push_san(move_san)

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=4)
    print(f"  Result: {result}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: material + (100 - 4 rounds) = material + 96")

    # Test 2: Late checkmate (round 50)
    print("\nTest 2: Late checkmate (round 50)")
    # Simulate by manually setting round count
    board = chess.Board()
    board.push_san('e4')
    board.push_san('e5')
    board.push_san('Qh5')
    board.push_san('Nc6')
    board.push_san('Qxf7#')

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=50)
    print(f"  Result: {result}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: material + (100 - 50 rounds) = material + 50")
    print(f"  Note: Quick mate (round 4) scored {96 - 50} points higher!")

    # Test 3: Loss
    print("\nTest 3: AI gets checkmated")
    board = chess.Board()
    moves = ['f3', 'e5', 'g4', 'Qh4#']  # Fool's mate
    for move_san in moves:
        board.push_san(move_san)

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=2)
    print(f"  Result: {result}")
    print(f"  Score: {score:.1f} (catastrophic)")

    # Test 4: Material calculation
    print("\nTest 4: Material calculation")
    board = chess.Board()
    white_material = scorer._calculate_material(board, chess.WHITE)
    black_material = scorer._calculate_material(board, chess.BLACK)
    print(f"  White material: {white_material:.0f} (8 pawns + 2 knights + 2 bishops + 2 rooks + 1 queen)")
    print(f"  Black material: {black_material:.0f}")
    print(f"  Expected: {8*100 + 2*320 + 2*330 + 2*500 + 900} each")

    # Test 5: Material delta
    print("\nTest 5: Material delta from capture")
    board = chess.Board()
    fen_before = board.fen()
    board.push_san('e4')
    board.push_san('d5')
    board.push_san('exd5')  # White captures pawn
    fen_after = board.fen()

    delta = scorer.calculate_material_delta(fen_before, fen_after, chess.WHITE)
    print(f"  Material delta after exd5: +{delta:.0f} (captured pawn)")

    print("\n" + "=" * 70)
    print("\nScoring System Summary:")
    print("  Win round 10:   material + 90 pts  (BEST)")
    print("  Win round 30:   material + 70 pts  (GOOD)")
    print("  Win round 50:   material + 50 pts  (OK)")
    print("  Draw:           0 pts               (NEUTRAL)")
    print("  Loss:           -100 pts            (WORST)")
    print("\nSystem learns: Quick checkmate > Slow checkmate > Draw > Loss")
    print("=" * 70)


if __name__ == '__main__':
    test_game_scorer()
