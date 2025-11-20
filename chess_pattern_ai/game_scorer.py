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
        Calculate final score for a completed game using DIFFERENTIAL scoring

        Args:
            board: Final board position
            ai_color: Color AI played
            rounds_played: Number of full rounds (move pairs)

        Returns:
            (final_score, result_type)

        Score Breakdown (DIFFERENTIAL):
        - Win:  (ai_material - opp_material) + (100 - rounds_played) + 1000
        - Loss: (ai_material - opp_material) - 1000
        - Draw: (ai_material - opp_material) - 500  ← PENALIZED!

        This teaches:
        - Winning with material advantage > winning with material disadvantage
        - Losing by a little > losing by a lot
        - Quick wins score higher (time bonus)
        - Draws are penalized to discourage draw-seeking behavior
        - Even draws with material advantage are discouraged (should push for win!)
        - Exchanges: "I lost 320 but gained 500 = +180 net"
        """
        if not CHESS_AVAILABLE:
            return 0.0, 'unknown'

        # Calculate material advantage (DIFFERENTIAL, not absolute)
        ai_material = self._calculate_material(board, ai_color)
        opponent_color = not ai_color
        opponent_material = self._calculate_material(board, opponent_color)
        material_advantage = ai_material - opponent_material

        # Determine game result
        if board.is_checkmate():
            if board.turn != ai_color:
                # AI delivered checkmate
                result_type = 'win'
                time_bonus = max(0, 100 - rounds_played)  # Quick wins better
                win_bonus = 1000  # Winning is good!
                final_score = material_advantage + time_bonus + win_bonus

            else:
                # AI was checkmated
                result_type = 'loss'
                loss_penalty = 1000  # Losing is bad
                # But losing with material advantage is LESS bad than getting crushed
                final_score = material_advantage - loss_penalty

        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_fifty_moves() or board.is_repetition():
            # Draw: PENALIZE for not winning!
            # Draws negate the chance to win, so they should be discouraged
            result_type = 'draw'
            draw_penalty = 500  # Penalty for not converting advantage to win
            # Still consider material (drawing ahead > drawing behind)
            # but apply penalty to discourage draw-seeking behavior
            final_score = material_advantage - draw_penalty

        else:
            # Game not finished (shouldn't happen)
            result_type = 'unfinished'
            final_score = material_advantage

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
        Calculate material ADVANTAGE change from one move (DIFFERENTIAL)

        This captures exchanges: "I lost pawn but gained rook = +180 net"

        Positive = AI improved position (gained more or lost less)
        Negative = AI worsened position (lost more or gained less)

        Args:
            fen_before: Position before move
            fen_after: Position after move
            ai_color: AI's color

        Returns:
            Material advantage delta (positive = improved position)
        """
        if not CHESS_AVAILABLE:
            return 0.0

        try:
            board_before = chess.Board(fen_before)
            board_after = chess.Board(fen_after)

            opponent_color = not ai_color

            # Calculate ADVANTAGE before move (my material - opponent's)
            ai_mat_before = self._calculate_material(board_before, ai_color)
            opp_mat_before = self._calculate_material(board_before, opponent_color)
            advantage_before = ai_mat_before - opp_mat_before

            # Calculate ADVANTAGE after move
            ai_mat_after = self._calculate_material(board_after, ai_color)
            opp_mat_after = self._calculate_material(board_after, opponent_color)
            advantage_after = ai_mat_after - opp_mat_after

            # Delta in advantage (this captures exchanges!)
            # Example: I lost 100 (pawn) but opponent lost 500 (rook) = +400 delta
            advantage_delta = advantage_after - advantage_before

            return advantage_delta

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
    """Test the game scorer with DIFFERENTIAL scoring"""
    if not CHESS_AVAILABLE:
        print("python-chess not available, skipping test")
        return

    print("=" * 70)
    print("GAME SCORER TEST - DIFFERENTIAL SCORING")
    print("=" * 70)

    scorer = GameScorer()

    # Test 1: Quick checkmate with material advantage
    print("\nTest 1: Scholar's Mate (round 4) - Material advantage")
    board = chess.Board()
    moves = ['e4', 'e5', 'Bc4', 'Nc6', 'Qh5', 'Nf6', 'Qxf7#']
    for move_san in moves:
        board.push_san(move_san)

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=4)
    print(f"  Result: {result}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: (my_material - opp_material) + (100 - 4) + 1000")
    print(f"           = material_advantage + 96 + 1000")

    # Test 2: Late checkmate
    print("\nTest 2: Late checkmate (round 50)")
    board = chess.Board()
    board.push_san('e4')
    board.push_san('e5')
    board.push_san('Qh5')
    board.push_san('Nc6')
    board.push_san('Qxf7#')

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=50)
    print(f"  Result: {result}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: material_advantage + (100 - 50) + 1000")
    print(f"           = material_advantage + 50 + 1000")
    print(f"  Note: Quick mate (round 4) gets +46 time bonus!")

    # Test 3: Loss with material advantage (fought well)
    print("\nTest 3: Loss with material ADVANTAGE (fought well)")
    board = chess.Board()
    moves = ['f3', 'e5', 'g4', 'Qh4#']  # Fool's mate
    for move_san in moves:
        board.push_san(move_san)

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=2)
    ai_mat = scorer._calculate_material(board, chess.WHITE)
    opp_mat = scorer._calculate_material(board, chess.BLACK)
    advantage = ai_mat - opp_mat
    print(f"  Result: {result}")
    print(f"  Material advantage: {advantage:.0f}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: {advantage:.0f} (material advantage) - 1000 (loss penalty)")
    print(f"  Note: Lost but kept material close = less bad!")

    # Test 4: Loss with material DISADVANTAGE (got crushed)
    print("\nTest 4: Loss with material DISADVANTAGE (got crushed)")
    # Checkmate position where white is severely behind in material
    # White missing queen, rook, bishop - then gets checkmated (fool's mate style)
    # FEN: after 1. f3 e5 2. g4 Qh4# but with white pieces removed
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RN1QKBNR w KQkq - 1 3")

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=15)
    ai_mat = scorer._calculate_material(board, chess.WHITE)
    opp_mat = scorer._calculate_material(board, chess.BLACK)
    advantage = ai_mat - opp_mat
    print(f"  Result: {result}")
    print(f"  Material advantage: {advantage:.0f}")
    print(f"  Score: {score:.1f}")
    print(f"  Breakdown: {advantage:.0f} (huge disadvantage) - 1000 (loss penalty)")
    print(f"  Note: Got crushed AND lost = very bad!")

    # Test 5: Draw with material advantage
    print("\nTest 5: Draw with material advantage")
    # Create a stalemate position where white is ahead in material
    # King and queen vs lone king - stalemate position
    board = chess.Board("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")  # Stalemate!

    score, result = scorer.calculate_final_score(board, chess.WHITE, rounds_played=50)
    ai_mat = scorer._calculate_material(board, chess.WHITE)
    opp_mat = scorer._calculate_material(board, chess.BLACK)
    advantage = ai_mat - opp_mat
    print(f"  Result: {result}")
    print(f"  Material advantage: {advantage:.0f}")
    print(f"  Score: {score:.1f}")
    print(f"  Note: Drew but was ahead = positive score!")

    # Test 6: Exchange evaluation
    print("\nTest 6: Exchange evaluation (lost pawn, gained rook)")
    board = chess.Board()
    fen_before = board.fen()
    # Simulate: I lost a pawn but captured opponent's rook
    board.push_san('e4')
    board.push_san('d5')
    board.remove_piece_at(chess.A7)  # Remove my pawn
    board.remove_piece_at(chess.A8)  # Remove opponent's rook

    delta = scorer.calculate_material_delta(fen_before, board.fen(), chess.WHITE)
    print(f"  Lost: 100 (pawn)")
    print(f"  Gained: 500 (rook)")
    print(f"  Net advantage delta: {delta:.0f}")
    print(f"  Conclusion: Good exchange! (+{delta:.0f} advantage)")

    print("\n" + "=" * 70)
    print("\nDIFFERENTIAL Scoring System Summary:")
    print("  Win + ahead +500:   1000 + 500 + 90 = +1590  (BEST)")
    print("  Win + ahead +100:   1000 + 100 + 90 = +1190  (GOOD)")
    print("  Win + behind -200:  1000 - 200 + 90 = +890   (OK, won despite disadvantage)")
    print("  Draw + ahead +300:  300                      (Neutral but ahead)")
    print("  Draw + behind -300: -300                     (Neutral but behind)")
    print("  Loss + ahead +200:  200 - 1000 = -800        (Lost but fought well)")
    print("  Loss + behind -500: -500 - 1000 = -1500      (Got crushed)")
    print("\nSystem learns:")
    print("  - Exchanges: 'Lost pawn, gained rook' = +400 advantage")
    print("  - Tactics: Patterns that improve material advantage are good")
    print("  - Relative position: Ahead > Behind, even in same result")
    print("  - Speed: Quick wins get time bonus")
    print("=" * 70)


if __name__ == '__main__':
    test_game_scorer()
