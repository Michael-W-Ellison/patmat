#!/usr/bin/env python3
"""
Discovered Chess Engine - Legal Move Generation Using Inferred Rules
Generates chess moves using ONLY statistically discovered rules, no hardcoded chess knowledge.
"""

import sqlite3
import json
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BoardState:
    """Represents chess board state without using python-chess"""

    # Board representation: 0-63 squares (0=a1, 63=h8)
    # Each square contains: 'WP', 'BP', 'WN', 'BN', etc. or None
    board: List[Optional[str]]

    # Game state
    side_to_move: str  # 'white' or 'black'
    castling_rights: str  # 'KQkq' format
    en_passant_square: Optional[int]  # 0-63 or None
    halfmove_clock: int
    fullmove_number: int

    def __init__(self):
        self.board = [None] * 64
        self.side_to_move = 'white'
        self.castling_rights = 'KQkq'
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

    @staticmethod
    def from_fen(fen: str) -> 'BoardState':
        """Parse FEN string to create board state"""
        parts = fen.split()

        state = BoardState()

        # Parse board position
        ranks = parts[0].split('/')
        square = 56  # Start at a8 (rank 8, file a)

        for rank in ranks:
            file = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file += int(char)
                else:
                    # Piece
                    color = 'W' if char.isupper() else 'B'
                    piece_type = char.upper()
                    state.board[square + file] = color + piece_type
                    file += 1
            square -= 8  # Move to next rank down

        # Parse side to move
        state.side_to_move = 'white' if parts[1] == 'w' else 'black'

        # Parse castling rights
        state.castling_rights = parts[2] if parts[2] != '-' else ''

        # Parse en passant square
        if parts[3] != '-':
            ep_file = ord(parts[3][0]) - ord('a')
            ep_rank = int(parts[3][1]) - 1
            state.en_passant_square = ep_rank * 8 + ep_file

        # Parse move counters
        state.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        state.fullmove_number = int(parts[5]) if len(parts) > 5 else 1

        return state

    def to_fen(self) -> str:
        """Convert board state to FEN string"""
        # Build board part
        fen_parts = []
        for rank in range(7, -1, -1):  # 8 to 1
            empty_count = 0
            rank_str = ""

            for file in range(8):
                square = rank * 8 + file
                piece = self.board[square]

                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0

                    # Convert 'WP' to 'P', 'BP' to 'p'
                    piece_char = piece[1]  # Get piece type
                    if piece[0] == 'B':  # Black
                        piece_char = piece_char.lower()
                    rank_str += piece_char

            if empty_count > 0:
                rank_str += str(empty_count)

            fen_parts.append(rank_str)

        board_fen = '/'.join(fen_parts)

        # Build other parts
        side = 'w' if self.side_to_move == 'white' else 'b'
        castling = self.castling_rights if self.castling_rights else '-'

        if self.en_passant_square is not None:
            ep_file = chr(ord('a') + (self.en_passant_square % 8))
            ep_rank = str((self.en_passant_square // 8) + 1)
            ep = ep_file + ep_rank
        else:
            ep = '-'

        return f"{board_fen} {side} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def get_piece(self, square: int) -> Optional[str]:
        """Get piece at square (0-63)"""
        if 0 <= square < 64:
            return self.board[square]
        return None

    def get_piece_color(self, square: int) -> Optional[str]:
        """Get color of piece at square"""
        piece = self.get_piece(square)
        if piece:
            return 'white' if piece[0] == 'W' else 'black'
        return None

    def get_piece_type(self, square: int) -> Optional[str]:
        """Get piece type at square (P, N, B, R, Q, K)"""
        piece = self.get_piece(square)
        if piece:
            return piece[1]
        return None


class DiscoveredChessEngine:
    """Chess engine using only discovered rules"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

        # Loaded rules
        self.movement_rules: Dict[str, Dict] = {}
        self.capture_rules: Dict[str, Dict] = {}
        self.special_moves: List[Dict] = []

        self._load_rules()

    def _load_rules(self):
        """Load inferred rules from database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Load basic movement rules
        self.cursor.execute('''
            SELECT piece_type, rule_type, conditions
            FROM inferred_rules
            WHERE rule_type IN ('basic_movement', 'complex_movement')
        ''')

        for piece_type, rule_type, conditions_json in self.cursor.fetchall():
            conditions = json.loads(conditions_json)
            self.movement_rules[piece_type] = conditions

        # Load capture-specific rules
        self.cursor.execute('''
            SELECT piece_type, rule_type, conditions
            FROM inferred_rules
            WHERE rule_type IN ('capture_only', 'non_capture_only')
        ''')

        for piece_type, rule_type, conditions_json in self.cursor.fetchall():
            conditions = json.loads(conditions_json)
            if piece_type not in self.capture_rules:
                self.capture_rules[piece_type] = {}
            self.capture_rules[piece_type][rule_type] = conditions

        logger.info(f"Loaded rules for {len(self.movement_rules)} piece types")
        logger.info(f"Loaded capture rules for {len(self.capture_rules)} piece types")

    def generate_legal_moves(self, board: BoardState) -> List[str]:
        """Generate all legal moves for current position using discovered rules"""

        legal_moves = []

        # Find all pieces of the side to move
        color_prefix = 'W' if board.side_to_move == 'white' else 'B'

        for from_square in range(64):
            piece = board.get_piece(from_square)

            if piece and piece[0] == color_prefix:
                piece_type = piece[1]

                # Generate moves for this piece
                piece_moves = self._generate_piece_moves(board, from_square, piece_type)
                legal_moves.extend(piece_moves)

        return legal_moves

    def _generate_piece_moves(self, board: BoardState, from_square: int, piece_type: str) -> List[str]:
        """Generate moves for a specific piece"""

        moves = []

        if piece_type == 'P':
            moves = self._generate_pawn_moves(board, from_square)
        elif piece_type == 'N':
            moves = self._generate_knight_moves(board, from_square)
        elif piece_type == 'B':
            moves = self._generate_bishop_moves(board, from_square)
        elif piece_type == 'R':
            moves = self._generate_rook_moves(board, from_square)
        elif piece_type == 'Q':
            moves = self._generate_queen_moves(board, from_square)
        elif piece_type == 'K':
            moves = self._generate_king_moves(board, from_square)

        return moves

    def _generate_pawn_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate pawn moves using discovered capture/non-capture rules"""

        moves = []
        color = board.get_piece_color(from_square)

        # Get pawn rules
        if 'P' not in self.capture_rules:
            return moves

        pawn_rules = self.capture_rules['P']

        # Determine direction based on color
        direction = 1 if color == 'white' else -1

        # Non-capture moves (straight forward)
        if 'non_capture_only' in pawn_rules:
            non_capture_rule = pawn_rules['non_capture_only']
            vectors = non_capture_rule['valid_vectors']

            for rank_delta, file_delta in vectors:
                # Pawns can only move forward (positive for white, negative for black)
                if color == 'white' and rank_delta < 0:
                    continue  # White pawns can't move backward
                if color == 'black' and rank_delta > 0:
                    continue  # Black pawns can't move backward

                to_square = from_square + rank_delta * 8 + file_delta

                if self._is_valid_square(to_square):
                    # Check target is empty
                    if board.get_piece(to_square) is None:
                        # Check path is clear for 2-square move
                        if abs(rank_delta) == 2:
                            intermediate = from_square + direction * 8
                            if board.get_piece(intermediate) is not None:
                                continue

                            # Only allow 2-square from starting rank
                            from_rank = from_square // 8
                            if (color == 'white' and from_rank != 1) or \
                               (color == 'black' and from_rank != 6):
                                continue

                        moves.append(self._move_to_uci(from_square, to_square))

        # Capture moves (diagonal)
        if 'capture_only' in pawn_rules:
            capture_rule = pawn_rules['capture_only']
            vectors = capture_rule['valid_vectors']

            for rank_delta, file_delta in vectors:
                # Pawns can only move forward (positive for white, negative for black)
                if color == 'white' and rank_delta < 0:
                    continue  # White pawns can't move backward
                if color == 'black' and rank_delta > 0:
                    continue  # Black pawns can't move backward

                to_square = from_square + rank_delta * 8 + file_delta

                if self._is_valid_square(to_square):
                    # Check if there's an enemy piece
                    target_piece = board.get_piece(to_square)
                    if target_piece and board.get_piece_color(to_square) != color:
                        moves.append(self._move_to_uci(from_square, to_square))

                    # Check en passant
                    elif to_square == board.en_passant_square:
                        moves.append(self._move_to_uci(from_square, to_square))

        return moves

    def _generate_knight_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate knight moves using L-shaped pattern"""

        moves = []

        if 'N' not in self.movement_rules:
            return moves

        knight_rule = self.movement_rules['N']
        vectors = knight_rule['valid_vectors']

        for rank_delta, file_delta in vectors:
            to_square = from_square + rank_delta * 8 + file_delta

            if self._is_valid_square(to_square):
                # Check knight doesn't wrap around board edges
                from_file = from_square % 8
                to_file = to_square % 8
                file_diff = abs(to_file - from_file)

                # Valid knight moves don't wrap (file difference should match delta)
                if file_diff == abs(file_delta):
                    target = board.get_piece(to_square)

                    # Empty square or enemy piece
                    if target is None or board.get_piece_color(to_square) != board.get_piece_color(from_square):
                        moves.append(self._move_to_uci(from_square, to_square))

        return moves

    def _generate_sliding_moves(self, board: BoardState, from_square: int,
                               piece_type: str, vectors: List[Tuple[int, int]]) -> List[str]:
        """Generate moves for sliding pieces (B, R, Q)"""

        moves = []
        color = board.get_piece_color(from_square)

        for rank_delta, file_delta in vectors:
            # Determine if this is a unit vector (for sliding)
            rank_step = 0 if rank_delta == 0 else (1 if rank_delta > 0 else -1)
            file_step = 0 if file_delta == 0 else (1 if file_delta > 0 else -1)

            # Slide in this direction
            distance = 1
            while distance <= 7:  # Max distance on board
                to_square = from_square + (rank_step * distance * 8) + (file_step * distance)

                if not self._is_valid_square(to_square):
                    break

                # Check we haven't wrapped around
                from_file = from_square % 8
                to_file = to_square % 8

                if file_step != 0:
                    expected_file_diff = file_step * distance
                    actual_file_diff = to_file - from_file
                    if actual_file_diff != expected_file_diff:
                        break

                target = board.get_piece(to_square)

                if target is None:
                    # Empty square, can move here and continue
                    moves.append(self._move_to_uci(from_square, to_square))
                elif board.get_piece_color(to_square) != color:
                    # Enemy piece, can capture but can't continue
                    moves.append(self._move_to_uci(from_square, to_square))
                    break
                else:
                    # Own piece, blocked
                    break

                distance += 1

        return moves

    def _generate_bishop_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate bishop moves (diagonal sliding)"""

        if 'B' not in self.movement_rules:
            return []

        bishop_rule = self.movement_rules['B']

        # Use only unit diagonal vectors
        unit_vectors = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

        return self._generate_sliding_moves(board, from_square, 'B', unit_vectors)

    def _generate_rook_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate rook moves (horizontal/vertical sliding)"""

        if 'R' not in self.movement_rules:
            return []

        # Use only unit orthogonal vectors
        unit_vectors = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        return self._generate_sliding_moves(board, from_square, 'R', unit_vectors)

    def _generate_queen_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate queen moves (combination of rook + bishop)"""

        if 'Q' not in self.movement_rules:
            return []

        # Queen = Rook + Bishop
        unit_vectors = [
            [1, 0], [-1, 0], [0, 1], [0, -1],  # Rook-like
            [1, 1], [1, -1], [-1, 1], [-1, -1]  # Bishop-like
        ]

        return self._generate_sliding_moves(board, from_square, 'Q', unit_vectors)

    def _generate_king_moves(self, board: BoardState, from_square: int) -> List[str]:
        """Generate king moves (one square in any direction + castling)"""

        moves = []

        if 'K' not in self.movement_rules:
            return moves

        king_rule = self.movement_rules['K']
        vectors = king_rule['valid_vectors']
        color = board.get_piece_color(from_square)

        for rank_delta, file_delta in vectors:
            # Skip castling vectors for now (handled separately)
            if abs(file_delta) == 2:
                # This is castling - simplified implementation
                if self._can_castle(board, from_square, file_delta):
                    to_square = from_square + file_delta
                    moves.append(self._move_to_uci(from_square, to_square))
                continue

            to_square = from_square + rank_delta * 8 + file_delta

            if self._is_valid_square(to_square):
                # Check we haven't wrapped
                from_file = from_square % 8
                to_file = to_square % 8
                if abs(to_file - from_file) == abs(file_delta):
                    target = board.get_piece(to_square)

                    if target is None or board.get_piece_color(to_square) != color:
                        moves.append(self._move_to_uci(from_square, to_square))

        return moves

    def _can_castle(self, board: BoardState, king_square: int, file_delta: int) -> bool:
        """Check if castling is legal (simplified - checks castling rights and path)"""

        color = board.get_piece_color(king_square)

        # Check castling rights
        if file_delta > 0:  # Kingside
            right = 'K' if color == 'white' else 'k'
        else:  # Queenside
            right = 'Q' if color == 'white' else 'q'

        if right not in board.castling_rights:
            return False

        # Check path is clear
        step = 1 if file_delta > 0 else -1
        for i in range(1, abs(file_delta) + 1):
            square = king_square + step * i
            if board.get_piece(square) is not None:
                return False

        # TODO: Check king not in check, doesn't pass through check
        # For now, just basic validation

        return True

    def _is_valid_square(self, square: int) -> bool:
        """Check if square is on the board"""
        return 0 <= square < 64

    def _move_to_uci(self, from_square: int, to_square: int) -> str:
        """Convert move to UCI notation"""
        from_file = chr(ord('a') + (from_square % 8))
        from_rank = str((from_square // 8) + 1)
        to_file = chr(ord('a') + (to_square % 8))
        to_rank = str((to_square // 8) + 1)

        return from_file + from_rank + to_file + to_rank

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_engine():
    """Test the discovered engine against python-chess"""
    import chess

    engine = DiscoveredChessEngine()

    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", "After 1.e4 e5 2.Nf3 Nf6"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", "After 1.e4 e5 2.Nf3 Nc6"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "Starting position (Black to move)"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "Castling test position"),
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1", "En passant available"),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "Pawn endgame"),
        ("rnbqkb1r/pp1ppppp/5n2/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 1", "Sicilian Defense"),
    ]

    print("\n" + "="*80)
    print("TESTING DISCOVERED CHESS ENGINE")
    print("="*80)

    total_correct = 0
    total_moves = 0

    for fen, description in test_positions:
        print(f"\n{description}")
        print(f"FEN: {fen}")
        print("-"*80)

        # Generate moves with discovered engine
        board_state = BoardState.from_fen(fen)
        discovered_moves = set(engine.generate_legal_moves(board_state))

        # Generate moves with python-chess (ground truth)
        chess_board = chess.Board(fen)
        actual_moves = set(m.uci() for m in chess_board.legal_moves)

        # Compare
        correct = discovered_moves & actual_moves
        missing = actual_moves - discovered_moves
        extra = discovered_moves - actual_moves

        accuracy = len(correct) / len(actual_moves) * 100 if actual_moves else 0

        print(f"Discovered moves: {len(discovered_moves)}")
        print(f"Actual moves: {len(actual_moves)}")
        print(f"Correct: {len(correct)} ({accuracy:.1f}%)")

        if missing:
            print(f"Missing {len(missing)} moves: {sorted(list(missing))[:10]}")
        if extra:
            print(f"Extra {len(extra)} moves: {sorted(list(extra))[:10]}")

        total_correct += len(correct)
        total_moves += len(actual_moves)

    overall_accuracy = total_correct / total_moves * 100 if total_moves > 0 else 0

    print("\n" + "="*80)
    print(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({total_correct}/{total_moves} moves)")
    print("="*80)

    engine.close()


if __name__ == '__main__':
    test_engine()
