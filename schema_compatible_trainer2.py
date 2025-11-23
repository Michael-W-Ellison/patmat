#!/usr/bin/env python3
"""
Schema-Compatible Headless Trainer

This version automatically detects your database schema and works with both:
- Original schema (headless_training.db)
- Enhanced schema (enhanced_training.db)

It will work with whatever database you have without modification.
"""

import chess
import json
import csv
import time
import random
import sqlite3
from datetime import datetime
from game_scorer import GameScorer

class SchemaCompatibleMovePrioritizer:
    """
    Move prioritizer that automatically adapts to your database schema
    """
    
    def __init__(self, db_path="headless_training.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.move_priorities = {}
        self.schema_type = "unknown"
        
        self._detect_and_init_schema()
        self._load_priorities()

    def _detect_and_init_schema(self):
        """Detect which schema version your database uses and initialize accordingly"""
        
        # Check if table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_move_patterns'")
        table_exists = self.cursor.fetchone() is not None
        
        if table_exists:
            # Check which columns exist
            self.cursor.execute("PRAGMA table_info(learned_move_patterns)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if 'repetition_count' in columns:
                self.schema_type = "enhanced"
                print("📊 Detected enhanced schema database")
            else:
                self.schema_type = "original"
                print("📊 Detected original schema database")
        else:
            # Create table with original schema (simpler and more reliable)
            self.schema_type = "original"
            self._create_original_schema()
            print("📊 Created new original schema database")
    
    def _create_original_schema(self):
        """Create the original, simpler schema"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_move_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                distance_from_start INTEGER,
                game_phase TEXT,
                times_seen INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                games_lost INTEGER DEFAULT 0,
                games_drawn INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_score REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                priority_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(piece_type, move_category, distance_from_start, game_phase)
            )
        ''')
        self.conn.commit()
    
    def _load_priorities(self):
        """Load priorities using schema-appropriate query"""
        
        if self.schema_type == "enhanced":
            # Enhanced schema query
            self.cursor.execute('''
                SELECT piece_type, move_category, distance_from_start, game_phase,
                       repetition_count, moves_since_progress, total_material_level,
                       priority_score
                FROM learned_move_patterns
                WHERE times_seen >= 5
            ''')
            
            for row in self.cursor.fetchall():
                key = tuple(row[:-1])  # All except priority_score
                priority = row[-1]
                self.move_priorities[key] = priority
                
        else:
            # Original schema query
            self.cursor.execute('''
                SELECT piece_type, move_category, distance_from_start, game_phase, priority_score
                FROM learned_move_patterns
                WHERE times_seen >= 5
            ''')
            
            for row in self.cursor.fetchall():
                key = tuple(row[:-1])  # All except priority_score
                priority = row[-1]
                self.move_priorities[key] = priority

        print(f"✓ Loaded {len(self.move_priorities)} learned move patterns ({self.schema_type} schema)")
    
    def get_move_priority(self, board, move):
        """Get move priority - works with both schemas"""
        
        characteristics = self._classify_move(board, move)
        if not characteristics:
            return 50.0  # Default neutral priority
            
        if self.schema_type == "enhanced":
            # Enhanced schema key
            key = (
                characteristics['piece_type'],
                characteristics['move_category'], 
                characteristics['distance_from_start'],
                characteristics['game_phase'],
                characteristics.get('repetition_count', 0),
                characteristics.get('moves_since_progress', 25),
                characteristics.get('total_material_level', 'medium')
            )
        else:
            # Original schema key  
            key = (
                characteristics['piece_type'],
                characteristics['move_category'],
                characteristics['distance_from_start'],
                characteristics['game_phase']
            )
            
        return self.move_priorities.get(key, 50.0)
    
    def sort_moves_by_priority(self, board, moves):
        """Sort moves by priority"""
        move_priorities = [(move, self.get_move_priority(board, move)) for move in moves]
        move_priorities.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_priorities]
    
    def record_game_moves(self, game_moves, ai_color, result, final_score):
        """Record game moves - adapts to schema"""
        
        board = chess.Board()
        
        for i, (fen, move_uci, move_san) in enumerate(game_moves):
            board.set_fen(fen)
            move = chess.Move.from_uci(move_uci)
            
            if board.turn != ai_color:
                board.push(move)
                continue
                
            characteristics = self._classify_move(board, move)
            if not characteristics:
                board.push(move)
                continue
                
            # Update statistics using appropriate schema
            if self.schema_type == "enhanced":
                self._update_enhanced_statistics(characteristics, result, final_score)
            else:
                self._update_original_statistics(characteristics, result, final_score)
                
            board.push(move)
    
    def _update_original_statistics(self, characteristics, result, final_score):
        """Update statistics using original schema"""
        
        # Get current stats
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn, total_score
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ? AND distance_from_start = ? AND game_phase = ?
        ''', (characteristics['piece_type'], characteristics['move_category'], 
              characteristics['distance_from_start'], characteristics['game_phase']))

        row = self.cursor.fetchone()
        if row:
            times_seen, won, lost, drawn, total_score = row
        else:
            times_seen = won = lost = drawn = total_score = 0

        # Update counters
        times_seen += 1
        total_score += final_score

        if result == 'win':
            won += 1
        elif result == 'loss':
            lost += 1
        else:
            drawn += 1

        # Calculate metrics
        total_games = won + lost + drawn
        win_rate = won / total_games if total_games > 0 else 0.0
        avg_score = total_score / times_seen if times_seen > 0 else 0.0
        confidence = min(1.0, times_seen / 20.0)
        
        # Priority score: normalize average score and apply confidence
        normalized_score = (avg_score + 1500) / 31  # Scale to roughly 0-100
        priority_score = normalized_score * confidence

        # Insert or update
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase)
            DO UPDATE SET
                times_seen = ?,
                games_won = ?,
                games_lost = ?,
                games_drawn = ?,
                win_rate = ?,
                total_score = ?,
                avg_score = ?,
                confidence = ?,
                priority_score = ?,
                updated_at = datetime('now')
        ''', (characteristics['piece_type'], characteristics['move_category'],
              characteristics['distance_from_start'], characteristics['game_phase'],
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score,
              # ON CONFLICT DO UPDATE values
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score))

        self.conn.commit()
    
    def _update_enhanced_statistics(self, characteristics, result, final_score):
        """Update statistics using enhanced schema"""
        
        # Enhanced schema update logic (more complex)
        rep_count = characteristics.get('repetition_count', 0)
        moves_since = characteristics.get('moves_since_progress', 25)
        material_level = characteristics.get('total_material_level', 'medium')
        
        # Get current stats
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn, total_score
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ? AND distance_from_start = ? 
              AND game_phase = ? AND repetition_count = ? 
              AND moves_since_progress = ? AND total_material_level = ?
        ''', (characteristics['piece_type'], characteristics['move_category'], 
              characteristics['distance_from_start'], characteristics['game_phase'],
              rep_count, moves_since, material_level))

        row = self.cursor.fetchone()
        if row:
            times_seen, won, lost, drawn, total_score = row
        else:
            times_seen = won = lost = drawn = total_score = 0

        # Update counters
        times_seen += 1
        total_score += final_score

        if result == 'win':
            won += 1
        elif result == 'loss':
            lost += 1
        else:
            drawn += 1

        # Calculate metrics
        total_games = won + lost + drawn
        win_rate = won / total_games if total_games > 0 else 0.0
        avg_score = total_score / times_seen if times_seen > 0 else 0.0
        confidence = min(1.0, times_seen / 20.0)
        
        normalized_score = (avg_score + 1500) / 31
        priority_score = normalized_score * confidence

        # Enhanced insert/update
        self.cursor.execute('''
            INSERT INTO learned_move_patterns
                (piece_type, move_category, distance_from_start, game_phase,
                 repetition_count, moves_since_progress, total_material_level,
                 times_seen, games_won, games_lost, games_drawn,
                 win_rate, total_score, avg_score, confidence, priority_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(piece_type, move_category, distance_from_start, game_phase,
                        repetition_count, moves_since_progress, total_material_level)
            DO UPDATE SET
                times_seen = ?,
                games_won = ?,
                games_lost = ?,
                games_drawn = ?,
                win_rate = ?,
                total_score = ?,
                avg_score = ?,
                confidence = ?,
                priority_score = ?,
                updated_at = datetime('now')
        ''', (characteristics['piece_type'], characteristics['move_category'],
              characteristics['distance_from_start'], characteristics['game_phase'],
              rep_count, moves_since, material_level,
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score,
              # ON CONFLICT DO UPDATE values
              times_seen, won, lost, drawn,
              win_rate, total_score, avg_score, confidence, priority_score))

        self.conn.commit()
    
    def _classify_move(self, board, move):
        """Classify move characteristics"""
        
        piece = board.piece_at(move.from_square)
        if not piece:
            return {}

        piece_type_map = {
            chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
            chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'
        }
        piece_type = piece_type_map.get(piece.piece_type, 'unknown')

        # Move category
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)

        if is_capture and is_check:
            move_category = 'capture_check'
        elif is_capture:
            move_category = 'capture'
        elif is_check:
            move_category = 'check'
        else:
            move_category = 'development' if self._is_development_move(board, move, piece) else 'quiet'

        # Game phase
        move_count = board.fullmove_number
        if move_count <= 15:
            game_phase = 'opening'
        elif move_count >= 40:
            game_phase = 'endgame'
        else:
            game_phase = 'middlegame'

        # Distance from start
        distance_from_start = self._estimate_distance_from_start(piece, move.from_square)

        result = {
            'piece_type': piece_type,
            'move_category': move_category,
            'distance_from_start': distance_from_start,
            'game_phase': game_phase,
        }
        
        # Add enhanced characteristics if using enhanced schema
        if self.schema_type == "enhanced":
            result.update({
                'repetition_count': self._count_repetitions(board),
                'moves_since_progress': self._estimate_moves_since_progress(board, move),
                'total_material_level': self._assess_material_level(board)
            })

        return result
    
    def _is_development_move(self, board, move, piece):
        """Check if this is a development move"""
        if piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
            return False
        
        from_rank = chess.square_rank(move.from_square)
        back_rank = 0 if piece.color == chess.WHITE else 7
        
        return from_rank == back_rank

    def _estimate_distance_from_start(self, piece, square):
        """Estimate distance from starting position"""
        rank = chess.square_rank(square)
        
        if piece.piece_type == chess.PAWN:
            return 2 if (piece.color == chess.WHITE and rank >= 4) or \
                        (piece.color == chess.BLACK and rank <= 3) else 1
        elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.WHITE else 7
            return 2 if rank != back_rank else 1
        else:
            return 1

    def _count_repetitions(self, board):
        """Count position repetitions (simplified)"""
        return 0

    def _estimate_moves_since_progress(self, board, move):
        """Estimate moves since last progress"""
        return 25 if not board.is_capture(move) else 0

    def _assess_material_level(self, board):
        """Assess total material on board"""
        total_pieces = len(board.piece_map())
        if total_pieces > 20:
            return 'high'
        elif total_pieces > 10:
            return 'medium'
        else:
            return 'low'
    
    def get_statistics(self):
        """Get learning statistics"""
        self.cursor.execute('''
            SELECT COUNT(*), AVG(confidence), AVG(win_rate)
            FROM learned_move_patterns
            WHERE times_seen >= 5
        ''')
        
        row = self.cursor.fetchone()
        if row and row[0]:
            return {
                'patterns_learned': row[0],
                'avg_confidence': row[1] or 0.0,
                'avg_win_rate': row[2] or 0.0
            }
        
        return {
            'patterns_learned': 0,
            'avg_confidence': 0.0,
            'avg_win_rate': 0.0
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class SchemaCompatibleHeadlessTrainer:
    """
    Headless trainer that works with your actual database schema
    """

    def __init__(self, db_path="headless_training.db"):
        self.scorer = GameScorer()
        self.prioritizer = SchemaCompatibleMovePrioritizer(db_path)

        # Game statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []

        # Draw type analysis
        self.draw_types = {
            'stalemate': 0,
            'insufficient_material': 0,
            'fifty_moves': 0,
            'repetition': 0,
            'avoidable': 0,
            'unavoidable': 0
        }

    def play_ai_move(self, board, ai_color):
        """AI move selection"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        legal_moves = self.prioritizer.sort_moves_by_priority(board, legal_moves)

        best_move = None
        best_score = -999999

        for move in legal_moves[:15]:
            # Material evaluation
            board.push(move)
            ai_mat = self.scorer._calculate_material(board, ai_color)
            opp_mat = self.scorer._calculate_material(board, not ai_color)
            material_score = ai_mat - opp_mat
            board.pop()

            # Pattern priority
            priority = self.prioritizer.get_move_priority(board, move)
            score = material_score + (priority * 20)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def play_opponent_move(self, board):
        """Random opponent move"""
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]

        if captures and random.random() < 0.3:
            return random.choice(captures)

        return random.choice(legal_moves)

    def play_game(self, ai_color, verbose=False):
        """Play a single game"""
        board = chess.Board()
        game_moves = []
        move_count = 0

        while not board.is_game_over():
            fen_before = board.fen()

            if board.turn == ai_color:
                move = self.play_ai_move(board, ai_color)
                if move:
                    move_san = board.san(move)
                    board.push(move)
                    game_moves.append((fen_before, move.uci(), move_san))

                    if verbose:
                        ai_mat = self.scorer._calculate_material(board, ai_color)
                        opp_mat = self.scorer._calculate_material(board, not ai_color)
                        advantage = ai_mat - opp_mat
                        priority = self.prioritizer.get_move_priority(board, move)
                        print(f"  AI: {move_san:8s} | Advantage: {advantage:+6.0f} | Priority: {priority:5.1f}")
            else:
                move = self.play_opponent_move(board)
                if move:
                    move_san = board.san(move)
                    board.push(move)
                    move_count += 1
                    if verbose:
                        print(f"  Opp: {move_san:8s}")

        # Calculate final result
        rounds_played = board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)

        # Update statistics
        self._update_statistics(result, final_score, board)

        # Record for learning
        self.prioritizer.record_game_moves(game_moves, ai_color, result, final_score)

        return result, final_score, rounds_played

    def _update_statistics(self, result, final_score, board):
        """Update game statistics"""
        
        # Draw analysis
        if result == 'draw':
            ai_mat = self.scorer._calculate_material(board, True)
            opp_mat = self.scorer._calculate_material(board, False)
            material_advantage = ai_mat - opp_mat

            if board.is_stalemate():
                self.draw_types['stalemate'] += 1
                if material_advantage > 10:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1
            elif board.is_insufficient_material():
                self.draw_types['insufficient_material'] += 1
                self.draw_types['unavoidable'] += 1
            elif board.is_fifty_moves():
                self.draw_types['fifty_moves'] += 1
                if material_advantage > 20:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1
            elif board.is_repetition():
                self.draw_types['repetition'] += 1
                if material_advantage > -20:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1

        # Update counters
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        self.score_history.append(final_score)

    def train(self, num_games, verbose=False, progress_interval=10):
        """Train the AI"""
        
        print("=" * 70)
        print(f"SCHEMA-COMPATIBLE PATTERN LEARNING TRAINING - {num_games} GAMES")
        print(f"Database: {self.prioritizer.db_path}")
        print(f"Schema: {self.prioritizer.schema_type}")
        print("=" * 70)

        start_time = time.time()

        for i in range(num_games):
            ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            if verbose or (i + 1) % progress_interval == 0:
                color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"
                print(f"\\n--- Game {i+1}/{num_games} (AI plays {color_str}) ---")

            result, score, rounds = self.play_game(ai_color, verbose=verbose)

            if (i + 1) % progress_interval == 0 or verbose:
                win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0
                avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
                print(f"  Result: {result.upper():6s} | Score: {score:+7.0f} | Rounds: {rounds:2d}")
                print(f"  Progress: {self.wins}W-{self.losses}L-{self.draws}D | Win Rate: {win_rate:.1f}% | Avg Score: {avg_score:+.0f}")

        elapsed = time.time() - start_time
        self._show_results(elapsed, num_games)

    def _show_results(self, elapsed, num_games):
        """Show training results"""
        
        print("\\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
        print(f"Results: {self.wins}W-{self.losses}L-{self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played*100):.1f}%")
        print(f"Average Score: {sum(self.score_history)/len(self.score_history):+.0f}")
        print(f"Best Score: {max(self.score_history):+.0f}")
        print(f"Worst Score: {min(self.score_history):+.0f}")

        # Draw analysis
        if self.draws > 0:
            print(f"\\nDraw Analysis ({self.draws} total draws):")
            print(f"  Stalemate:              {self.draw_types['stalemate']:3d} ({self.draw_types['stalemate']/self.draws*100:5.1f}%)")
            print(f"  Insufficient Material:  {self.draw_types['insufficient_material']:3d} ({self.draw_types['insufficient_material']/self.draws*100:5.1f}%)")
            print(f"  Fifty-Move Rule:        {self.draw_types['fifty_moves']:3d} ({self.draw_types['fifty_moves']/self.draws*100:5.1f}%)")
            print(f"  Threefold Repetition:   {self.draw_types['repetition']:3d} ({self.draw_types['repetition']/self.draws*100:5.1f}%)")
            print(f"  ---")
            print(f"  AVOIDABLE (AI's fault): {self.draw_types['avoidable']:3d} ({self.draw_types['avoidable']/self.draws*100:5.1f}%)")
            print(f"  Unavoidable:            {self.draw_types['unavoidable']:3d} ({self.draw_types['unavoidable']/self.draws*100:5.1f}%)")

        # Pattern learning statistics
        stats = self.prioritizer.get_statistics()
        print(f"\\nLearning Statistics:")
        print(f"  Patterns Learned: {stats['patterns_learned']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg Win Rate: {stats['avg_win_rate']:.1%}")

        print("\\n" + "=" * 70)

    def close(self):
        """Close resources"""
        self.prioritizer.close()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Schema-Compatible Chess AI Trainer')
    parser.add_argument('num_games', type=int, help='Number of games to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=10, help='Progress update interval')
    parser.add_argument('--db', default='headless_training.db', help='Database file to use')

    args = parser.parse_args()

    trainer = SchemaCompatibleHeadlessTrainer(db_path=args.db)

    try:
        trainer.train(args.num_games, verbose=args.verbose, progress_interval=args.progress)

    finally:
        trainer.close()

if __name__ == '__main__':
    main()
