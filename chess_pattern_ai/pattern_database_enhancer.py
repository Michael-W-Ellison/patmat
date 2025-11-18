#!/usr/bin/env python3
"""
Pattern Database Enhancer - Continuous Learning Database

CRITICAL: The AI must enhance its core database after every game!

This implements the missing feedback loop:
1. AI plays game
2. Tracks mistakes and successes
3. STORES learned patterns back to database
4. REFERENCES these patterns in future games
5. Patterns accumulate over time = continuous improvement

Without this, the AI forgets everything after each game!
"""

import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime
from position_abstractor import PositionAbstractor

logger = logging.getLogger(__name__)


class PatternDatabaseEnhancer:
    """Enhances the core pattern database with learned experiences"""

    def __init__(self, db_path: str = "rule_discovery.db", verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose  # Control logging verbosity during search
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Initialize position abstractor for generalized pattern matching
        self.abstractor = PositionAbstractor()
        if self.verbose:
            logger.info("Position abstractor initialized for generalized pattern matching")

        self._create_learning_tables()

    def _create_learning_tables(self):
        """Create tables for storing learned patterns from gameplay"""

        # Table 0: Games with score-based tracking
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                white_player TEXT,
                black_player TEXT,
                result TEXT,  -- 'win', 'loss', 'draw'
                ai_color TEXT,  -- 'white' or 'black'
                played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- SCORE-BASED LEARNING (new!)
                rounds_played INTEGER,  -- Number of full rounds (move pairs)
                final_score REAL,  -- Calculated score: win=material+(100-rounds), loss=-100, draw=0
                ai_material REAL,  -- Final AI material count
                opponent_material REAL,  -- Final opponent material count

                -- Performance metrics
                moves_count INTEGER,
                game_duration_seconds REAL
            )
        ''')

        # Table 1: Mistakes to avoid
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_mistakes (
                mistake_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen_before TEXT,
                move_made TEXT,
                material_lost REAL,
                pieces_lost TEXT,
                opponent_response TEXT,
                game_id INTEGER,
                move_number INTEGER,
                times_seen INTEGER DEFAULT 1,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- NEW: Abstracted pattern for generalized matching
                position_pattern TEXT,
                move_pattern TEXT,

                -- Index for fast lookup during move evaluation
                UNIQUE(fen_before, move_made)
            )
        ''')

        # MIGRATION: Add pattern columns to existing table if they don't exist
        # Must run BEFORE index creation
        self._migrate_pattern_columns()

        # Create index on abstracted patterns for fast lookups
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_mistake_patterns
            ON learned_mistakes(position_pattern, move_pattern)
        ''')

        # Table 2: Successful tactics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_tactics (
                tactic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen_before TEXT,
                move_sequence TEXT,
                material_gained REAL,
                pieces_captured TEXT,
                tactic_type TEXT,  -- 'fork', 'pin', 'skewer', 'discovered_attack', 'other'
                who_played TEXT,   -- 'ai' or 'opponent'
                game_id INTEGER,
                move_number INTEGER,
                times_seen INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(fen_before, move_sequence)
            )
        ''')

        # Table 3: Position evaluations (correct the evaluation function)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_corrections (
                correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT,
                predicted_score REAL,
                actual_outcome REAL,  -- Material change that actually happened
                error_magnitude REAL,
                evaluator_at_fault TEXT,  -- Which evaluator gave wrong signal
                game_id INTEGER,
                move_number INTEGER,
                learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(fen, game_id)
            )
        ''')

        # Table 4: Opponent patterns (what beats us)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS opponent_winning_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen_before TEXT,
                opponent_move TEXT,
                material_gained_by_opponent REAL,
                our_weakness_exploited TEXT,
                game_id INTEGER,
                move_number INTEGER,
                times_seen INTEGER DEFAULT 1,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(fen_before, opponent_move)
            )
        ''')

        # Table 5: Learning session statistics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                games_played INTEGER,
                wins INTEGER,
                losses INTEGER,
                draws INTEGER,
                mistakes_learned INTEGER,
                tactics_learned INTEGER,
                patterns_from_opponent INTEGER,
                database_patterns_analyzed INTEGER,
                weight_adjustments TEXT  -- JSON of weight changes
            )
        ''')

        self.conn.commit()
        logger.info("✅ Learning database tables initialized")

    def _migrate_pattern_columns(self):
        """
        Migration: Add position_pattern and move_pattern columns if they don't exist.
        Backfill existing mistakes with abstracted patterns.
        """
        try:
            # Check if columns exist
            self.cursor.execute("PRAGMA table_info(learned_mistakes)")
            columns = [row[1] for row in self.cursor.fetchall()]

            needs_migration = False

            # Add columns if missing
            if 'position_pattern' not in columns:
                logger.info("⚙️  Adding position_pattern column to learned_mistakes...")
                self.cursor.execute("ALTER TABLE learned_mistakes ADD COLUMN position_pattern TEXT")
                needs_migration = True

            if 'move_pattern' not in columns:
                logger.info("⚙️  Adding move_pattern column to learned_mistakes...")
                self.cursor.execute("ALTER TABLE learned_mistakes ADD COLUMN move_pattern TEXT")
                needs_migration = True

            if needs_migration:
                self.conn.commit()

                # Backfill existing mistakes with patterns
                logger.info("⚙️  Backfilling abstracted patterns for existing mistakes...")
                self.cursor.execute('''
                    SELECT mistake_id, fen_before, move_made
                    FROM learned_mistakes
                    WHERE position_pattern IS NULL OR move_pattern IS NULL
                ''')

                mistakes_to_update = self.cursor.fetchall()
                updated_count = 0

                for mistake_id, fen, move in mistakes_to_update:
                    try:
                        position_pattern = self.abstractor.abstract_position(fen)
                        move_pattern = self.abstractor.abstract_move(fen, move)

                        self.cursor.execute('''
                            UPDATE learned_mistakes
                            SET position_pattern = ?, move_pattern = ?
                            WHERE mistake_id = ?
                        ''', (position_pattern, move_pattern, mistake_id))

                        updated_count += 1
                    except Exception as e:
                        logger.warning(f"Could not abstract mistake {mistake_id}: {e}")
                        continue

                self.conn.commit()
                logger.info(f"✅ Backfilled {updated_count} mistakes with abstracted patterns")
            else:
                logger.debug("Pattern columns already exist, skipping migration")

        except Exception as e:
            logger.error(f"Migration error: {e}")
            # Don't fail initialization if migration fails
            pass

    def store_mistake(self, mistake_data: Dict) -> bool:
        """
        Store a mistake in the database to avoid repeating it.

        Args:
            mistake_data: Dictionary with mistake information

        Returns:
            True if stored, False if duplicate
        """
        try:
            # Compute abstracted patterns for generalized matching
            fen = mistake_data['fen_before']
            move = mistake_data['move_made']

            position_pattern = self.abstractor.abstract_position(fen)
            move_pattern = self.abstractor.abstract_move(fen, move)

            logger.debug(f"Storing mistake with patterns: pos={position_pattern[:50]}..., move={move_pattern}")

            self.cursor.execute('''
                INSERT INTO learned_mistakes
                (fen_before, move_made, material_lost, pieces_lost, opponent_response,
                 game_id, move_number, position_pattern, move_pattern)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fen_before, move_made) DO UPDATE SET
                    times_seen = times_seen + 1,
                    last_seen = CURRENT_TIMESTAMP,
                    material_lost = (material_lost * times_seen + excluded.material_lost) / (times_seen + 1),
                    position_pattern = excluded.position_pattern,
                    move_pattern = excluded.move_pattern
            ''', (
                mistake_data['fen_before'],
                mistake_data['move'],
                mistake_data['material_lost'],
                ','.join(mistake_data.get('pieces_lost', [])),
                mistake_data.get('opponent_response', ''),
                mistake_data.get('game_id', 0),
                mistake_data.get('move_number', 0),
                position_pattern,
                move_pattern
            ))

            self.conn.commit()

            # Check if this was a new mistake or seen before
            if self.cursor.rowcount > 0:
                logger.info(f"📝 Stored mistake: {mistake_data['move']} loses {mistake_data['material_lost']:.1f} material")
                return True
            else:
                logger.warning(f"⚠️  REPEATED MISTAKE: {mistake_data['move']} (seen before!)")
                return False

        except Exception as e:
            logger.error(f"Error storing mistake: {e}")
            return False

    def store_successful_tactic(self, tactic_data: Dict) -> bool:
        """Store a successful tactic for future use"""
        try:
            self.cursor.execute('''
                INSERT INTO learned_tactics
                (fen_before, move_sequence, material_gained, pieces_captured,
                 tactic_type, who_played, game_id, move_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fen_before, move_sequence) DO UPDATE SET
                    times_seen = times_seen + 1,
                    last_seen = CURRENT_TIMESTAMP,
                    success_rate = (success_rate * times_seen + 1.0) / (times_seen + 1),
                    material_gained = (material_gained * times_seen + excluded.material_gained) / (times_seen + 1)
            ''', (
                tactic_data['fen_before'],
                tactic_data['move_sequence'],
                tactic_data['material_gained'],
                ','.join(tactic_data.get('pieces_captured', [])),
                tactic_data.get('tactic_type', 'other'),
                tactic_data.get('who_played', 'ai'),
                tactic_data.get('game_id', 0),
                tactic_data.get('move_number', 0)
            ))

            self.conn.commit()
            logger.info(f"✅ Stored successful tactic: {tactic_data['move_sequence']} "
                       f"gained {tactic_data['material_gained']:.1f} material")
            return True

        except Exception as e:
            logger.error(f"Error storing tactic: {e}")
            return False

    def store_evaluation_correction(self, correction_data: Dict):
        """Store evaluation error to calibrate future evaluations"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO evaluation_corrections
                (fen, predicted_score, actual_outcome, error_magnitude,
                 evaluator_at_fault, game_id, move_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                correction_data['fen'],
                correction_data['predicted_score'],
                correction_data['actual_outcome'],
                correction_data['error_magnitude'],
                correction_data.get('evaluator_at_fault', 'unknown'),
                correction_data.get('game_id', 0),
                correction_data.get('move_number', 0)
            ))

            self.conn.commit()
            logger.debug(f"Stored evaluation correction for position")

        except Exception as e:
            logger.error(f"Error storing evaluation correction: {e}")

    def store_opponent_pattern(self, pattern_data: Dict):
        """Store opponent's successful pattern (what beats us)"""
        try:
            self.cursor.execute('''
                INSERT INTO opponent_winning_patterns
                (fen_before, opponent_move, material_gained_by_opponent,
                 our_weakness_exploited, game_id, move_number)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fen_before, opponent_move) DO UPDATE SET
                    times_seen = times_seen + 1,
                    last_seen = CURRENT_TIMESTAMP
            ''', (
                pattern_data['fen_before'],
                pattern_data['opponent_move'],
                pattern_data['material_gained'],
                pattern_data.get('weakness_exploited', 'unknown'),
                pattern_data.get('game_id', 0),
                pattern_data.get('move_number', 0)
            ))

            self.conn.commit()
            logger.info(f"📚 Learned from opponent: {pattern_data['opponent_move']} "
                       f"exploits our weakness")

        except Exception as e:
            logger.error(f"Error storing opponent pattern: {e}")

    def check_for_known_mistake(self, fen: str, move: str) -> Optional[Dict]:
        """
        Check if a move is a known mistake before making it!

        Uses TWO-TIER matching:
        1. Exact FEN match (precise, rare)
        2. Abstracted pattern match (generalized, common)

        Args:
            fen: Current position
            move: Proposed move in UCI format

        Returns:
            Mistake data if this is a known mistake, None otherwise
        """
        # TIER 1: Try exact FEN match first (most precise)
        self.cursor.execute('''
            SELECT material_lost, times_seen, pieces_lost
            FROM learned_mistakes
            WHERE fen_before = ? AND move_made = ?
        ''', (fen, move))

        result = self.cursor.fetchone()

        if result:
            material_lost, times_seen, pieces_lost = result
            logger.warning(f"⚠️  EXACT MATCH: Known mistake detected!")
            logger.warning(f"   Move {move} has lost {material_lost:.1f} material {times_seen}x before")
            logger.warning(f"   🚫 AVOID THIS MOVE!")

            return {
                'material_lost': material_lost,
                'times_seen': times_seen,
                'pieces_lost': pieces_lost.split(',') if pieces_lost else []
            }

        # TIER 2: Try abstracted pattern match (generalized)
        position_pattern = self.abstractor.abstract_position(fen)
        move_pattern = self.abstractor.abstract_move(fen, move)

        self.cursor.execute('''
            SELECT AVG(material_lost), SUM(times_seen), COUNT(*)
            FROM learned_mistakes
            WHERE position_pattern = ? AND move_pattern = ?
            AND material_lost > 2.0
        ''', (position_pattern, move_pattern))

        pattern_result = self.cursor.fetchone()

        if pattern_result and pattern_result[2] > 0:  # At least 1 similar mistake found
            avg_material, total_seen, match_count = pattern_result
            logger.warning(f"⚠️  PATTERN MATCH: Similar mistake detected!")
            logger.warning(f"   {match_count} similar positions lost avg {avg_material:.1f} material")
            logger.warning(f"   Seen {total_seen}x total across similar positions")
            logger.warning(f"   Pattern: {position_pattern}")
            logger.warning(f"   ⚠️  CAUTION: This move type is risky!")

            return {
                'material_lost': avg_material,
                'times_seen': int(total_seen),
                'pieces_lost': [],
                'is_pattern_match': True,
                'match_count': match_count
            }

        return None

    def get_known_tactics_for_position(self, fen: str) -> List[Dict]:
        """Get successful tactics learned for similar positions"""
        self.cursor.execute('''
            SELECT move_sequence, material_gained, tactic_type, success_rate, times_seen
            FROM learned_tactics
            WHERE fen_before = ?
            ORDER BY success_rate * material_gained DESC
            LIMIT 5
        ''', (fen,))

        tactics = []
        for row in self.cursor.fetchall():
            tactics.append({
                'move_sequence': row[0],
                'material_gained': row[1],
                'tactic_type': row[2],
                'success_rate': row[3],
                'times_seen': row[4]
            })

        # Only log if verbose mode enabled (prevents log spam during minimax search)
        if tactics and self.verbose:
            logger.info(f"💡 Found {len(tactics)} known successful tactics for this position")
            for t in tactics:
                logger.info(f"   {t['move_sequence']}: +{t['material_gained']:.1f} "
                           f"({t['success_rate']:.0%} success, seen {t['times_seen']}x)")

        return tactics

    def enhance_from_game_analysis(self, game_analysis, move_pairs: List) -> Dict:
        """
        Enhance database with all learnings from a game.

        This is called after every game to store learned patterns.
        """
        stats = {
            'mistakes_stored': 0,
            'tactics_stored': 0,
            'corrections_stored': 0,
            'opponent_patterns_stored': 0
        }

        logger.info("\n" + "="*80)
        logger.info("ENHANCING PATTERN DATABASE FROM GAME")
        logger.info("="*80)

        # Store all critical mistakes
        for mistake in game_analysis.critical_mistakes:
            mistake_data = {
                'fen_before': mistake.fen_before,
                'move': mistake.move_uci,
                'material_lost': -mistake.material_change,
                'pieces_lost': mistake.pieces_lost_by_us,
                'opponent_response': '',  # Would need move_pairs to get this
                'game_id': game_analysis.game_id,
                'move_number': mistake.move_number
            }

            if self.store_mistake(mistake_data):
                stats['mistakes_stored'] += 1

        # Store all successful tactics
        for tactic in game_analysis.tactical_wins:
            tactic_data = {
                'fen_before': tactic.fen_before,
                'move_sequence': tactic.move_uci,
                'material_gained': tactic.material_change,
                'pieces_captured': tactic.pieces_captured_by_us,
                'tactic_type': self._classify_tactic(tactic),
                'who_played': 'ai',
                'game_id': game_analysis.game_id,
                'move_number': tactic.move_number
            }

            if self.store_successful_tactic(tactic_data):
                stats['tactics_stored'] += 1

        # Store evaluation corrections
        for error in game_analysis.evaluation_errors:
            correction_data = {
                'fen': error['fen_before'],
                'predicted_score': error['predicted'],
                'actual_outcome': error['actual'],
                'error_magnitude': error['error_magnitude'],
                'evaluator_at_fault': 'unknown',  # Would need to analyze which evaluator
                'game_id': game_analysis.game_id,
                'move_number': error['move_number']
            }

            self.store_evaluation_correction(correction_data)
            stats['corrections_stored'] += 1

        # If AI lost, store opponent's successful patterns
        if not game_analysis.ai_won:
            # Extract opponent's material-gaining moves
            import chess
            board = chess.Board()
            opponent_color = 'black' if game_analysis.ai_color == 'white' else 'white'

            for ai_move, opp_move in move_pairs:
                try:
                    # AI moves
                    board.push(chess.Move.from_uci(ai_move))
                    fen_before_opp = board.fen()

                    # Opponent moves
                    board.push(chess.Move.from_uci(opp_move))

                    # Check if opponent gained material (this beat us!)
                    # Simplified - would need full material tracking
                    # For now, just store the pattern

                    pattern_data = {
                        'fen_before': fen_before_opp,
                        'opponent_move': opp_move,
                        'material_gained': 0.0,  # Would calculate
                        'weakness_exploited': 'unknown',
                        'game_id': game_analysis.game_id,
                        'move_number': len(move_pairs)
                    }

                    self.store_opponent_pattern(pattern_data)
                    stats['opponent_patterns_stored'] += 1

                except:
                    continue

        logger.info(f"\nDatabase enhancement complete:")
        logger.info(f"  Mistakes stored: {stats['mistakes_stored']}")
        logger.info(f"  Tactics stored: {stats['tactics_stored']}")
        logger.info(f"  Evaluation corrections: {stats['corrections_stored']}")
        logger.info(f"  Opponent patterns: {stats['opponent_patterns_stored']}")
        logger.info("="*80 + "\n")

        return stats

    def _classify_tactic(self, tactic) -> str:
        """Classify what type of tactic this was"""
        # Simplified classification based on pieces captured
        if len(tactic.pieces_captured_by_us) == 0:
            return 'positional'
        elif len(tactic.pieces_captured_by_us) == 1:
            return 'simple_capture'
        else:
            return 'multi_piece_tactic'

    def get_database_statistics(self) -> Dict:
        """Get statistics on what's been learned"""
        stats = {}

        # Total mistakes learned
        self.cursor.execute('SELECT COUNT(*), SUM(times_seen) FROM learned_mistakes')
        unique_mistakes, total_mistakes = self.cursor.fetchone()
        stats['unique_mistakes'] = unique_mistakes
        stats['total_mistakes'] = total_mistakes or 0

        # Total tactics learned
        self.cursor.execute('SELECT COUNT(*), SUM(times_seen) FROM learned_tactics')
        unique_tactics, total_tactics = self.cursor.fetchone()
        stats['unique_tactics'] = unique_tactics
        stats['total_tactics'] = total_tactics or 0

        # Evaluation corrections
        self.cursor.execute('SELECT COUNT(*) FROM evaluation_corrections')
        stats['eval_corrections'] = self.cursor.fetchone()[0]

        # Opponent patterns
        self.cursor.execute('SELECT COUNT(*), SUM(times_seen) FROM opponent_winning_patterns')
        unique_opp, total_opp = self.cursor.fetchone()
        stats['unique_opponent_patterns'] = unique_opp
        stats['total_opponent_patterns'] = total_opp or 0

        return stats

    def print_statistics(self, force: bool = False):
        """
        Print learning database statistics.

        Args:
            force: If True, print even if not verbose. If False, only print in verbose mode.
        """
        if not force and not self.verbose:
            return  # Skip statistics printing during silent operation

        stats = self.get_database_statistics()

        logger.info("\n" + "="*80)
        logger.info("LEARNED PATTERN DATABASE STATISTICS")
        logger.info("="*80)
        logger.info(f"Mistakes to avoid: {stats['unique_mistakes']} unique "
                   f"({stats['total_mistakes']} total occurrences)")
        logger.info(f"Successful tactics: {stats['unique_tactics']} unique "
                   f"({stats['total_tactics']} total occurrences)")
        logger.info(f"Evaluation corrections: {stats['eval_corrections']}")
        logger.info(f"Opponent patterns: {stats['unique_opponent_patterns']} unique "
                   f"({stats['total_opponent_patterns']} total occurrences)")
        logger.info("="*80 + "\n")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
