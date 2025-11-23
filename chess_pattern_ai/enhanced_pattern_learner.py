#!/usr/bin/env python3
"""
Enhanced Contextual Pattern Learning System

This system learns from pattern failures by analyzing:
1. WHY patterns failed (missing preconditions, violated constraints)
2. WHEN patterns work vs when they don't (context-dependent success)
3. META-PATTERNS that span multiple moves
4. CAUSAL relationships between actions and consequences

Key Innovation: Instead of just counting wins/losses, this system analyzes
the CONDITIONS that make patterns successful or dangerous.
"""

import chess
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class PatternOutcome(Enum):
    SUCCESS = "success"
    FAILED_TACTICAL = "failed_tactical"  # Lost material to tactics
    FAILED_POSITIONAL = "failed_positional"  # Poor position
    FAILED_TIMING = "failed_timing"  # Right idea, wrong time
    INCOMPLETE = "incomplete"  # Pattern started but never finished

@dataclass
class ContextualFactors:
    """Rich context that determines when patterns succeed vs fail"""
    
    # Piece safety context
    piece_defended: bool
    piece_attackers: int  # How many enemy pieces attack this square
    piece_defenders: int  # How many friendly pieces defend this square
    
    # Material context  
    material_balance: int  # Our material - their material
    material_trend: str  # 'gaining', 'losing', 'stable'
    
    # Positional context
    king_safety: str  # 'safe', 'exposed', 'critical' 
    center_control: str  # 'dominant', 'equal', 'weak'
    development_phase: str  # 'underdeveloped', 'developing', 'developed'
    
    # Tactical context
    forcing_move: bool  # Check, capture, or threat
    tactical_motifs: List[str]  # ['pin', 'fork', 'skewer', 'discovery']
    escape_squares: int  # Available squares if attacked
    
    # Game flow context
    recent_blunders: int  # Mistakes in last 3 moves
    pressure_level: str  # 'attacking', 'defending', 'neutral'
    time_pressure: bool  # Simulated time pressure
    
    # Pattern sequence context
    pattern_chain_length: int  # Part of multi-move pattern?
    incomplete_patterns: List[str]  # Started patterns not yet finished

@dataclass 
class PatternResult:
    """Tracks what happened after applying a pattern"""
    
    immediate_material_change: int  # Material gained/lost next move
    position_evaluation_change: int  # Position got better/worse
    tactical_consequences: List[str]  # ['lost_piece', 'gained_tempo', etc.]
    pattern_completion_status: str  # 'completed', 'abandoned', 'interrupted'
    
    # Learning insights
    failure_cause: Optional[str]  # Why did this pattern fail?
    success_factors: List[str]  # What made this pattern work?
    
class EnhancedPatternLearner:
    """
    Advanced pattern learning that understands causality and context
    """
    
    def __init__(self, db_path='enhanced_patterns.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_enhanced_database()
        
        # Pattern failure analysis
        self.failure_patterns = {}
        self.success_conditions = {}
        self.meta_patterns = {}
        
    def _init_enhanced_database(self):
        """Create enhanced pattern database with contextual factors"""
        
        # Main contextual patterns table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS contextual_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Basic pattern identification
                piece_type TEXT NOT NULL,
                move_category TEXT NOT NULL,
                game_phase TEXT NOT NULL,
                
                -- Rich contextual factors
                piece_defended BOOLEAN,
                piece_attackers INTEGER,
                piece_defenders INTEGER,
                material_balance INTEGER,
                material_trend TEXT,
                king_safety TEXT,
                center_control TEXT,
                development_phase TEXT,
                forcing_move BOOLEAN,
                tactical_motifs TEXT,  -- JSON list
                escape_squares INTEGER,
                pressure_level TEXT,
                pattern_chain_length INTEGER,
                
                -- Outcome tracking
                times_seen INTEGER DEFAULT 0,
                times_succeeded INTEGER DEFAULT 0,
                times_failed_tactical INTEGER DEFAULT 0,
                times_failed_positional INTEGER DEFAULT 0,
                times_failed_timing INTEGER DEFAULT 0,
                times_incomplete INTEGER DEFAULT 0,
                
                -- Success metrics
                avg_material_gain REAL DEFAULT 0.0,
                avg_position_improvement REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                context_specificity REAL DEFAULT 0.0,  -- How context-dependent is success?
                
                -- Learning insights
                failure_causes TEXT,  -- JSON list of why patterns fail
                success_conditions TEXT,  -- JSON list of success requirements
                
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Unique constraint on full context
                UNIQUE(piece_type, move_category, game_phase, piece_defended, 
                       material_balance, king_safety, forcing_move, pressure_level)
            )
        ''')
        
        # Pattern failure analysis table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                failure_type TEXT,
                failure_cause TEXT,
                preceding_context TEXT,  -- What led to this failure?
                lesson_learned TEXT,     -- What should AI avoid next time?
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(pattern_id) REFERENCES contextual_patterns(id)
            )
        ''')
        
        # Meta-pattern sequences table  
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_name TEXT,
                pattern_sequence TEXT,  -- JSON list of pattern IDs
                success_rate REAL,
                avg_length INTEGER,
                completion_requirements TEXT,  -- JSON list
                common_failures TEXT,  -- JSON list
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pattern insights and rules
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_type TEXT,  -- 'constraint', 'precondition', 'consequence'
                rule_description TEXT,
                confidence REAL,
                supporting_evidence INTEGER,  -- How many observations support this?
                counterexamples INTEGER,      -- How many contradict it?
                pattern_scope TEXT,  -- Which patterns does this rule apply to?
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def analyze_context(self, board: chess.Board, move: chess.Move, 
                       game_history: List[str]) -> ContextualFactors:
        """Extract rich contextual factors from position"""
        
        # Piece safety analysis
        to_square = move.to_square
        piece_defended = self._is_square_defended(board, to_square, board.turn)
        piece_attackers = len(board.attackers(not board.turn, to_square))
        piece_defenders = len(board.attackers(board.turn, to_square))
        
        # Material context
        material_balance = self._calculate_material_balance(board)
        material_trend = self._analyze_material_trend(game_history)
        
        # King safety
        king_safety = self._evaluate_king_safety(board)
        
        # Center control
        center_control = self._evaluate_center_control(board)
        
        # Development
        development_phase = self._evaluate_development(board)
        
        # Tactical context
        forcing_move = board.is_check() or board.is_capture(move)
        tactical_motifs = self._detect_tactical_motifs(board, move)
        escape_squares = len(list(board.legal_moves)) if board.turn else 0
        
        # Game flow
        recent_blunders = self._count_recent_blunders(game_history)
        pressure_level = self._evaluate_pressure(board)
        
        # Pattern chains
        pattern_chain_length = self._analyze_pattern_chain(game_history)
        incomplete_patterns = self._find_incomplete_patterns(board, game_history)
        
        return ContextualFactors(
            piece_defended=piece_defended,
            piece_attackers=piece_attackers, 
            piece_defenders=piece_defenders,
            material_balance=material_balance,
            material_trend=material_trend,
            king_safety=king_safety,
            center_control=center_control,
            development_phase=development_phase,
            forcing_move=forcing_move,
            tactical_motifs=tactical_motifs,
            escape_squares=escape_squares,
            recent_blunders=recent_blunders,
            pressure_level=pressure_level,
            time_pressure=False,  # Placeholder
            pattern_chain_length=pattern_chain_length,
            incomplete_patterns=incomplete_patterns
        )
    
    def learn_from_pattern_outcome(self, pattern_context: Dict, 
                                  result: PatternResult, 
                                  game_outcome: str):
        """Learn why this specific pattern succeeded or failed"""
        
        # Determine outcome type
        outcome = self._classify_outcome(result, game_outcome)
        
        # Update pattern statistics
        pattern_id = self._update_contextual_pattern(pattern_context, outcome, result)
        
        # Analyze failure if pattern failed
        if outcome != PatternOutcome.SUCCESS:
            self._analyze_pattern_failure(pattern_id, pattern_context, result, outcome)
            
        # Extract rules and insights
        self._extract_learned_rules(pattern_context, result, outcome)
        
        # Update meta-patterns
        self._update_meta_patterns(pattern_context, result)
        
    def _classify_outcome(self, result: PatternResult, game_outcome: str) -> PatternOutcome:
        """Determine why pattern succeeded or failed"""
        
        if result.immediate_material_change < -50:  # Lost significant material
            return PatternOutcome.FAILED_TACTICAL
            
        if result.position_evaluation_change < -30:  # Position got much worse
            return PatternOutcome.FAILED_POSITIONAL
            
        if result.pattern_completion_status == 'abandoned':
            return PatternOutcome.INCOMPLETE
            
        if 'wrong_timing' in result.tactical_consequences:
            return PatternOutcome.FAILED_TIMING
            
        return PatternOutcome.SUCCESS
    
    def _analyze_pattern_failure(self, pattern_id: int, context: Dict, 
                                result: PatternResult, outcome: PatternOutcome):
        """Deep analysis of why this pattern failed"""
        
        # Identify specific failure cause
        failure_cause = self._diagnose_failure_cause(context, result, outcome)
        
        # Extract lesson for future
        lesson = self._extract_lesson(context, result, failure_cause)
        
        # Store failure analysis
        self.cursor.execute('''
            INSERT INTO pattern_failures 
            (pattern_id, failure_type, failure_cause, lesson_learned)
            VALUES (?, ?, ?, ?)
        ''', (pattern_id, outcome.value, failure_cause, lesson))
        
        # Update failure patterns database
        self._update_failure_patterns(context, failure_cause)
        
    def _diagnose_failure_cause(self, context: Dict, result: PatternResult, 
                               outcome: PatternOutcome) -> str:
        """Determine specific reason for pattern failure"""
        
        if outcome == PatternOutcome.FAILED_TACTICAL:
            if not context.get('piece_defended', False):
                return "moved_to_undefended_square"
            elif context.get('piece_attackers', 0) > context.get('piece_defenders', 0):
                return "insufficient_piece_defense" 
            elif 'fork' in result.tactical_consequences:
                return "fell_into_fork"
            elif 'pin' in result.tactical_consequences:
                return "fell_into_pin"
            else:
                return "unknown_tactical_failure"
                
        elif outcome == PatternOutcome.FAILED_POSITIONAL:
            if context.get('king_safety') == 'exposed':
                return "compromised_king_safety"
            elif context.get('development_phase') == 'underdeveloped':
                return "premature_attack"
            elif context.get('center_control') == 'weak':
                return "neglected_center"
            else:
                return "unknown_positional_failure"
                
        elif outcome == PatternOutcome.FAILED_TIMING:
            if context.get('material_balance', 0) < -200:
                return "attacked_while_behind"
            elif context.get('pressure_level') == 'defending':
                return "attacked_while_defending"
            else:
                return "poor_timing"
                
        return "unknown_failure"
    
    def _extract_lesson(self, context: Dict, result: PatternResult, 
                       failure_cause: str) -> str:
        """Generate specific lesson to avoid this failure"""
        
        lessons = {
            "moved_to_undefended_square": "Ensure piece is defended before moving to attacked square",
            "insufficient_piece_defense": "Count attackers vs defenders before moving",
            "fell_into_fork": "Check for knight fork patterns before moving",
            "fell_into_pin": "Verify piece can move without exposing valuable piece",
            "compromised_king_safety": "Secure king position before launching attack",
            "premature_attack": "Complete development before attacking",
            "neglected_center": "Control center before advancing on flanks",
            "attacked_while_behind": "Avoid risky attacks when behind in material",
            "attacked_while_defending": "Consolidate defense before counterattacking"
        }
        
        return lessons.get(failure_cause, "Analyze position more carefully")
    
    def _extract_learned_rules(self, context: Dict, result: PatternResult, 
                              outcome: PatternOutcome):
        """Extract generalizable rules from this experience"""
        
        if outcome == PatternOutcome.SUCCESS:
            # Learn success conditions
            if context.get('piece_defended') and result.immediate_material_change > 0:
                self._record_rule(
                    "precondition",
                    "Pieces should be defended when capturing",
                    context, result
                )
                
        elif outcome == PatternOutcome.FAILED_TACTICAL:
            # Learn danger patterns
            if not context.get('piece_defended'):
                self._record_rule(
                    "constraint", 
                    "Never move to undefended attacked square unless forcing",
                    context, result
                )
    
    def _record_rule(self, rule_type: str, description: str, 
                    context: Dict, result: PatternResult):
        """Record a learned rule with evidence"""
        
        # Check if rule already exists
        self.cursor.execute('''
            SELECT id, supporting_evidence, counterexamples 
            FROM learned_rules 
            WHERE rule_description = ?
        ''', (description,))
        
        existing = self.cursor.fetchone()
        
        if existing:
            # Update evidence
            rule_id, evidence, counter = existing
            new_evidence = evidence + 1 if result.immediate_material_change >= 0 else evidence
            new_counter = counter + 1 if result.immediate_material_change < 0 else counter
            confidence = new_evidence / (new_evidence + new_counter) if (new_evidence + new_counter) > 0 else 0
            
            self.cursor.execute('''
                UPDATE learned_rules 
                SET supporting_evidence = ?, counterexamples = ?, confidence = ?
                WHERE id = ?
            ''', (new_evidence, new_counter, confidence, rule_id))
        else:
            # Create new rule
            confidence = 1.0 if result.immediate_material_change >= 0 else 0.0
            self.cursor.execute('''
                INSERT INTO learned_rules 
                (rule_type, rule_description, confidence, supporting_evidence, counterexamples)
                VALUES (?, ?, ?, 1, 0)
            ''', (rule_type, description, confidence))
    
    def get_pattern_advice(self, board: chess.Board, move: chess.Move, 
                          game_history: List[str]) -> Dict:
        """Get contextual advice about whether to make this move"""
        
        context = self.analyze_context(board, move, game_history)
        
        # Check for learned danger patterns
        warnings = self._check_danger_patterns(context, move, board)
        
        # Check for success patterns  
        encouragements = self._check_success_patterns(context, move, board)
        
        # Check learned rules
        rule_violations = self._check_rule_violations(context, move, board)
        
        # Calculate confidence
        confidence = self._calculate_move_confidence(context, warnings, encouragements)
        
        return {
            'confidence': confidence,
            'warnings': warnings,
            'encouragements': encouragements, 
            'rule_violations': rule_violations,
            'context_analysis': asdict(context),
            'recommendation': 'proceed' if confidence > 0.6 else 'caution' if confidence > 0.3 else 'avoid'
        }
    
    def _check_danger_patterns(self, context: ContextualFactors, 
                              move: chess.Move, board: chess.Board) -> List[str]:
        """Check if this move matches known failure patterns"""
        
        warnings = []
        
        # Undefended piece movement
        if not context.piece_defended and context.piece_attackers > 0:
            warnings.append("Moving to attacked undefended square")
            
        # Material imbalance attacks
        if context.material_balance < -200 and context.forcing_move:
            warnings.append("Risky attack while behind in material")
            
        # King safety violations
        if context.king_safety == 'exposed' and move.piece_type != chess.KING:
            warnings.append("Attacking while king is exposed")
            
        # Development violations
        if context.development_phase == 'underdeveloped' and context.forcing_move:
            warnings.append("Premature attack before completing development")
            
        return warnings
    
    def _check_success_patterns(self, context: ContextualFactors,
                               move: chess.Move, board: chess.Board) -> List[str]:
        """Check if this move matches known success patterns"""
        
        encouragements = []
        
        # Well-supported captures
        if context.piece_defended and context.forcing_move:
            encouragements.append("Well-supported aggressive move")
            
        # Material advantage pressing
        if context.material_balance > 100 and context.forcing_move:
            encouragements.append("Pressing material advantage") 
            
        # Safe king with attack
        if context.king_safety == 'safe' and context.forcing_move:
            encouragements.append("Safe to attack with secure king")
            
        return encouragements
    
    def _check_rule_violations(self, context: ContextualFactors,
                              move: chess.Move, board: chess.Board) -> List[str]:
        """Check against learned rules"""
        
        violations = []
        
        # Query learned rules
        self.cursor.execute('''
            SELECT rule_description, confidence 
            FROM learned_rules 
            WHERE rule_type = 'constraint' AND confidence > 0.7
        ''')
        
        for rule_desc, confidence in self.cursor.fetchall():
            if self._rule_applies(rule_desc, context, move, board):
                violations.append(f"{rule_desc} (confidence: {confidence:.1%})")
                
        return violations
    
    def _rule_applies(self, rule_desc: str, context: ContextualFactors,
                     move: chess.Move, board: chess.Board) -> bool:
        """Check if a learned rule applies to current situation"""
        
        # Simple rule matching - could be made more sophisticated
        if "undefended attacked square" in rule_desc:
            return not context.piece_defended and context.piece_attackers > 0
            
        if "behind in material" in rule_desc:
            return context.material_balance < -100
            
        return False
    
    def _calculate_move_confidence(self, context: ContextualFactors,
                                  warnings: List[str], 
                                  encouragements: List[str]) -> float:
        """Calculate confidence in this move based on learned patterns"""
        
        base_confidence = 0.5
        
        # Reduce confidence for warnings
        confidence_penalty = len(warnings) * 0.2
        
        # Increase confidence for encouragements  
        confidence_bonus = len(encouragements) * 0.15
        
        # Context modifiers
        if context.piece_defended:
            confidence_bonus += 0.1
        if context.king_safety == 'safe':
            confidence_bonus += 0.1
        if context.material_balance > 0:
            confidence_bonus += 0.05
            
        final_confidence = max(0.0, min(1.0, 
            base_confidence - confidence_penalty + confidence_bonus))
        
        return final_confidence
    
    # Helper methods for context analysis
    def _is_square_defended(self, board: chess.Board, square: int, color: bool) -> bool:
        """Check if square is defended by pieces of given color"""
        return len(board.attackers(color, square)) > 0
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = we're ahead)"""
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
                       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        
        white_material = sum(piece_values[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type]
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        return white_material - black_material if board.turn == chess.WHITE else black_material - white_material
    
    def _analyze_material_trend(self, game_history: List[str]) -> str:
        """Analyze if we're gaining, losing, or stable in material"""
        # Simplified - would analyze recent captures
        return 'stable'
    
    def _evaluate_king_safety(self, board: chess.Board) -> str:
        """Evaluate king safety"""
        king_square = board.king(board.turn)
        if not king_square:
            return 'critical'
            
        attackers = len(board.attackers(not board.turn, king_square))
        defenders = len(board.attackers(board.turn, king_square))
        
        if attackers == 0:
            return 'safe'
        elif defenders >= attackers:
            return 'safe' 
        else:
            return 'exposed'
    
    def _evaluate_center_control(self, board: chess.Board) -> str:
        """Evaluate center control"""
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        our_control = sum(1 for sq in center_squares 
                         if board.attackers(board.turn, sq))
        their_control = sum(1 for sq in center_squares 
                           if board.attackers(not board.turn, sq))
        
        if our_control > their_control:
            return 'dominant'
        elif our_control == their_control:
            return 'equal'
        else:
            return 'weak'
    
    def _evaluate_development(self, board: chess.Board) -> str:
        """Evaluate development phase"""
        # Count developed pieces (simplified)
        developed = 0
        for square, piece in board.piece_map().items():
            if piece.color == board.turn and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if chess.square_rank(square) not in [0, 7]:  # Moved from back rank
                    developed += 1
        
        if developed >= 3:
            return 'developed'
        elif developed >= 1:
            return 'developing'
        else:
            return 'underdeveloped'
    
    def _detect_tactical_motifs(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Detect tactical patterns"""
        motifs = []
        
        # Simplified tactical detection
        if board.is_check():
            motifs.append('check')
        if board.is_capture(move):
            motifs.append('capture')
            
        return motifs
    
    def _count_recent_blunders(self, game_history: List[str]) -> int:
        """Count recent mistakes"""
        # Placeholder - would analyze recent moves for blunders
        return 0
    
    def _evaluate_pressure(self, board: chess.Board) -> str:
        """Evaluate who has pressure"""
        # Simplified pressure evaluation
        if board.is_check():
            return 'attacking' if board.turn else 'defending'
        return 'neutral'
    
    def _analyze_pattern_chain(self, game_history: List[str]) -> int:
        """Analyze if this is part of a multi-move pattern"""
        # Placeholder - would detect pattern sequences
        return 1
    
    def _find_incomplete_patterns(self, board: chess.Board, game_history: List[str]) -> List[str]:
        """Find patterns that were started but not completed"""
        # Placeholder - would track incomplete tactical/strategic patterns
        return []
    
    def _update_contextual_pattern(self, context: Dict, outcome: PatternOutcome, 
                                  result: PatternResult) -> int:
        """Update pattern database with contextual outcome"""
        
        # This would implement the full database update logic
        # Placeholder for now
        return 1
    
    def _update_failure_patterns(self, context: Dict, failure_cause: str):
        """Update database of known failure patterns"""
        pass
    
    def _update_meta_patterns(self, context: Dict, result: PatternResult):
        """Update multi-move pattern sequences"""
        pass
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Example usage and integration
class ContextAwareChessAI:
    """Chess AI that uses enhanced contextual pattern learning"""
    
    def __init__(self):
        self.pattern_learner = EnhancedPatternLearner()
        self.game_history = []
        
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate move using contextual pattern analysis"""
        
        advice = self.pattern_learner.get_pattern_advice(board, move, self.game_history)
        
        # Base score from traditional evaluation
        base_score = 0.0  # Would integrate with existing evaluation
        
        # Adjust based on pattern learning
        if advice['recommendation'] == 'avoid':
            base_score -= 200  # Strong negative modifier
        elif advice['recommendation'] == 'caution':
            base_score -= 50   # Mild negative modifier
        elif advice['recommendation'] == 'proceed':
            base_score += 25   # Small positive modifier
            
        # Additional adjustments for specific warnings/encouragements
        base_score -= len(advice['warnings']) * 30
        base_score += len(advice['encouragements']) * 20
        
        return base_score
    
    def learn_from_game(self, moves: List[Tuple], game_result: str):
        """Learn from complete game using enhanced analysis"""
        
        board = chess.Board()
        
        for i, (fen, move_uci, move_san) in enumerate(moves):
            board.set_fen(fen)
            move = chess.Move.from_uci(move_uci)
            
            # Analyze context
            context = self.pattern_learner.analyze_context(board, move, self.game_history)
            
            # Apply move and analyze result
            board.push(move)
            
            # Calculate immediate consequences
            material_change = 0  # Would calculate actual material change
            position_change = 0  # Would evaluate position change
            
            result = PatternResult(
                immediate_material_change=material_change,
                position_evaluation_change=position_change,
                tactical_consequences=[],
                pattern_completion_status='completed',
                failure_cause=None,
                success_factors=[]
            )
            
            # Learn from this pattern application
            pattern_context = {
                'piece_type': board.piece_at(move.from_square).piece_type,
                'move_category': 'capture' if board.is_capture(move) else 'quiet',
                'game_phase': 'opening' if len(moves) < 15 else 'endgame' if len(moves) > 40 else 'middlegame'
            }
            
            self.pattern_learner.learn_from_pattern_outcome(
                pattern_context, result, game_result
            )
            
            self.game_history.append(move_san)


if __name__ == "__main__":
    # Example usage
    learner = EnhancedPatternLearner()
    
    print("Enhanced Pattern Learning System initialized")
    print("This system learns from WHY patterns succeed or fail")
    print("Key features:")
    print("- Contextual factor analysis")
    print("- Failure cause diagnosis") 
    print("- Rule extraction from experience")
    print("- Meta-pattern recognition")
    print("- Causal reasoning about moves")
    
    learner.close()
