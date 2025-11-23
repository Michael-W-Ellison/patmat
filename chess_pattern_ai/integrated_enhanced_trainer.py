#!/usr/bin/env python3
"""
Integration Layer: Enhanced Pattern Learning with Existing System

This shows how to integrate the enhanced contextual pattern learning
with your existing HeadlessTrainer and LearnableMovePrioritizer.

Key Integration Points:
1. Replace simple pattern counting with contextual analysis
2. Add failure diagnosis to move evaluation
3. Implement learned rule checking
4. Create feedback loop for pattern improvement
"""

import chess
from typing import Dict, List, Tuple, Optional
from enhanced_pattern_learner import EnhancedPatternLearner, ContextualFactors
from learnable_move_prioritizer import LearnableMovePrioritizer
from game_scorer import GameScorer


class IntegratedPatternAI:
    """
    Enhanced AI that integrates contextual pattern learning with existing system
    """
    
    def __init__(self, db_path='enhanced_training.db'):
        # Keep existing components
        self.traditional_prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = GameScorer()
        
        # Add enhanced pattern learning
        self.enhanced_learner = EnhancedPatternLearner(db_path)
        
        # Game state tracking
        self.current_game_moves = []
        self.move_evaluations = []  # Store why each move was made
        
    def evaluate_move_enhanced(self, board: chess.Board, move: chess.Move) -> Dict:
        """
        Comprehensive move evaluation combining traditional + enhanced learning
        """
        
        # 1. Traditional evaluation (material + basic patterns)
        basic_priority = self.traditional_prioritizer.get_move_priority(board, move)
        
        board.push(move)
        material_score = self._calculate_material_advantage(board)
        board.pop()
        
        traditional_score = material_score + (basic_priority * 20)
        
        # 2. Enhanced contextual analysis
        pattern_advice = self.enhanced_learner.get_pattern_advice(
            board, move, [m[2] for m in self.current_game_moves]
        )
        
        # 3. Combine evaluations intelligently
        final_evaluation = self._combine_evaluations(
            traditional_score, pattern_advice, board, move
        )
        
        return final_evaluation
    
    def _combine_evaluations(self, traditional_score: float, 
                           pattern_advice: Dict, 
                           board: chess.Board, 
                           move: chess.Move) -> Dict:
        """
        Intelligently combine traditional and enhanced evaluations
        """
        
        base_score = traditional_score
        confidence_multiplier = pattern_advice['confidence']
        
        # Apply contextual modifiers
        if pattern_advice['recommendation'] == 'avoid':
            # Enhanced learning strongly advises against this
            base_score *= 0.1  # Severe penalty
            reasoning = f"Pattern learning advises avoiding: {pattern_advice['warnings']}"
            
        elif pattern_advice['recommendation'] == 'caution':
            # Some concerns but not definitively bad
            base_score *= 0.5  # Moderate penalty
            reasoning = f"Caution advised: {pattern_advice['warnings']}"
            
        elif pattern_advice['recommendation'] == 'proceed':
            # Enhanced learning supports this move
            base_score *= (1.0 + confidence_multiplier * 0.5)  # Confidence bonus
            reasoning = f"Pattern learning supports: {pattern_advice['encouragements']}"
            
        # Rule violation penalties
        if pattern_advice['rule_violations']:
            base_score *= 0.3  # Heavy penalty for violating learned rules
            reasoning += f" | Rule violations: {pattern_advice['rule_violations']}"
        
        # Final confidence-weighted score
        final_score = base_score * confidence_multiplier
        
        return {
            'score': final_score,
            'confidence': pattern_advice['confidence'],
            'reasoning': reasoning,
            'traditional_score': traditional_score,
            'pattern_advice': pattern_advice,
            'move_quality': self._assess_move_quality(pattern_advice)
        }
    
    def _assess_move_quality(self, pattern_advice: Dict) -> str:
        """Assess overall move quality for learning"""
        
        if pattern_advice['confidence'] > 0.8 and pattern_advice['recommendation'] == 'proceed':
            return 'excellent'
        elif pattern_advice['confidence'] > 0.6 and not pattern_advice['rule_violations']:
            return 'good'
        elif pattern_advice['confidence'] > 0.4:
            return 'questionable'
        else:
            return 'poor'
    
    def select_best_move(self, board: chess.Board) -> Tuple[chess.Move, Dict]:
        """
        Select best move using enhanced pattern learning
        """
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}
        
        # Evaluate all legal moves
        move_evaluations = []
        
        for move in legal_moves:
            evaluation = self.evaluate_move_enhanced(board, move)
            move_evaluations.append((move, evaluation))
        
        # Sort by final score
        move_evaluations.sort(key=lambda x: x[1]['score'], reverse=True)
        
        best_move, best_eval = move_evaluations[0]
        
        # Store evaluation for learning
        self.move_evaluations.append({
            'move': best_move,
            'evaluation': best_eval,
            'all_evaluations': move_evaluations
        })
        
        return best_move, best_eval
    
    def record_move_result(self, board: chess.Board, move: chess.Move, 
                          evaluation: Dict):
        """
        Record move and its immediate consequences for learning
        """
        
        # Capture position before move
        fen_before = board.fen()
        
        # Execute move
        board.push(move)
        
        # Analyze immediate consequences
        material_change = self._calculate_immediate_material_change(
            board, move, fen_before
        )
        
        position_change = self._calculate_position_change(board, fen_before)
        
        # Record for current game
        self.current_game_moves.append((fen_before, move.uci(), board.san(move)))
        
        # Store move result for later analysis
        move_result = {
            'fen_before': fen_before,
            'move': move,
            'material_change': material_change,
            'position_change': position_change,
            'evaluation': evaluation
        }
        
        return move_result
    
    def learn_from_completed_game(self, final_result: str, final_score: int):
        """
        Comprehensive learning from completed game
        """
        
        print(f"\n=== ENHANCED LEARNING SESSION ===")
        print(f"Game Result: {final_result}, Score: {final_score}")
        
        # 1. Traditional pattern learning (keep existing)
        self.traditional_prioritizer.record_game_moves(
            self.current_game_moves, chess.WHITE, final_result, final_score
        )
        
        # 2. Enhanced contextual learning
        self._analyze_game_for_enhanced_learning(final_result, final_score)
        
        # 3. Extract insights and rules
        insights = self._extract_game_insights()
        
        # 4. Update failure/success patterns
        self._update_pattern_database()
        
        print(f"Learned {len(insights)} new insights from this game")
        
        # Reset for next game
        self.current_game_moves = []
        self.move_evaluations = []
        
        return insights
    
    def _analyze_game_for_enhanced_learning(self, final_result: str, final_score: int):
        """
        Analyze each move in context for enhanced learning
        """
        
        print(f"\nAnalyzing {len(self.move_evaluations)} moves for pattern learning...")
        
        board = chess.Board()
        
        for i, move_data in enumerate(self.move_evaluations):
            move = move_data['move']
            evaluation = move_data['evaluation']
            
            # Set board position
            if i < len(self.current_game_moves):
                board.set_fen(self.current_game_moves[i][0])
            
            # Analyze what actually happened after this move
            actual_result = self._analyze_move_outcome(
                move, evaluation, final_result, final_score, i
            )
            
            # Learn from the outcome
            self._learn_from_move_outcome(
                board, move, evaluation, actual_result
            )
            
            # Apply move for next iteration
            board.push(move)
    
    def _analyze_move_outcome(self, move: chess.Move, evaluation: Dict, 
                            final_result: str, final_score: int, move_index: int) -> Dict:
        """
        Determine what actually happened after making this move
        """
        
        predicted_quality = evaluation['move_quality']
        
        # Analyze if prediction was correct
        if final_result == 'win' and final_score > 200:
            actual_quality = 'excellent' if move_index < len(self.move_evaluations) * 0.7 else 'good'
        elif final_result == 'draw':
            actual_quality = 'questionable'
        else:  # loss
            actual_quality = 'poor' if final_score < -200 else 'questionable'
        
        prediction_accuracy = self._compare_predictions(predicted_quality, actual_quality)
        
        return {
            'predicted_quality': predicted_quality,
            'actual_quality': actual_quality,
            'prediction_accuracy': prediction_accuracy,
            'final_result': final_result,
            'final_score': final_score
        }
    
    def _learn_from_move_outcome(self, board: chess.Board, move: chess.Move, 
                               evaluation: Dict, outcome: Dict):
        """
        Learn from what actually happened vs what was predicted
        """
        
        if outcome['prediction_accuracy'] == 'wrong':
            # Our pattern learning made a mistake - analyze why
            self._analyze_prediction_failure(board, move, evaluation, outcome)
            
        elif outcome['prediction_accuracy'] == 'correct':
            # Successful prediction - reinforce the patterns
            self._reinforce_successful_patterns(board, move, evaluation, outcome)
    
    def _analyze_prediction_failure(self, board: chess.Board, move: chess.Move,
                                  evaluation: Dict, outcome: Dict):
        """
        Deep analysis of why our pattern prediction failed
        """
        
        print(f"  ❌ Prediction failure: {evaluation['reasoning']}")
        
        # Extract what went wrong
        if outcome['predicted_quality'] == 'excellent' and outcome['actual_quality'] == 'poor':
            failure_type = "overconfident_prediction"
            lesson = "Pattern seemed good but missed critical flaw"
            
        elif outcome['predicted_quality'] == 'poor' and outcome['actual_quality'] == 'excellent':
            failure_type = "missed_opportunity" 
            lesson = "Rejected good move due to overly cautious patterns"
            
        else:
            failure_type = "moderate_misprediction"
            lesson = "Pattern evaluation needs refinement"
        
        # Learn specific lesson
        context = evaluation['pattern_advice']['context_analysis']
        self._record_pattern_failure_lesson(
            context, move, failure_type, lesson, evaluation
        )
    
    def _reinforce_successful_patterns(self, board: chess.Board, move: chess.Move,
                                     evaluation: Dict, outcome: Dict):
        """
        Strengthen patterns that made correct predictions
        """
        
        print(f"  ✓ Successful prediction: {evaluation['reasoning']}")
        
        # Identify which patterns contributed to success
        successful_patterns = evaluation['pattern_advice']['encouragements']
        avoided_failures = evaluation['pattern_advice']['warnings']
        
        # Reinforce these pattern recognitions
        self._strengthen_pattern_confidence(successful_patterns, avoided_failures)
    
    def _compare_predictions(self, predicted: str, actual: str) -> str:
        """Compare predicted vs actual move quality"""
        
        quality_order = ['poor', 'questionable', 'good', 'excellent']
        pred_idx = quality_order.index(predicted)
        actual_idx = quality_order.index(actual)
        
        if abs(pred_idx - actual_idx) <= 1:
            return 'correct'
        else:
            return 'wrong'
    
    def _extract_game_insights(self) -> List[str]:
        """Extract key insights from this game"""
        
        insights = []
        
        # Analyze prediction accuracy
        correct_predictions = sum(1 for eval_data in self.move_evaluations 
                                if eval_data.get('prediction_accuracy') == 'correct')
        
        accuracy_rate = correct_predictions / len(self.move_evaluations) if self.move_evaluations else 0
        
        if accuracy_rate > 0.7:
            insights.append("Pattern recognition performed well this game")
        elif accuracy_rate < 0.5:
            insights.append("Pattern recognition needs improvement - many failed predictions")
        
        # Analyze common failure patterns
        failure_types = [eval_data.get('failure_type') 
                        for eval_data in self.move_evaluations 
                        if eval_data.get('failure_type')]
        
        if 'overconfident_prediction' in failure_types:
            insights.append("Tendency to overestimate move quality - need more caution")
            
        if 'missed_opportunity' in failure_types:
            insights.append("Tendency to underestimate moves - need more confidence in good patterns")
        
        return insights
    
    def _record_pattern_failure_lesson(self, context: Dict, move: chess.Move,
                                     failure_type: str, lesson: str, evaluation: Dict):
        """Record specific lesson about why pattern failed"""
        
        # This would integrate with the enhanced learner's failure analysis
        # For now, just log the lesson
        print(f"    Lesson learned: {lesson}")
    
    def _strengthen_pattern_confidence(self, successful_patterns: List[str], 
                                     avoided_failures: List[str]):
        """Strengthen confidence in successful pattern recognition"""
        
        # This would update pattern confidence scores
        # For now, just log the success
        print(f"    Strengthened confidence in: {successful_patterns}")
    
    def _update_pattern_database(self):
        """Update pattern database with new insights"""
        
        # This would commit all learned lessons to the enhanced database
        pass
    
    # Helper methods
    def _calculate_material_advantage(self, board: chess.Board) -> float:
        """Calculate current material advantage"""
        return self.scorer._calculate_material(board, board.turn) - \
               self.scorer._calculate_material(board, not board.turn)
    
    def _calculate_immediate_material_change(self, board: chess.Board, 
                                           move: chess.Move, fen_before: str) -> int:
        """Calculate immediate material change from this move"""
        
        # Set up before position
        before_board = chess.Board(fen_before)
        before_material = self._calculate_material_advantage(before_board)
        
        # Current position material
        current_material = self._calculate_material_advantage(board)
        
        return current_material - before_material
    
    def _calculate_position_change(self, board: chess.Board, fen_before: str) -> int:
        """Estimate positional change (simplified)"""
        
        # Simplified positional evaluation
        # In reality, this would use sophisticated position evaluation
        return 0
    
    def close(self):
        """Clean up resources"""
        self.traditional_prioritizer.close()
        self.enhanced_learner.close()


# Example usage - Enhanced Headless Trainer
class EnhancedHeadlessTrainer:
    """
    Headless trainer using enhanced pattern learning
    """
    
    def __init__(self, db_path='enhanced_training.db'):
        self.ai = IntegratedPatternAI(db_path)
        self.scorer = GameScorer()
        
        # Game statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.insights_learned = []
    
    def play_enhanced_game(self, ai_color: bool, verbose: bool = False) -> Tuple[str, int, int]:
        """
        Play a game with enhanced pattern learning
        """
        
        board = chess.Board()
        move_count = 0
        
        if verbose:
            print(f"\n=== ENHANCED GAME (AI plays {'WHITE' if ai_color else 'BLACK'}) ===")
        
        while not board.is_game_over():
            if board.turn == ai_color:
                # AI move with enhanced evaluation
                move, evaluation = self.ai.select_best_move(board)
                
                if move:
                    if verbose:
                        print(f"AI Move: {board.san(move)} | "
                              f"Quality: {evaluation['move_quality']} | "
                              f"Confidence: {evaluation['confidence']:.2f}")
                        if evaluation['reasoning']:
                            print(f"  Reasoning: {evaluation['reasoning']}")
                    
                    # Record move and immediate result
                    self.ai.record_move_result(board, move, evaluation)
                
            else:
                # Opponent move (random for now)
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = legal_moves[0] if len(legal_moves) == 1 else \
                           random.choice(legal_moves)
                    board.push(move)
                    move_count += 1
        
        # Calculate final result
        rounds_played = board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)
        
        # Enhanced learning from completed game
        insights = self.ai.learn_from_completed_game(result, final_score)
        self.insights_learned.extend(insights)
        
        # Update statistics
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1
        
        if verbose:
            print(f"\nGame Result: {result.upper()}, Score: {final_score:+d}")
            print(f"Insights learned: {len(insights)}")
        
        return result, final_score, rounds_played
    
    def train_enhanced(self, num_games: int, verbose: bool = False):
        """
        Train using enhanced pattern learning
        """
        
        print("=" * 70)
        print(f"ENHANCED PATTERN LEARNING TRAINING - {num_games} GAMES")
        print("=" * 70)
        
        start_time = time.time()
        
        for i in range(num_games):
            ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            
            result, score, rounds = self.play_enhanced_game(ai_color, verbose)
            
            if (i + 1) % 10 == 0 or verbose:
                win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0
                print(f"Game {i+1}: {result.upper()} ({score:+d}) | "
                      f"WR: {win_rate:.1f}% | "
                      f"Insights: {len(self.insights_learned)}")
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("ENHANCED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Results: {self.wins}W-{self.losses}L-{self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played*100):.1f}%")
        print(f"Total Insights Learned: {len(self.insights_learned)}")
        print(f"Time: {elapsed:.1f}s")
        
        # Show key insights
        if self.insights_learned:
            print(f"\nKey Insights Learned:")
            for insight in set(self.insights_learned):  # Remove duplicates
                count = self.insights_learned.count(insight)
                print(f"  - {insight} ({count}x)")
    
    def close(self):
        """Clean up resources"""
        self.ai.close()


if __name__ == "__main__":
    import time
    import random
    
    # Example usage
    trainer = EnhancedHeadlessTrainer()
    
    try:
        # Train with enhanced pattern learning
        trainer.train_enhanced(num_games=5, verbose=True)
        
    finally:
        trainer.close()
