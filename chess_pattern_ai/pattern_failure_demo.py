#!/usr/bin/env python3
"""
Pattern Failure Analysis Demonstration

This demonstrates how the enhanced pattern learning system analyzes
WHY patterns fail and learns specific lessons to improve future decisions.

Example scenarios:
1. "I captured a piece but lost my queen" → Learn about piece protection
2. "I gave check but fell into counterplay" → Learn about timing
3. "I developed a piece but weakened king safety" → Learn about trade-offs
"""

import chess
from typing import Dict, List, Tuple, Optional
import json

class PatternFailureDemo:
    """
    Demonstrates enhanced pattern failure analysis
    """
    
    def __init__(self):
        self.learned_rules = []
        self.failure_patterns = {}
        self.success_conditions = {}
    
    def demonstrate_failure_analysis(self):
        """
        Show how enhanced system learns from different types of failures
        """
        
        print("=" * 80)
        print("ENHANCED PATTERN FAILURE ANALYSIS DEMONSTRATION")
        print("=" * 80)
        
        # Scenario 1: Tactical failure
        self.demo_tactical_failure()
        
        # Scenario 2: Positional failure  
        self.demo_positional_failure()
        
        # Scenario 3: Timing failure
        self.demo_timing_failure()
        
        # Scenario 4: Successful pattern recognition
        self.demo_successful_pattern()
        
        # Show what was learned
        self.show_learned_insights()
    
    def demo_tactical_failure(self):
        """
        Scenario: AI captures a piece but loses material to tactics
        """
        
        print("\n📋 SCENARIO 1: TACTICAL FAILURE")
        print("Situation: AI captures pawn but loses queen to fork")
        print("-" * 50)
        
        # Simulate the scenario
        scenario = {
            'move_made': 'Qxd7',
            'pattern_used': 'queen capture endgame',
            'expected_outcome': 'material gain (+100)',
            'actual_outcome': 'lost queen to knight fork (-900)',
            'context': {
                'piece_defended': False,
                'knight_fork_threat': True,
                'king_safety': 'exposed',
                'material_balance': -50
            }
        }
        
        print(f"Move made: {scenario['move_made']}")
        print(f"Pattern: {scenario['pattern_used']}")
        print(f"Expected: {scenario['expected_outcome']}")
        print(f"Actual: {scenario['actual_outcome']}")
        
        # Analyze failure
        analysis = self.analyze_tactical_failure(scenario)
        
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"Root cause: {analysis['root_cause']}")
        print(f"Missing check: {analysis['missing_precondition']}")
        print(f"Lesson learned: {analysis['lesson']}")
        
        # Extract rule
        rule = self.extract_rule_from_failure(analysis)
        self.learned_rules.append(rule)
        
        print(f"📚 RULE LEARNED: {rule['description']}")
        print(f"Confidence: {rule['confidence']:.1%}")
    
    def analyze_tactical_failure(self, scenario: Dict) -> Dict:
        """
        Detailed analysis of why tactical pattern failed
        """
        
        context = scenario['context']
        
        # Identify root cause
        if not context['piece_defended'] and context['knight_fork_threat']:
            root_cause = "moved_valuable_piece_to_undefended_square_with_fork_threat"
            missing_precondition = "verify_no_knight_forks_before_queen_moves"
            lesson = "Always check for knight fork patterns before moving queen to undefended square"
            
        elif context['king_safety'] == 'exposed':
            root_cause = "tactical_vulnerability_due_to_king_exposure"
            missing_precondition = "secure_king_before_aggressive_moves"
            lesson = "Secure king position before making aggressive captures"
            
        else:
            root_cause = "insufficient_tactical_calculation"
            missing_precondition = "calculate_opponent_threats"
            lesson = "Calculate opponent's best response before capturing"
        
        return {
            'root_cause': root_cause,
            'missing_precondition': missing_precondition,
            'lesson': lesson,
            'scenario': scenario
        }
    
    def demo_positional_failure(self):
        """
        Scenario: AI makes developing move but weakens position
        """
        
        print("\n📋 SCENARIO 2: POSITIONAL FAILURE")
        print("Situation: AI develops knight but weakens pawn structure")
        print("-" * 50)
        
        scenario = {
            'move_made': 'Nf3',
            'pattern_used': 'knight development opening',
            'expected_outcome': 'improved development',
            'actual_outcome': 'weakened kingside, lost castling rights',
            'context': {
                'development_phase': 'early',
                'king_safety': 'castling_available',
                'pawn_structure': 'intact',
                'center_control': 'contested'
            },
            'consequences': {
                'king_safety': 'compromised',
                'pawn_structure': 'weakened',
                'tactical_vulnerabilities': ['back_rank_weakness', 'f7_square']
            }
        }
        
        print(f"Move made: {scenario['move_made']}")
        print(f"Pattern: {scenario['pattern_used']}")
        print(f"Expected: {scenario['expected_outcome']}")
        print(f"Actual: {scenario['actual_outcome']}")
        
        # Analyze failure
        analysis = self.analyze_positional_failure(scenario)
        
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"Positional cost: {analysis['positional_cost']}")
        print(f"Trade-off missed: {analysis['tradeoff_analysis']}")
        print(f"Lesson learned: {analysis['lesson']}")
        
        rule = self.extract_rule_from_failure(analysis)
        self.learned_rules.append(rule)
        
        print(f"📚 RULE LEARNED: {rule['description']}")
    
    def analyze_positional_failure(self, scenario: Dict) -> Dict:
        """
        Analysis of positional pattern failure
        """
        
        consequences = scenario['consequences']
        
        if 'back_rank_weakness' in consequences['tactical_vulnerabilities']:
            positional_cost = "created_back_rank_weakness"
            tradeoff_analysis = "development_gain_vs_king_safety_loss"
            lesson = "Prioritize king safety over piece development in opening"
            
        else:
            positional_cost = "weakened_pawn_structure"
            tradeoff_analysis = "short_term_activity_vs_long_term_structure"
            lesson = "Consider long-term positional consequences of developing moves"
        
        return {
            'positional_cost': positional_cost,
            'tradeoff_analysis': tradeoff_analysis,
            'lesson': lesson,
            'scenario': scenario
        }
    
    def demo_timing_failure(self):
        """
        Scenario: Right pattern, wrong timing
        """
        
        print("\n📋 SCENARIO 3: TIMING FAILURE")
        print("Situation: AI attacks while underdeveloped")
        print("-" * 50)
        
        scenario = {
            'move_made': 'Qh5',
            'pattern_used': 'queen attack middlegame',
            'expected_outcome': 'attacking pressure',
            'actual_outcome': 'queen trapped, tempo lost',
            'context': {
                'development_phase': 'underdeveloped',
                'pieces_developed': 2,
                'opponent_development': 4,
                'material_balance': 0,
                'attack_readiness': 'premature'
            }
        }
        
        print(f"Move made: {scenario['move_made']}")
        print(f"Pattern: {scenario['pattern_used']}")
        print(f"Expected: {scenario['expected_outcome']}")
        print(f"Actual: {scenario['actual_outcome']}")
        
        analysis = self.analyze_timing_failure(scenario)
        
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"Timing issue: {analysis['timing_problem']}")
        print(f"Prerequisites missed: {analysis['prerequisites']}")
        print(f"Lesson learned: {analysis['lesson']}")
        
        rule = self.extract_rule_from_failure(analysis)
        self.learned_rules.append(rule)
        
        print(f"📚 RULE LEARNED: {rule['description']}")
    
    def analyze_timing_failure(self, scenario: Dict) -> Dict:
        """
        Analysis of timing-based pattern failure
        """
        
        context = scenario['context']
        
        if context['development_phase'] == 'underdeveloped':
            timing_problem = "premature_attack_before_development"
            prerequisites = ["complete_minor_piece_development", "secure_king_position"]
            lesson = "Complete development before launching attacks"
            
        elif context['pieces_developed'] < context['opponent_development']:
            timing_problem = "attacking_from_position_of_weakness"
            prerequisites = ["achieve_development_parity", "establish_coordination"]
            lesson = "Don't attack when behind in development"
            
        else:
            timing_problem = "poor_sequence_timing"
            prerequisites = ["prepare_attack_properly"]
            lesson = "Prepare attacks systematically"
        
        return {
            'timing_problem': timing_problem,
            'prerequisites': prerequisites,
            'lesson': lesson,
            'scenario': scenario
        }
    
    def demo_successful_pattern(self):
        """
        Scenario: Pattern works as expected - reinforce learning
        """
        
        print("\n📋 SCENARIO 4: SUCCESSFUL PATTERN")
        print("Situation: AI executes well-timed tactical combination")
        print("-" * 50)
        
        scenario = {
            'move_made': 'Nxe5',
            'pattern_used': 'knight capture middlegame',
            'expected_outcome': 'material gain and position',
            'actual_outcome': 'won pawn, improved position, forced opponent weakness',
            'context': {
                'piece_defended': True,
                'tactical_motifs': ['fork_threat', 'discovered_attack'],
                'king_safety': 'safe',
                'development_phase': 'developed',
                'material_balance': 50
            },
            'success_factors': [
                'piece_was_defended',
                'king_was_safe', 
                'development_complete',
                'tactical_sequence_calculated'
            ]
        }
        
        print(f"Move made: {scenario['move_made']}")
        print(f"Pattern: {scenario['pattern_used']}")
        print(f"Expected: {scenario['expected_outcome']}")
        print(f"Actual: {scenario['actual_outcome']}")
        
        analysis = self.analyze_successful_pattern(scenario)
        
        print(f"\n✅ SUCCESS ANALYSIS:")
        print(f"Success factors: {', '.join(analysis['success_factors'])}")
        print(f"Pattern reinforced: {analysis['pattern_strength']}")
        print(f"Confidence boost: +{analysis['confidence_increase']:.1%}")
        
        # Reinforce successful pattern
        self.reinforce_successful_pattern(analysis)
    
    def analyze_successful_pattern(self, scenario: Dict) -> Dict:
        """
        Analysis of successful pattern application
        """
        
        success_factors = scenario['success_factors']
        context = scenario['context']
        
        pattern_strength = "high" if len(success_factors) >= 3 else "moderate"
        confidence_increase = 0.1 * len(success_factors)
        
        return {
            'success_factors': success_factors,
            'pattern_strength': pattern_strength,
            'confidence_increase': confidence_increase,
            'scenario': scenario
        }
    
    def extract_rule_from_failure(self, analysis: Dict) -> Dict:
        """
        Extract a learnable rule from failure analysis
        """
        
        lesson = analysis['lesson']
        scenario = analysis['scenario']
        
        # Create specific, actionable rule
        rule = {
            'type': 'constraint',
            'description': lesson,
            'pattern_scope': scenario['pattern_used'],
            'conditions': self._extract_conditions(analysis),
            'confidence': 0.8,  # Start with moderate confidence
            'evidence': 1,
            'source': 'failure_analysis'
        }
        
        return rule
    
    def _extract_conditions(self, analysis: Dict) -> List[str]:
        """Extract specific conditions that trigger this rule"""
        
        scenario = analysis['scenario']
        context = scenario.get('context', {})
        
        conditions = []
        
        if 'piece_defended' in context:
            conditions.append(f"piece_defended: {context['piece_defended']}")
        if 'king_safety' in context:
            conditions.append(f"king_safety: {context['king_safety']}")
        if 'development_phase' in context:
            conditions.append(f"development_phase: {context['development_phase']}")
            
        return conditions
    
    def reinforce_successful_pattern(self, analysis: Dict):
        """
        Reinforce patterns that led to success
        """
        
        scenario = analysis['scenario']
        pattern = scenario['pattern_used']
        
        if pattern not in self.success_conditions:
            self.success_conditions[pattern] = []
        
        self.success_conditions[pattern].extend(analysis['success_factors'])
        
        print(f"📈 PATTERN REINFORCED: {pattern}")
    
    def show_learned_insights(self):
        """
        Display all insights learned from the demonstrations
        """
        
        print("\n" + "=" * 80)
        print("ENHANCED LEARNING INSIGHTS SUMMARY")
        print("=" * 80)
        
        print(f"\n📚 RULES LEARNED: {len(self.learned_rules)}")
        for i, rule in enumerate(self.learned_rules, 1):
            print(f"{i}. {rule['description']}")
            print(f"   Pattern scope: {rule['pattern_scope']}")
            print(f"   Confidence: {rule['confidence']:.1%}")
            print(f"   Conditions: {', '.join(rule['conditions'])}")
            print()
        
        print(f"🎯 SUCCESS PATTERNS REINFORCED:")
        for pattern, factors in self.success_conditions.items():
            unique_factors = list(set(factors))
            print(f"• {pattern}: {', '.join(unique_factors)}")
        
        print(f"\n🧠 META-INSIGHTS:")
        print("• Piece safety is critical before making captures")
        print("• King safety should be secured before attacking")  
        print("• Development timing affects attack success")
        print("• Pattern context determines success vs failure")
        print("• Same move can be excellent or terrible based on conditions")
        
        print(f"\n🔄 HOW AI IMPROVES:")
        print("1. Before each move, check learned rules for violations")
        print("2. Verify success conditions are met for chosen patterns")
        print("3. If pattern fails, analyze WHY and update rules")
        print("4. Gradually build sophisticated pattern library")
        print("5. Develop contextual understanding of when patterns work")


def demonstrate_pattern_evolution():
    """
    Show how patterns evolve from simple to sophisticated through learning
    """
    
    print("\n" + "=" * 80)
    print("PATTERN EVOLUTION DEMONSTRATION")
    print("=" * 80)
    
    print("\n🏃 STAGE 1: NAIVE PATTERN")
    print("Rule: 'Capture pieces when possible'")
    print("Result: Often loses material to tactics")
    
    print("\n🚶 STAGE 2: SAFETY-AWARE PATTERN")
    print("Rule: 'Capture pieces only if piece is defended'") 
    print("Result: Fewer tactical losses, but still positional problems")
    
    print("\n🧗 STAGE 3: CONTEXT-AWARE PATTERN")
    print("Rule: 'Capture pieces when defended AND king is safe AND no tactical motifs present'")
    print("Result: Much more reliable, but sometimes misses opportunities")
    
    print("\n🎯 STAGE 4: SOPHISTICATED PATTERN")
    print("Rule: 'Capture when defended OR forcing move OR material advantage justifies risk'")
    print("     'Check for: forks, pins, skewers, discovery attacks'")
    print("     'Consider: king safety, development, pawn structure impact'")
    print("     'Timing: ensure prerequisites met for chosen pattern'")
    print("Result: High success rate with contextual understanding")
    
    print("\n💡 KEY INSIGHT:")
    print("Enhanced learning transforms simple pattern counting into")
    print("sophisticated contextual reasoning about WHEN and WHY patterns work!")


if __name__ == "__main__":
    demo = PatternFailureDemo()
    
    # Run the demonstration
    demo.demonstrate_failure_analysis()
    
    # Show pattern evolution
    demonstrate_pattern_evolution()
    
    print("\n" + "=" * 80)
    print("This is how your enhanced pattern system would learn from failures!")
    print("Instead of just counting wins/losses, it understands WHY patterns fail")
    print("and builds increasingly sophisticated rules for pattern application.")
    print("=" * 80)
