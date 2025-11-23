#!/usr/bin/env python3
"""
Quick Test - Verify Enhanced System Recovery
"""

import chess
from game_scorer import GameScorer

# Try to import the prioritizer
try:
    from learnable_move_prioritizer import LearnableMovePrioritizer
    prioritizer = LearnableMovePrioritizer("enhanced_training.db")
    print("✓ Successfully loaded prioritizer")
except Exception as e:
    print(f"❌ Failed to load prioritizer: {e}")
    exit(1)

# Test queen move priorities
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Test a typical queen development move
queen_move = chess.Move.from_uci("d1d3")  # Queen to d3

try:
    priority = prioritizer.get_move_priority(board, queen_move)
    print(f"👑 Queen move priority: {priority:.1f}")
    
    if priority > 20:
        print("✅ Queen priorities look healthy!")
    else:
        print("⚠️ Queen priorities still low")
        
except Exception as e:
    print(f"❌ Error testing queen move: {e}")

# Quick pattern check
try:
    stats = prioritizer.get_statistics()
    print(f"\n📊 Pattern Statistics:")
    print(f"   Patterns learned: {stats['patterns_learned']}")
    print(f"   Average confidence: {stats['avg_confidence']:.2f}")
    print(f"   Average win rate: {stats['avg_win_rate']:.1%}")
    
    if stats['avg_win_rate'] > 0.15:
        print("✅ System appears recovered!")
    else:
        print("⚠️ Win rates still problematic")
        
except Exception as e:
    print(f"❌ Error getting statistics: {e}")

prioritizer.close()
print("\n🧪 Quick test complete")
