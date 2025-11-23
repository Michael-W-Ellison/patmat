#!/usr/bin/env python3
"""
Test Enhanced Code with Migrated Database
"""

import sys
sys.path.append('chess_pattern_ai')

try:
    from learnable_move_prioritizer import LearnableMovePrioritizer
    
    print("🧪 Testing enhanced code with migrated database...")
    
    # Initialize with migrated database
    prioritizer = LearnableMovePrioritizer("headless_training.db")
    print("✅ Enhanced prioritizer loaded successfully")
    
    # Get statistics
    stats = prioritizer.get_statistics()
    print(f"📊 Patterns: {stats['patterns_learned']}")
    print(f"📊 Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"📊 Avg win rate: {stats['avg_win_rate']:.1%}")
    
    prioritizer.close()
    
    print("\n🎉 Enhanced code works with migrated database!")
    print("\nReady to train:")
    print("   python chess_pattern_ai/headless_trainer.py 50")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Migration may not have completed successfully")
