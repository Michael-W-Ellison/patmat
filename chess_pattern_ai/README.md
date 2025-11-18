# Chess Pattern Recognition AI - Backup

## Contents

This backup contains the complete pattern recognition chess AI system.

### Core Components

1. **Main AI**
   - `fast_learning_ai.py` - Main entry point with optimized search
   - `integrated_chess_ai.py` - Integrated AI with all evaluators
   - `integrated_ai_with_clustering.py` - AI with position clustering
   - `optimized_search.py` - Intelligent move pruning and search

2. **Pattern Learning System**
   - `pattern_abstraction_engine.py` - Extracts abstract patterns from mistakes
   - `test_learning_ai_with_clustering.py` - Game tracking and learning
   - `opening_performance_tracker.py` - Opening book learning

3. **Evaluators**
   - `discovered_chess_engine.py` - Discovered piece movement rules
   - `mobility_evaluator.py` - Piece mobility evaluation
   - `tactical_evaluator.py` - Tactical pattern recognition
   - `pawn_structure_evaluator.py` - Pawn structure analysis
   - `positional_evaluator.py` - Positional evaluation
   - `king_safety_evaluator.py` - King safety assessment

4. **Pattern Recognition**
   - `integrate_clustering.py` - Position clustering system
   - `adaptive_pattern_cache.py` - Learning pattern cache
   - `pattern_database_enhancer.py` - Database pattern enhancement

5. **Database**
   - `rule_discovery.db` - Clean 2.9MB knowledge base with:
     - 11 abstract patterns (learned principles)
     - 156 movement rules
     - ~30 evaluation weights
     - 20 position cluster centers
     - Game statistics

6. **Utilities**
   - `create_clean_database.py` - Database cleanup tool
   - `cleanup_database.py` - Alternative cleanup approach

### Key Features

1. **Pattern Abstraction**
   - Learns WHY moves are bad, not just WHICH moves
   - Extracts abstract patterns like "hanging pieces" instead of memorizing positions

2. **Outcome-Aware Learning**
   - Tracks win/loss/draw for each pattern
   - Applies stronger penalties to patterns with 0% win rate
   - Connects actions to game outcomes

3. **Optimized Search**
   - Intelligent move pruning (25x faster)
   - Two-stage search: root ordering + deep search
   - Adaptive pattern caching

4. **Compact Knowledge Base**
   - 2.9MB database (was 22GB before cleanup)
   - Only abstract knowledge, no position memorization
   - 13,776 essential rows vs 47 million bloat rows

## Usage

### Run AI vs Stockfish
```bash
python3 fast_learning_ai.py 10
```

### Run with Stockfish Feedback
```bash
python3 fast_learning_ai.py 20 --stockfish-feedback
```

### Clean Database (if needed)
```bash
python3 create_clean_database.py
```

## Requirements

- Python 3.8+
- python-chess
- numpy
- scikit-learn
- sqlite3

## Documentation

See `docs/` folder for:
- Outcome-aware learning implementation details
- Database cleanup summary
- Test results and analysis

## Backup Date

Created: 2025-11-17 22:11:41

## Version

Pattern Recognition Chess AI v2.0
- Abstract pattern learning
- Outcome-aware penalties
- Clean compact database (2.9MB)
