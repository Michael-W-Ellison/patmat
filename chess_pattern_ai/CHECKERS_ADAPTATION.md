# Adapting Pattern Recognition AI for Checkers

## Overview

The learning-based architecture can be adapted for checkers with minimal changes. The core pattern recognition and outcome-aware learning transfer directly.

## Architecture Compatibility

### ✓ What Transfers Directly (80% of code)

1. **Pattern Abstraction Engine** (`pattern_abstraction_engine.py`)
   - Extracts abstract patterns: "exposed_piece", "forced_capture_miss", "weak_back_row"
   - Works identically for any board game
   - No changes needed

2. **Outcome-Aware Learning** (entire system)
   - Tracks win/loss/draw for each pattern
   - Applies penalties based on win rates
   - 0% win rate patterns get massive penalties
   - Game-agnostic

3. **Adaptive Pattern Cache** (`adaptive_pattern_cache.py`)
   - Caches pattern evaluations
   - No changes needed

4. **Search Algorithms** (`optimized_search.py`)
   - Minimax with alpha-beta pruning
   - Quiescence search
   - Move ordering
   - Works for any two-player zero-sum game

5. **Opening Performance Tracker** (`opening_performance_tracker.py`)
   - Tracks which openings lead to wins
   - Game-agnostic

6. **Temporal Evaluator** (`temporal_evaluator.py`)
   - Adapts weights by game phase (opening/midgame/endgame)
   - Concept applies to checkers

### 🔧 What Needs Adaptation (20% of code)

#### 1. Board Representation

**Chess:** 64 squares, all usable
**Checkers:** 64 squares, only 32 dark squares usable

```python
class CheckersBoard:
    """Checkers board representation"""

    def __init__(self):
        # Only dark squares (1, 3, 5, 7 on odd ranks, 0, 2, 4, 6 on even ranks)
        self.dark_squares = self._get_dark_squares()
        self.board = [None] * 32  # Only 32 playable squares
        self.side_to_move = 'black'  # Black moves first in checkers

    def _get_dark_squares(self):
        """Map 32 playable squares to 64-square board"""
        dark_squares = []
        for rank in range(8):
            for file in range(8):
                if (rank + file) % 2 == 1:  # Dark squares
                    dark_squares.append(rank * 8 + file)
        return dark_squares
```

#### 2. Piece Movement Discovery

**Chess:** Complex - 6 piece types, 156 movement rules
**Checkers:** Simpler - 2 piece types (men, kings)

```python
class CheckersRuleDiscovery:
    """Discover checkers movement rules from observation"""

    def discover_man_movement(self, games):
        """
        Discover: Men move diagonally forward one square
        Capture: Jump over enemy piece to empty square beyond
        """
        patterns = {
            'forward_diagonal_move': [],
            'capture_jump': [],
            'forced_capture': []  # Must capture if possible
        }

        for game in games:
            for move in game.moves:
                if is_man_move(move):
                    if is_capture(move):
                        patterns['capture_jump'].append(move)
                    else:
                        patterns['forward_diagonal_move'].append(move)

        return self._extract_movement_rules(patterns)

    def discover_king_movement(self, games):
        """
        Discover: Kings can move diagonally forward OR backward
        Capture: Can jump forward or backward
        """
        # Similar pattern extraction
        pass
```

#### 3. Piece Values

**Chess:** P=1, N=3, B=3, R=5, Q=9
**Checkers:** Man=1, King=3 (much simpler!)

```python
# discovered_piece_values table
# piece_type | relative_value | confidence
# -----------|----------------|------------
# 'M'        | 1.0            | 0.95
# 'K'        | 3.0            | 0.95
```

#### 4. Evaluation Features

Replace chess-specific evaluators with checkers-specific:

```python
# CHESS evaluators to REMOVE:
# - pawn_structure_evaluator.py (no pawns in checkers)
# - tactical_evaluator.py (different tactics)

# CHECKERS evaluators to ADD:
# - back_row_evaluator.py (control of promotion squares)
# - bridge_evaluator.py (checkers-specific formations)
# - tempo_evaluator.py (tempo matters more in checkers)
```

**Back Row Evaluator:**
```python
class BackRowEvaluator:
    """Evaluate control of back row (promotion squares)"""

    def evaluate_back_row_control(self, board):
        """
        Checkers insight: Controlling opponent's back row
        prevents their men from promoting to kings
        """
        black_back_row = [0, 2, 4, 6]  # Rank 0
        white_back_row = [56, 58, 60, 62]  # Rank 7

        score = 0.0

        # Check who controls promotion squares
        for sq in black_back_row:
            if board.is_controlled_by(sq, 'white'):
                score += 10.0  # Good for white

        for sq in white_back_row:
            if board.is_controlled_by(sq, 'black'):
                score -= 10.0  # Good for black

        return score
```

#### 5. Game Rules

**Forced Capture Rule:**
```python
def get_legal_moves(board):
    """
    Checkers rule: If a capture is possible, MUST capture
    """
    captures = get_all_captures(board)

    if captures:
        return captures  # MUST capture
    else:
        return get_all_quiet_moves(board)
```

**Multi-Jump:**
```python
def generate_multi_jumps(board, piece_square):
    """
    Checkers rule: Continue jumping if more captures available
    """
    jumps = []

    # Get initial jump
    for target in get_jump_targets(board, piece_square):
        board.push(jump_move(piece_square, target))

        # Recursively check for more jumps
        if has_more_jumps(board, target):
            jumps.extend(generate_multi_jumps(board, target))
        else:
            jumps.append(get_move_sequence(board))

        board.pop()

    return jumps
```

## Implementation Plan

### Phase 1: Core Adaptation (1-2 days)
1. Create `checkers_board.py` - Board representation
2. Create `checkers_rule_discovery.py` - Discover movement rules
3. Create `checkers_evaluator.py` - Basic evaluation (piece count, back row control)

### Phase 2: Learning Integration (1 day)
4. Adapt `pattern_abstraction_engine.py` for checkers patterns
5. Connect to existing learning infrastructure
6. Test with 100 games vs random player

### Phase 3: Refinement (1-2 days)
7. Add checkers-specific evaluators (bridges, tempo, kings vs men)
8. Tune search depth (checkers games are shorter)
9. Train against stronger opponents

### Phase 4: Validation (1 day)
10. Test suite for checkers rules
11. Verify pattern learning
12. Benchmark against known checkers programs

## Expected Results

### Advantages for Checkers

1. **Simpler Rules** → Faster rule discovery
   - Only 2 piece types vs 6 in chess
   - Simpler movement patterns
   - Fewer possible moves per position (avg ~7 vs ~35)

2. **Shorter Games** → Faster learning
   - Average checkers game: 40-50 moves
   - Average chess game: 80+ moves
   - Can play 2x more games in same time

3. **Forced Capture** → Clearer tactics
   - Must-capture rule makes tactics more forced
   - Pattern "missed_forced_capture" → 0% win rate
   - Easier to learn optimal play

### Learning Trajectory

Based on chess results (0% → 5-10% → 20-30%):

```
Games 1-50:   Learn basic rules, 0-5% win rate
Games 51-200: Avoid obvious blunders, 10-20% win rate
Games 201-500: Learn tactical patterns, 25-35% win rate
Games 501+:   Refine strategy, 35-45% win rate
```

**Prediction:** Checkers AI reaches competence faster than chess AI due to simpler rule space.

## Code Reuse Estimate

- **80%** of learning infrastructure: No changes
- **15%** of evaluation code: Adapt to checkers features
- **5%** new code: Checkers-specific rules and patterns

**Total effort:** ~5-7 days to adapt entire system

## Example Learned Patterns

### Chess Patterns (Current)
```
tempo_loss: moved_same_piece_twice → 0% win rate
hanging_piece: knight_undefended → 0% win rate
premature_development: queen_before_minors → 0% win rate
```

### Expected Checkers Patterns
```
exposed_back_row: left_promotion_square_unguarded → 0% win rate
missed_king_promotion: failed_to_advance_to_back_rank → 5% win rate
piece_sacrifice: gave_up_piece_without_forcing_sequence → 0% win rate
isolated_piece: moved_piece_away_from_group → 10% win rate
weak_bridge: failed_to_maintain_connected_pieces → 15% win rate
```

## Key Insight

The pattern recognition approach is **more powerful for checkers** than chess because:

1. **Smaller state space** → Patterns generalize better
2. **Forced captures** → Tactics are more deterministic
3. **Simpler piece types** → Fewer pattern categories needed
4. **Shorter games** → Faster cause-effect learning

## Database Schema Changes

Most tables stay the same, just update piece types:

```sql
-- discovered_piece_values
-- OLD (Chess): P, N, B, R, Q, K (6 types)
-- NEW (Checkers): M, K (2 types - Man, King)

-- abstract_patterns
-- Same structure, different pattern types:
-- - "exposed_back_row" instead of "hanging_piece"
-- - "weak_bridge" instead of "doubled_pawns"

-- games table
-- No changes needed!
```

## Conclusion

**The pattern recognition architecture is ideally suited for checkers.**

The learning approach:
- ✓ Discovers rules from observation
- ✓ Learns patterns from mistakes
- ✓ Tracks outcomes (win/loss/draw)
- ✓ Applies penalties to losing patterns
- ✓ Improves over time

**This is game-agnostic and should work for:**
- Checkers ✓
- Chess ✓
- Othello ✓
- Connect Four ✓
- Go (with modifications for larger board) ✓
- Any deterministic two-player game ✓

The core insight: **Learn what loses games, avoid those patterns.** This works universally.

## Next Steps

If you want to adapt for checkers:

1. Create `checkers_board.py` (150 lines)
2. Create `checkers_evaluator.py` (100 lines)
3. Adapt `fast_learning_ai.py` to use checkers board (20 lines changed)
4. Run 100 games and watch it learn

**Estimated time:** 1-2 days for basic working system, 1 week for competitive play.
