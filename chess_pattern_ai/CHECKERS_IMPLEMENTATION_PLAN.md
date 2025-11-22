# Checkers Implementation Plan

## Goal
Extend the pattern recognition AI to play checkers using the same learning philosophy:
- No hardcoded strategy knowledge
- Learn patterns from observation
- Differential scoring
- Progressive difficulty training

## Phase 1: Game Engine (1-2 days)

### Option A: Use Existing Library
```bash
pip install imparaai-checkers
# or
pip install checkers  # simpler library
```

**Pros:**
- Fast implementation
- Tested rules
- Standard notation

**Cons:**
- External dependency
- May not match our API style

### Option B: Custom Implementation
```python
class CheckersBoard:
    def __init__(self):
        # 8x8 board, only diagonals used
        # 12 pieces per side (rows 0-2 and 5-7)

    def legal_moves(self):
        # Generate legal moves (forced captures first)

    def push(self, move):
        # Apply move, check for king promotion

    def is_game_over(self):
        # No legal moves or no pieces

    def result(self):
        # Which side won
```

**Pros:**
- Full control
- Matches our API
- Educational

**Cons:**
- More work
- Need thorough testing

### Recommended: **Option A** for speed, **Option B** for learning

## Phase 2: Scoring Adapter (0.5 days)

```python
class CheckersScorer(GameScorer):
    """Checkers-specific scoring using differential approach"""

    PIECE_VALUES = {
        'man': 100,    # Regular piece
        'king': 300    # Promoted king (3x value)
    }

    def calculate_material(self, board, color):
        """Count men + kings"""
        men = count_men(board, color) * 100
        kings = count_kings(board, color) * 300
        return men + kings

    # Inherit differential scoring from parent!
    # - Win: advantage + time_bonus + 1000
    # - Loss: advantage - 1000
    # - Draw: advantage
```

## Phase 3: Pattern Classifier (0.5 days)

```python
class CheckersMovePrioritizer(LearnableMovePrioritizer):
    """Checkers move patterns"""

    def classify_move(self, board, move):
        return {
            'piece_type': 'king' if is_king else 'man',
            'move_category': self._get_category(move),
            'game_phase': self._get_phase(board),
            'distance': 1  # Checkers moves are simpler
        }

    def _get_category(self, move):
        if is_capture(move):
            if is_multi_jump(move):
                return 'multi_jump'  # Double/triple jump
            return 'single_jump'
        return 'quiet'

    def _get_phase(self, board):
        piece_count = total_pieces(board)
        if piece_count > 16:
            return 'opening'
        elif piece_count > 8:
            return 'middlegame'
        else:
            return 'endgame'
```

## Phase 4: Integration (1 day)

### Checkers Progressive Trainer
```python
# checkers_progressive_trainer.py
# Nearly identical to chess version!

class CheckersProgressiveTrainer:
    def __init__(self, opponent_levels):
        self.scorer = CheckersScorer()
        self.prioritizer = CheckersMovePrioritizer()
        # Rest is the same!
```

### Checkers Headless Trainer
```python
# checkers_headless_trainer.py
# Same structure, different rules engine
```

### Checkers GUI
```python
# checkers_gui.py
# Reuse GUI framework, swap board rendering
```

## Phase 5: Testing (0.5 days)

```bash
# Quick test against random
python3 checkers_headless_trainer.py 100

# Progressive training
python3 checkers_progressive_trainer.py 500

# View patterns
python3 pattern_viewer_gui.py checkers_training.db
```

## Expected Results

### Learning Speed
Checkers should learn **FASTER** than chess:
- Simpler patterns (fewer piece types)
- More forced moves (captures mandatory)
- Clearer tactical sequences

**Prediction:**
- Chess: 50 games to beat random consistently
- Checkers: 20 games to beat random consistently

### Pattern Examples

**Chess learns:**
- Knight forks
- Bishop pins
- Center control
- Pawn structure

**Checkers learns:**
- Force opponent to jump
- Multi-jump combinations
- King promotion timing
- Edge control
- Sacrifice for better position

### Scoring Examples

```
Chess win, ahead +500:   500 + 90 + 1000 = +1590
Checkers win, ahead +400: 400 + 90 + 1000 = +1490

Both use same differential formula!
```

## Code Reuse Percentage

Estimate: **85% code reuse**

### Shared (85%):
- Differential scoring formula ✓
- Pattern learning system ✓
- Database schema ✓
- Training infrastructure ✓
- Pattern decay manager ✓
- Pattern viewer GUI ✓
- All documentation structure ✓

### New (15%):
- Checkers rules engine
- Piece value constants
- Move classification details
- Board rendering (GUI)

## File Structure

```
chess_pattern_ai/
├── game_scorer.py              # Base class (shared)
├── learnable_move_prioritizer.py  # Base class (shared)
├── pattern_decay_manager.py   # Shared
├── pattern_viewer_gui.py       # Shared
│
├── chess/                      # Chess-specific
│   ├── chess_scorer.py
│   ├── chess_move_prioritizer.py
│   ├── progressive_trainer.py
│   └── ...
│
└── checkers/                   # Checkers-specific (NEW)
    ├── checkers_board.py       # Rules engine
    ├── checkers_scorer.py      # Scoring adapter
    ├── checkers_move_prioritizer.py
    ├── progressive_trainer.py
    └── headless_trainer.py
```

## Validation of Learning Philosophy

Adding checkers would **prove** the system is:
1. ✅ Not chess-specific
2. ✅ Truly learning from patterns
3. ✅ Game-agnostic architecture
4. ✅ Based on observation, not rules

**If it works for checkers:** Strong evidence the approach is sound!

## Time Estimate

| Phase | Time | Difficulty |
|-------|------|------------|
| Game engine | 1-2 days | Medium |
| Scoring adapter | 0.5 days | Easy |
| Pattern classifier | 0.5 days | Easy |
| Integration | 1 day | Medium |
| Testing | 0.5 days | Easy |
| **Total** | **3-5 days** | **Moderate** |

## Benefits

1. **Validation**: Proves learning system is game-agnostic
2. **Comparison**: Chess vs Checkers learning curves
3. **Research**: Which patterns transfer? Which are game-specific?
4. **Demonstration**: "Works for any turn-based game"
5. **Fun**: Watch AI discover checkers strategy from scratch!

## Future Games

If checkers works, could add:
- **Go** (pattern recognition on 19x19 board)
- **Othello/Reversi** (position control patterns)
- **Connect Four** (vertical pattern recognition)
- **Tic-tac-toe** (trivial, solves quickly)

All using the **same learning framework**!

## Next Steps

1. Research checkers libraries
2. Choose implementation approach
3. Create checkers/ subdirectory
4. Implement rules engine
5. Adapt scoring and patterns
6. Test with 100-game runs
7. Compare learning curves
8. Document findings

## Questions to Answer

- Does differential scoring work equally well?
- Do patterns learn faster or slower?
- Are there "universal" patterns (both games)?
- How does material advantage translate?
- Does progressive training work the same?

This would be a **excellent research project** and validation of the learning philosophy!
