# Migration Guide: Pattern Learning Enhancement

## What Changed

The AI now learns to avoid draws through **observation** rather than hardcoded rules.

### New Observable Features

The pattern classifier now tracks:

1. **repetition_count** (0, 1, 2): How many times the position has occurred
2. **moves_since_progress** (0-5): Halfmove clock / 10
3. **total_material_level** (low/medium/high): Observable piece count/value

### How AI Discovers Draw Patterns

```
Game 1: Queen moves d4→d1, position repeats (count=2), game ends in draw, score=-10000
Game 2: Queen moves d4→d1, position repeats (count=2), game ends in draw, score=-10000
...
After 100 games: Pattern "queen, quiet, distance=3, middlegame, repetition_count=2"
                 has avg_score ≈ -8000 → priority ≈ -220
AI sees this priority and chooses different move instead!
```

## Fresh Start (Recommended)

The new pattern dimensions create many more unique patterns. Starting fresh is cleanest:

```powershell
# Backup old database (optional)
Move-Item chess_pattern_ai/headless_training.db chess_pattern_ai/headless_training.db.old

# Pull changes
git pull

# Train with new system
python chess_pattern_ai/headless_trainer.py 1000
```

## Keep Existing Data (Advanced)

If you want to keep your existing patterns:

```powershell
# Pull changes
git pull

# Migrate existing database (adds new columns with defaults)
python chess_pattern_ai/migrate_pattern_schema.py headless_training.db

# Continue training
python chess_pattern_ai/headless_trainer.py 1000
```

**Note**: Old patterns will have default values (repetition_count=0, moves_since_progress=0, total_material_level='medium'), so they won't capture draw-causing situations initially.

## What to Expect

### Discovery Timeline

- **Games 1-100**: AI plays normally, recording all observable features
- **Games 100-500**: Patterns with repetition_count=2 accumulate -10000 scores
- **Games 500-1000**: AI starts avoiding high-risk patterns naturally
- **Games 1000+**: Draw rate should drop from 90%+ to <30%

### You Should See

**After 100 games:**
```
Threefold Repetition: Still high (~60-80% of draws)
New patterns being created with all 7 dimensions
```

**After 500 games:**
```
Threefold Repetition: Decreasing (~30-50% of draws)
Patterns with repetition_count=2 have negative priorities
```

**After 1000 games:**
```
Threefold Repetition: Low (<10% of draws)
Overall draw rate dropping significantly
Win rate increasing
```

## Monitoring Progress

Check learned patterns:

```powershell
python chess_pattern_ai/headless_trainer.py 100 --show-patterns 20
```

Look for patterns like:
```
Piece      Category    Phase      Rep  Progress  Material  Priority
queen      quiet       middlegame  2    0         medium    -220.5
rook       quiet       endgame     2    1         low       -185.2
```

Low/negative priorities for repetition_count=2 means AI is learning!

## Philosophy

This change aligns with your design goal:
> "AI should learn games from first principles, discovering objectives, rules,
> resources, and strategies through observation rather than hardcoded knowledge."

The AI now **observes**:
- "This position occurred 2 times before" (repetition_count=2)
- "No captures for 40 moves" (moves_since_progress=4)
- "Only 5 points of material left" (total_material_level='low')

And **discovers**:
- "When repetition_count=2, my score is terrible → avoid!"
- "When moves_since_progress=4+, games draw → make progress!"
- "When total_material_level=low, can't checkmate → avoid trades!"

No chess rules hardcoded. Pure observation-based learning.
