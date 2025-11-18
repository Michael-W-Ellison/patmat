# Pattern Decay & Database Viewer

## The Problem

When training with progressive difficulty (Stockfish levels 0-20), patterns that work well at easy levels may stop working at harder levels, but the AI keeps prioritizing them because old wins influence the average forever.

### Example Problem Scenario

```
Knight development pattern:
- Games 1-100 (Level 0):  Wins, avg score +1200, priority = 87
- Games 101-200 (Level 10): Losses, avg score -800

After game 200:
  Overall avg = (100*1200 + 100*-800) / 200 = +200
  Priority = 55 (still prioritized!)

Problem: Pattern stopped working 100 games ago, but AI still uses it!
```

## Solutions

### 1. Pattern Decay Manager

Applies **exponential decay weighting** so recent games matter more than old games.

#### How It Works

```python
# Recent games weighted higher
Game N (most recent):   weight = 1.0
Game N-1:               weight = 0.95
Game N-2:               weight = 0.95² = 0.90
Game N-10:              weight = 0.95¹⁰ = 0.60
Game N-50:              weight = 0.95⁵⁰ = 0.08  (barely counts!)
```

With decay weighting, after 50 games:
- Recent 10 games: ~73% of total weight
- Old 100+ games: ~2% of total weight

**Old losses fade away. Recent performance dominates.**

#### Usage

```bash
# Recompute priorities with decay (decay rate 0.95)
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute

# Faster decay (recent games count even more)
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute --decay-rate 0.90

# Slower decay (more historical weight)
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute --decay-rate 0.98

# Show patterns that are declining
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --show-declining 10
```

#### Recommended Workflow

**During Progressive Training:**

```bash
# Train for 100 games
python3 chess_pattern_ai/progressive_trainer.py 100

# Recompute with decay
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute

# Train 100 more
python3 chess_pattern_ai/progressive_trainer.py 100

# Recompute again
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute
```

**Why this helps:**
- After AI advances from Level 0 to Level 1, old patterns get de-prioritized
- AI adapts faster to new opponent strength
- Declining patterns identified and fixed

### 2. Pattern Viewer GUI

Interactive database viewer showing:
- All learned patterns with statistics
- Filter by piece type, category, phase
- Sort by priority, win rate, games played
- Double-click pattern to see details
- Trend analysis (improving/declining)

#### Usage

```bash
# View patterns from any database
python3 chess_pattern_ai/pattern_viewer_gui.py progressive_training.db

# View headless training patterns
python3 chess_pattern_ai/pattern_viewer_gui.py headless_training.db

# View GUI training patterns
python3 chess_pattern_ai/pattern_viewer_gui.py gui_training.db
```

#### Features

**Main View:**
- **Filter**: By piece (pawn, knight, etc.) or category (capture, development, etc.)
- **Sort**: By priority, win rate, games played, or most recent
- **Statistics Bar**: Total patterns, avg priority, avg confidence, overall win rate

**Pattern Details (double-click):**
- Games played (W-L-D breakdown)
- Average differential score
- Priority (0-100)
- Confidence level (0.0-1.0)
- Last updated timestamp
- Interpretation (HIGH/MEDIUM/LOW priority, Excellent/Good/Losing pattern)

**Example Output:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Database Statistics                                             │
│ Total Patterns: 156  |  Avg Priority: 54.2  |  Win Rate: 58.3% │
├─────────────────────────────────────────────────────────────────┤
│ Piece    Category      Phase      Games  Wins  Losses  Win%    │
├─────────────────────────────────────────────────────────────────┤
│ knight   capture       opening    45     35    8       77.8%   │
│ bishop   development   opening    52     31    18      59.6%   │
│ pawn     capture       middlegame 89     48    35      53.9%   │
│ queen    quiet         endgame    12     5     7       41.7%   │
└─────────────────────────────────────────────────────────────────┘

Double-click for details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pattern: KNIGHT - Capture (opening)

Performance:
  Games: 45  |  W-L-D: 35-8-2  |  Win Rate: 77.8%
  Average Score: +823 (differential)

Learning Stats:
  Priority: 78.2 / 100
  Confidence: 0.90 / 1.00

Interpretation:
  Priority 78: HIGH - Frequently selected
  Score +823: Excellent pattern
  Confidence 0.90: High confidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Decay Rate Configuration

### Choosing Decay Rate

| Decay Rate | Description | Best For |
|------------|-------------|----------|
| 0.90 | Fast decay | Rapid opponent changes (progressive training) |
| 0.95 | Medium decay | **Recommended default** |
| 0.98 | Slow decay | Stable opponents, long-term patterns |
| 0.99 | Very slow | Historical analysis |

### Effect of Decay Rate

**With 0.90 (fast decay):**
- Last 10 games: 65% of weight
- Last 50 games: 99% of weight
- Games 100+ ago: <1% weight
- **Use when**: Opponent strength changes frequently

**With 0.95 (medium decay):**
- Last 10 games: 60% of weight
- Last 50 games: 92% of weight
- Games 100+ ago: 5% weight
- **Use when**: Standard progressive training

**With 0.98 (slow decay):**
- Last 10 games: 45% of weight
- Last 50 games: 74% of weight
- Games 100+ ago: 18% weight
- **Use when**: Want historical trends to matter

## Integration with Training

### Manual Recompute (Recommended for Learning)

```bash
# Train 200 games
python3 chess_pattern_ai/progressive_trainer.py 200

# Recompute with decay
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --recompute

# Check declining patterns
python3 chess_pattern_ai/pattern_decay_manager.py progressive_training.db --show-declining 10

# View in GUI
python3 chess_pattern_ai/pattern_viewer_gui.py progressive_training.db

# Continue training
python3 chess_pattern_ai/progressive_trainer.py 200
```

### Automatic Recompute (Future Enhancement)

Could be added to trainers to auto-recompute every N games:
```python
# In progressive_trainer.py (future)
if games_played % 100 == 0:
    decay_manager.recompute_all_priorities()
```

## Identifying Problem Patterns

### Using Pattern Viewer

1. **Open database**: `python3 pattern_viewer_gui.py progressive_training.db`
2. **Sort by Win Rate** (ascending)
3. **Look for**:
   - High games played (>50)
   - Low win rate (<40%)
   - Still high priority (>50)
4. **These are problem patterns** - old wins keeping priority high despite recent losses

### Using Decay Manager

```bash
# Show top 10 declining patterns
python3 pattern_decay_manager.py progressive_training.db --show-declining 10
```

Output:
```
TOP 10 DECLINING PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Piece      Category       Phase        Trend      Priority
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
knight     quiet          opening      -450       42.3
bishop     development    middlegame   -380       51.2
queen      capture        opening      -320       68.7  ← Problem!
```

**Interpretation:**
- Queen captures in opening: High priority (68.7) but declining (-320 trend)
- This pattern worked at low levels but fails at high levels
- After recompute with decay, priority will drop significantly

## Benefits

### With Decay Weighting:
✅ AI adapts to changing opponent strength
✅ Recent performance drives decisions
✅ Old patterns naturally fade
✅ Faster learning at new difficulty levels
✅ Identifies declining patterns automatically

### With Pattern Viewer:
✅ See exactly what AI has learned
✅ Identify problem patterns
✅ Verify learning is working
✅ Debug unexpected behavior
✅ Track improvement over time

## Example Workflow

### Training Session with Analysis

```bash
# 1. Initial training (100 games)
python3 progressive_trainer.py 100

# 2. View current patterns
python3 pattern_viewer_gui.py progressive_training.db
# (Check: Are there high-priority patterns with low win rates?)

# 3. Recompute with decay
python3 pattern_decay_manager.py progressive_training.db --recompute

# 4. View updated patterns
python3 pattern_viewer_gui.py progressive_training.db
# (Check: Did problem patterns drop in priority?)

# 5. Check declining patterns
python3 pattern_decay_manager.py progressive_training.db --show-declining 10

# 6. Continue training
python3 progressive_trainer.py 200

# 7. Repeat steps 2-6 periodically
```

## Technical Details

### Database Schema Addition

Pattern Decay Manager adds:
```sql
CREATE TABLE pattern_game_history (
    piece_type TEXT,
    move_category TEXT,
    distance_from_start INTEGER,
    game_phase TEXT,
    game_number INTEGER,      -- For tracking recency
    result TEXT,              -- 'win', 'loss', 'draw'
    score REAL,               -- Differential score
    timestamp DATETIME
);
```

This tracks **individual game results** per pattern, enabling:
- Recency weighting
- Trend analysis
- Historical performance graphs

### Compatibility

Both tools are compatible with all trainers:
- ✅ Progressive Trainer
- ✅ Headless Trainer
- ✅ GUI Trainer
- ✅ Any custom trainer using `LearnableMovePrioritizer`

## Future Enhancements

Potential additions:
- [ ] Automatic decay during training
- [ ] Pattern performance graphs over time
- [ ] Export pattern reports (CSV/PDF)
- [ ] Compare patterns across databases
- [ ] Pattern similarity detection
- [ ] Clustering of related patterns
- [ ] A/B testing of decay rates

## Conclusion

The combination of **Pattern Decay Manager** and **Pattern Viewer GUI** solves the "old wins dominate" problem and provides visibility into what the AI has learned.

**Use decay weighting** to ensure the AI adapts to changing opponents.
**Use the pattern viewer** to understand and debug the learning process.

Together, they make the learning system more adaptive and transparent!
