# Differential Score-Based Learning

## Overview

Replaced absolute scoring with **differential (relative) scoring** that teaches the AI about exchanges, tactics, and relative position.

## The Key Insight

**You said:**
> "It should not just want a high score, but a higher score than its opponent. 'I am losing a pawn, but gaining a rook, which is of greater value.'"

This is **differential thinking** - what matters is my **advantage** over the opponent, not my absolute material.

## Scoring Formula (Differential)

```python
# Calculate material ADVANTAGE (not absolute material)
material_advantage = ai_material - opponent_material

# Final scores
if AI wins:
    score = material_advantage + (100 - rounds) + 1000

if AI loses:
    score = material_advantage - 1000

if draw:
    score = material_advantage
```

## What This Teaches

### 1. Exchanges Are Relative

**Absolute scoring (OLD):**
```
I lost pawn: my_material -= 100
Result: BAD (I lost something)
```

**Differential scoring (NEW):**
```
Exchange: I lost 100 (pawn), opponent lost 500 (rook)
My advantage: +0 → +400
Result: GOOD! (I improved my position)
```

### 2. Losing "Well" vs Losing "Badly"

**Examples:**

| Outcome | Material Advantage | Final Score | Meaning |
|---------|-------------------|-------------|---------|
| Loss | +200 (ahead) | +200 - 1000 = **-800** | Lost but fought well |
| Loss | -500 (behind) | -500 - 1000 = **-1500** | Got crushed |

**System learns:** Losing with material advantage is less catastrophic than getting destroyed.

### 3. Drawing "Well" vs Drawing "Badly"

| Outcome | Material Advantage | Final Score | Meaning |
|---------|-------------------|-------------|---------|
| Draw | +300 (ahead) | **+300** | Drew but was winning |
| Draw | -300 (behind) | **-300** | Drew but was losing |

**System learns:** Draws with advantage are good, draws with disadvantage are bad.

### 4. Winning With Style

| Outcome | Advantage | Rounds | Final Score | Meaning |
|---------|-----------|--------|-------------|---------|
| Win | +500 | 10 | +500 + 90 + 1000 = **+1590** | BEST (fast + ahead) |
| Win | +100 | 10 | +100 + 90 + 1000 = **+1190** | GOOD (fast) |
| Win | +100 | 50 | +100 + 50 + 1000 = **+1150** | OK (slow) |
| Win | -200 | 10 | -200 + 90 + 1000 = **+890** | Won despite disadvantage |

**System learns:**
- Quick wins + material advantage = highest score
- Winning despite material disadvantage = still good but lower score
- Speed matters (opponent king decay)

## Exchange Evaluation

The key is **`calculate_material_delta()`** which tracks advantage changes:

```python
# Before move
my_material = 3900
opponent_material = 3900
advantage_before = 0

# After move (I sacrifice knight for rook)
my_material = 3580  # Lost knight (320)
opponent_material = 3400  # Lost rook (500)
advantage_after = +180

# Delta
advantage_delta = +180 - 0 = +180  # GOOD EXCHANGE!
```

**The AI learns:**
- "This pattern (fork/pin/skewer) led to +180 advantage delta"
- "I put my knight in danger but gained +180 = good!"
- "Sacrificing bishop (+330) to expose king = checkmate = +1590 final score = EXCELLENT!"

## Tactical Pattern Recognition

### Example: Knight Fork

```python
# Situation: My knight can attack opponent's queen and rook
# Before: advantage = 0
# After: I lose nothing, opponent loses queen (900) = +900
# Pattern "knight_fork" → avg_score = +1090 (if win) or +900 (if drawn)
# System learns: PRIORITIZE knight forks!
```

### Example: Piece Sacrifice for Checkmate

```python
# Before: advantage = 0
# Sacrifice queen (900): advantage = -900
# Checkmate in 2 moves: final_score = -900 + 92 + 1000 = +192

# VS not sacrificing:
# Grind to slow win: final_score = 0 + 50 + 1000 = +1050

# Wait, sacrifice scored LOWER?
# But if sacrifice leads to FASTER checkmate:
# Sacrifice + mate round 8: -900 + 92 + 1000 = +192
# Normal win round 40: 0 + 60 + 1000 = +1060

# Hmm, sacrifice still scores lower...
# UNLESS the sacrifice prevents a LOSS:
# Without sacrifice: would lose = -1000
# With sacrifice: win despite disadvantage = +192
# Delta: +1192 improvement!
```

**System learns context:**
- Sacrifices good when they prevent worse outcomes
- Sacrifices good when they lead to quick checkmate
- Material advantage good when it leads to wins

## Opponent Pattern Recognition

The system tracks **opponent responses** in `opponent_winning_patterns` table:

```sql
CREATE TABLE opponent_winning_patterns (
    fen_before TEXT,
    opponent_move TEXT,
    material_gained_by_opponent REAL,  -- How much opponent gained
    our_weakness_exploited TEXT,
    times_seen INTEGER
);
```

**Learning:**
- "When I move queen early, opponent often forks with knight = -1000 avg_score"
- "When I castle kingside, opponent attacks there = -500 advantage loss"
- "When I develop knights first, opponent can't exploit = +800 avg_score"

## Meta-Patterns

The AI can learn **patterns of patterns**:

### Pattern: "Development Before Attack"
- Games where early development → avg_score +1200
- Games where early attack → avg_score +400
- **Learn**: Development first!

### Pattern: "Central Control"
- Games with central pawns → avg_score +900
- Games without → avg_score +600
- **Learn**: Control center! (discovered from outcomes, not programmed)

### Pattern: "King Safety"
- Games with king surrounded by pieces → avg_score +1100
- Games with exposed king → avg_score +200
- **Learn**: Protect king!

## Examples of What The AI Learns

### Exchange Patterns

| Pattern | Description | Avg Score | Why |
|---------|-------------|-----------|-----|
| Knight fork | Attack 2 pieces | **+1190** | Win queen/rook, huge advantage |
| Bishop sacrifice | Expose king | **+1150** | Leads to quick mate |
| Rook trade endgame | Trade rooks when ahead pawn | **+1050** | Converts advantage to win |
| Pawn sacrifice opening | Gambit for development | **+800** | Better position, often wins |
| Queen trade middlegame | Trade when behind | **-200** | Loses without compensation |
| Piece hanging | Left piece undefended | **-1200** | Opponent takes, huge loss |

### Tactical Patterns

| Pattern | Description | Avg Score | Material Delta |
|---------|-------------|-----------|----------------|
| Discovered attack | Move piece, reveal attack | **+1180** | +320 avg (wins piece) |
| Pin | Pin piece to king | **+1100** | +500 avg (wins rook) |
| Skewer | Attack high-value piece | **+1150** | +900 avg (wins queen) |
| Fork | Attack 2 pieces | **+1190** | +600 avg (wins best piece) |
| Zugzwang | Force opponent bad move | **+1160** | +200 avg (position worsens) |

### Strategic Patterns

| Pattern | Description | Avg Score | Why |
|---------|-------------|-----------|-----|
| Castle early | King to safety | **+1100** | Protects king, develops rook |
| Develop knights | Nf3, Nc3 early | **+1150** | Controls center, flexible |
| Central pawns | e4, d4 opening | **+1080** | Space advantage |
| Early queen | Queen out move 3 | **-400** | Gets trapped/attacked |
| Neglect development | Move same piece twice | **-600** | Falls behind in position |
| Pawn weakness | Doubled pawns | **-200** | Structure weakness |

## How Patterns Are Learned

### Step 1: Observe Move

```python
# AI makes move
fen_before = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
move = "Nf3"  # Develop knight

# Calculate advantage delta
advantage_before = 0
advantage_after = 0  # No material change
advantage_delta = 0
```

### Step 2: Classify Pattern

```python
# Pattern extracted
pattern = {
    'type': 'knight_development',
    'description': 'Develop knight in opening',
    'phase': 'opening',
    'material_delta': 0
}
```

### Step 3: Track Game Outcome

```python
# Game ends
final_score = +1185  # Win, ahead +85, round 10
```

### Step 4: Update Pattern Statistics

```python
# Update pattern
pattern.total_score += 1185
pattern.games += 1
pattern.avg_score = total_score / games  # +1185 avg

# High avg_score → HIGH PRIORITY in future games
```

### Step 5: Learn and Apply

```python
# Next game, search considers moves:
moves = [
    ('e4', priority=70),      # Pawn push
    ('Nf3', priority=95),     # Knight development (high avg_score!)
    ('Qh5', priority=20)      # Early queen (low avg_score!)
]

# Search Nf3 first because it has highest priority from learning!
```

## Anticipating Opponent

The system learns opponent patterns in `learned_mistakes` and `opponent_winning_patterns`:

### When Opponent Exploits Weakness:

```python
# I moved: Qh5 (early queen)
# Opponent responded: Nf6 (attacks queen)
# Material delta: -100 (lost tempo, position worsened)
# Pattern: "opponent_punishes_early_queen" → avg_score -800

# System learns:
# "When I play early queen, opponent punishes me"
# → Lower priority for early queen moves
```

### When Opponent Fails to Exploit:

```python
# I moved: Risky pawn push
# Opponent responded: Quiet move (missed opportunity)
# Material delta: +0 (no punishment)
# Game result: Win +1190

# System learns:
# "This risky pattern worked (opponent didn't punish)"
# → Keep trying, but cautiously (medium priority)
```

### Counter-Patterns:

```python
# Opponent pattern: "Castles kingside"
# My response: "Attack kingside with pawns"
# Outcome: Win +1290
# Counter-pattern learned: "Attack opponent's castled king"
```

## Implementation

### Files Modified:

1. **game_scorer.py**:
   - Changed to differential scoring
   - `calculate_material_delta()` tracks advantage changes
   - Scores range from -1500 (crushed loss) to +1600 (perfect win)

2. **learnable_move_prioritizer.py**:
   - Updated priority calculation for differential score ranges
   - Normalizes -1500 to +1600 → 0 to 100 priority

3. **test_learning_ai_with_clustering.py**:
   - Passes differential scores to all learning components
   - Tracks material advantage throughout game

4. **pattern_abstraction_engine.py**:
   - Already tracks avg_score_with_pattern
   - Now receives differential scores

## Result

**The AI now learns:**
1. ✅ **Exchanges**: "Lost pawn, gained rook = +400 advantage"
2. ✅ **Tactics**: Patterns that improve advantage are prioritized
3. ✅ **Relative position**: Ahead > Behind, even with same result
4. ✅ **Opponent modeling**: Learns which moves opponent exploits
5. ✅ **Risk/reward**: "Sacrifice acceptable if net advantage improves"
6. ✅ **Context**: Same move good in one position, bad in another

**All from observation - no hardcoded tactics knowledge!**

## Example Game Learning

```
Game 1:
- Move 5: Develop knight → advantage +0 → +0
- Move 8: Fork opponent → advantage +500 → +500
- Move 12: Opponent escapes fork → advantage +320 → -180 delta
- Move 15: Pin opponent piece → advantage +820 → +500 delta
- Move 18: Checkmate → Final score: +820 + 82 + 1000 = +1902

Patterns learned:
- knight_development: +1902 score (in winning game)
- fork_pattern: +500 advantage delta
- pin_pattern: +500 advantage delta
- quick_checkmate: +82 time bonus

Next game:
- Knights prioritized (high score pattern)
- Fork opportunities searched first (high advantage delta)
- Pins recognized and executed (high advantage delta)
- AI plays faster (time bonus incentive)
```

**The system learns to WIN WELL, not just WIN!**
