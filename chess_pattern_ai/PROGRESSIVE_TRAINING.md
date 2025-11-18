# Progressive Training System

## Overview

Adaptive training system that automatically increases opponent difficulty as the AI improves. The AI starts playing against Stockfish level 0 and progressively advances to higher levels.

## How It Works

### Adaptive Difficulty Scaling

1. **Start at Level 0**: AI begins against weakest Stockfish (skill level 0)
2. **Track Consecutive Wins**: System counts wins in a row
3. **Auto-Promotion**: After 10 consecutive wins, AI advances to next level
4. **Reset on Loss/Draw**: Losing or drawing resets the win streak
5. **Continue Learning**: AI keeps learning patterns at all levels

### Promotion Requirements

```
Level 0 → Level 1:   10 consecutive wins
Level 1 → Level 2:   10 consecutive wins
...
Level 19 → Level 20: 10 consecutive wins
```

**Why consecutive wins?**
- Ensures consistent performance, not lucky wins
- Forces AI to master each level before advancing
- Draws and losses reset streak (must prove dominance)

## Usage

### Basic Training

```bash
# Train with default settings (max 1000 games)
python3 chess_pattern_ai/progressive_trainer.py

# Train for specific number of games
python3 chess_pattern_ai/progressive_trainer.py 500

# Verbose mode (show all moves)
python3 chess_pattern_ai/progressive_trainer.py 100 --verbose
```

### Advanced Options

```bash
# Specify Stockfish path
python3 chess_pattern_ai/progressive_trainer.py 200 --stockfish /usr/local/bin/stockfish

# Start at higher level (resume training)
python3 chess_pattern_ai/progressive_trainer.py 300 --start-level 5

# Require more wins for promotion (harder)
python3 chess_pattern_ai/progressive_trainer.py 500 --wins-needed 20

# Export results and show patterns
python3 chess_pattern_ai/progressive_trainer.py 200 --export --show-patterns 15
```

### Full Command Reference

```bash
python3 chess_pattern_ai/progressive_trainer.py [MAX_GAMES] [OPTIONS]

Positional Arguments:
  MAX_GAMES           Maximum number of games (default: 1000)

Options:
  --stockfish PATH    Path to Stockfish binary (default: /usr/games/stockfish)
  --start-level N     Starting Stockfish level (default: 0)
  --wins-needed N     Consecutive wins to advance (default: 10)
  -v, --verbose       Show detailed game output
  --export            Export results to JSON
  --show-patterns N   Show top N patterns after training
  -h, --help          Show help message
```

## Example Output

### Training Session

```
================================================================================
PROGRESSIVE TRAINING - Adaptive Difficulty vs Stockfish
================================================================================
Starting at level 0
Promotion requirement: 10 consecutive wins
Max games: 100
================================================================================

  Game 10: WIN    | Score:  +1350 | Rounds: 18
  Level 0: 7W-2L-1D | Win Rate: 70.0% | Streak: 4/10

  Game 20: WIN    | Score:  +1450 | Rounds: 15
  Level 0: 14W-4L-2D | Win Rate: 70.0% | Streak: 9/10

  Game 23: WIN    | Score:  +1520 | Rounds: 12
  Level 0: 17W-4L-2D | Win Rate: 73.9% | Streak: 10/10

  🎉 PROMOTED TO LEVEL 1! 🎉
  Conquered level 0 with 10 consecutive wins!

  Game 30: LOSS   | Score:   -850 | Rounds: 28
  Level 1: 4W-3L-0D | Win Rate: 57.1% | Streak: 0/10

================================================================================
TRAINING SUMMARY
================================================================================

Time: 125.3s (1.25s per game)
Total Games: 100
Current Level: 1 / 20
Current Streak: 6/10

--------------------------------------------------------------------------------
PERFORMANCE BY LEVEL
--------------------------------------------------------------------------------
Level    Games    W-L-D        Win%     Avg Score    Best
--------------------------------------------------------------------------------
0        23       17-4-2       73.9%    +892         +1520
1        77       48-22-7      62.3%    +423         +1650

--------------------------------------------------------------------------------
LEARNING PROGRESS
--------------------------------------------------------------------------------
Patterns Learned: 156
Average Confidence: 0.742
Average Win Rate: 64.8%

================================================================================
```

## Stockfish Level Progression

### Level Characteristics

| Level | Strength | Description | Expected AI Win Rate |
|-------|----------|-------------|---------------------|
| 0 | Very Weak | Random-like moves | ~80% (early) |
| 1-3 | Beginner | Basic tactics | ~65% |
| 4-7 | Intermediate | Good tactics, some strategy | ~50% |
| 8-12 | Advanced | Strong tactical + positional | ~30% |
| 13-16 | Expert | Near-master level | ~15% |
| 17-20 | Master+ | Championship level | <5% |

### Training Milestones

**Beginner Phase (Levels 0-3)**
- AI learns basic patterns
- Discovers tactical motifs (forks, pins)
- Fast games, clear mistakes

**Intermediate Phase (Levels 4-7)**
- AI refines exchange evaluation
- Learns positional patterns
- Longer games, fewer blunders

**Advanced Phase (Levels 8-12)**
- AI discovers strategic patterns
- Complex pattern interactions
- Requires deep learning

**Expert Phase (Levels 13+)**
- Rare promotions (weeks/months)
- Discovers subtle patterns
- Meta-pattern recognition

## Training Strategies

### Quick Testing (100-200 games)
```bash
python3 chess_pattern_ai/progressive_trainer.py 200 --verbose
```
- Good for: Testing the system
- Expected: Reach level 1-2
- Time: ~3-5 minutes

### Standard Training (500-1000 games)
```bash
python3 chess_pattern_ai/progressive_trainer.py 1000
```
- Good for: Initial learning
- Expected: Reach level 3-5
- Time: ~20-30 minutes

### Extended Training (5000+ games)
```bash
python3 chess_pattern_ai/progressive_trainer.py 5000 --export
```
- Good for: Advanced patterns
- Expected: Reach level 6-10
- Time: 2-3 hours

### Marathon Training (Overnight)
```bash
nohup python3 chess_pattern_ai/progressive_trainer.py 50000 --export > training.log 2>&1 &
```
- Good for: Expert-level learning
- Expected: Reach level 10-15
- Time: 12+ hours

## Understanding the Results

### Differential Scores

**Against Stockfish:**
- **+1500+**: Crushing win (AI dominated)
- **+1000 to +1500**: Solid win
- **+500 to +1000**: Close win
- **-500 to +500**: Draw or close loss
- **-1000 to -500**: Clear loss
- **-1500**: Crushed by Stockfish

**Score progression as levels increase:**
- Level 0-3: Avg +800 (clear wins)
- Level 4-7: Avg +200 (close games)
- Level 8+: Avg -200 (mostly losses)

### Win Rate Expectations

**Healthy progression:**
```
Level 0:  70-80% win rate → Promote quickly
Level 1:  60-70% win rate → Steady progress
Level 2:  55-65% win rate → Slowing down
Level 3:  50-60% win rate → Harder fights
Level 4:  45-55% win rate → Real challenge
```

**Warning signs:**
- Win rate <30% for 50+ games → Stuck, may not advance
- Win rate >90% → Consider increasing --wins-needed
- Many draws → AI learning defensive play (good!)

## Database Persistence

Training data is stored in `progressive_training.db`:
- All learned patterns
- Move priorities and statistics
- Compatible with other trainers

### Resume Training

To continue from where you left off:
```bash
# Check current level from previous run
# Then start from that level
python3 chess_pattern_ai/progressive_trainer.py 1000 --start-level 5
```

### Reset Training

To start fresh:
```bash
rm chess_pattern_ai/progressive_training.db
python3 chess_pattern_ai/progressive_trainer.py 1000
```

## Export & Analysis

### Export Results

```bash
python3 chess_pattern_ai/progressive_trainer.py 500 --export
```

Creates `progressive_training_YYYYMMDD_HHMMSS.json` with:
- Summary statistics
- Performance by level
- Score history
- Level progression timeline

### View Top Patterns

```bash
python3 chess_pattern_ai/progressive_trainer.py 200 --show-patterns 20
```

Shows:
- Top 20 learned patterns
- Win rates per pattern
- Confidence levels
- Priority scores

## Tips for Success

### 1. Be Patient
- Level 0→1: Fast (20-50 games)
- Level 1→2: Moderate (50-150 games)
- Level 2→3: Slower (100-300 games)
- Each level requires exponentially more games

### 2. Monitor Progress
- Check win rates regularly
- If stuck at a level for 200+ games, AI may have plateaued
- Consider adding more search depth or better evaluation

### 3. Export Data
- Use `--export` every 500-1000 games
- Analyze score trends
- Identify which levels cause difficulties

### 4. Adjust Requirements
```bash
# Make promotion easier (faster training)
python3 chess_pattern_ai/progressive_trainer.py 1000 --wins-needed 5

# Make promotion harder (more thorough learning)
python3 chess_pattern_ai/progressive_trainer.py 1000 --wins-needed 20
```

## Comparison to Other Training

### vs Random Opponent
- ✅ Progressive: Adaptive difficulty, continuous challenge
- ❌ Random: Fixed difficulty, plateaus quickly

### vs Fixed Stockfish Level
- ✅ Progressive: Always appropriate challenge
- ❌ Fixed: Too easy or too hard, inefficient learning

### vs Headless Trainer
- ✅ Progressive: Better long-term performance
- ✅ Headless: Faster initial learning
- **Recommendation**: Use both!
  1. Headless for first 100-200 games (quick patterns)
  2. Progressive for advanced learning (500+ games)

## Troubleshooting

### "Error initializing Stockfish"
```bash
# Check Stockfish location
which stockfish

# Use correct path
python3 chess_pattern_ai/progressive_trainer.py 100 --stockfish $(which stockfish)
```

### "Stockfish not found"
```bash
# Install Stockfish
sudo apt-get install stockfish  # Ubuntu/Debian
brew install stockfish          # macOS
```

### AI stuck at level (200+ games, no promotion)
- This is normal for higher levels!
- AI may need 500-1000+ games to master difficult levels
- Consider:
  - Reducing --wins-needed to 5
  - Training longer (5000+ games)
  - Accepting current level as "good enough"

### Games too slow
- Reduce Stockfish thinking time (edit `play_game_vs_stockfish`)
- Use faster hardware
- Run overnight training

## Future Enhancements

Potential additions:
- [ ] Dynamic wins-needed based on level
- [ ] Automatic time control adjustment
- [ ] Multi-engine support (Lc0, Komodo)
- [ ] Opening book integration
- [ ] Endgame tablebase support
- [ ] GUI integration
- [ ] Parallel game execution
- [ ] Cloud training support

## Philosophy

Progressive training embodies the learning system's core philosophy:

**"Learn through appropriate challenge"**

- Too easy → no learning
- Too hard → random guessing
- Just right → pattern discovery

By automatically adjusting difficulty, the AI always faces opponents that are:
- Challenging enough to require learning
- Beatable enough to provide positive reinforcement
- Consistent enough to validate pattern quality

This is how the AI discovers chess patterns from observation, not hardcoded knowledge!
