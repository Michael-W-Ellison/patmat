# Learnable Move Prioritization System

## Overview

Replaced hardcoded chess knowledge (center squares, back ranks) with a learning-based system that discovers which types of moves lead to wins through observation.

## Problem: Hardcoded Square Knowledge

**Previous violations in `optimized_search.py`:**

```python
# VIOLATION: Hardcoded center squares
if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
    sensible_moves.append(move)

# VIOLATION: Hardcoded back rank knowledge
if from_rank in [0, 7]:  # "back rank"
    sensible_moves.append(move)
```

These violated the learning philosophy - the system shouldn't "know" that center squares or back ranks are important.

## Solution: Learn From Outcomes

Instead of hardcoding which squares are "good", the system now:

1. **Classifies moves by observable characteristics**:
   - Piece type (pawn, knight, bishop, rook, queen, king)
   - Move category (capture, check, capture+check, quiet, development)
   - Distance from starting area (observable, not "good" or "bad")
   - Game phase (opening, middlegame, endgame)

2. **Tracks win rates for each move type**:
   - Records which move characteristics appeared in winning games
   - Calculates win rate: `wins / (wins + losses + draws)`
   - Builds confidence as more games are observed

3. **Calculates learned priorities**:
   - `priority = win_rate × confidence × 100`
   - High win rate + high confidence = high priority
   - Moves searched in priority order (best first)

## How It Works

### During Games

```python
# After each game, record all AI moves
for move in ai_moves:
    # Classify: piece=knight, category=development, distance=2, phase=opening
    characteristics = classify_move(board, move)

    # Update statistics based on game outcome
    if game_result == 'win':
        knight_development_opening.wins += 1
    # Calculate new win_rate and priority
```

### During Search

```python
# Instead of hardcoded squares:
for move in legal_moves:
    # Get LEARNED priority for this move type
    priority = get_learned_priority(board, move)

    # Include if historically successful
    if priority >= 30.0:  # Learned threshold
        search_this_move()

# Sort by learned priority (best first)
moves = sort_by_learned_priority(legal_moves)
```

## Database Schema

```sql
CREATE TABLE learned_move_patterns (
    -- Observable characteristics (NO hardcoded squares)
    piece_type TEXT,           -- 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
    move_category TEXT,        -- 'capture', 'check', 'quiet', 'development'
    distance_from_start INT,   -- How far from starting area (0-8)
    game_phase TEXT,           -- 'opening', 'middlegame', 'endgame'

    -- Outcome tracking
    times_seen INTEGER,
    games_won INTEGER,
    games_lost INTEGER,
    games_drawn INTEGER,
    win_rate REAL,             -- wins / total_games

    -- Statistical confidence
    confidence REAL,           -- min(1.0, times_seen / 50.0)

    -- Learned priority (used during search)
    priority_score REAL        -- win_rate × confidence × 100
);
```

## Example Learning

After 50 games, the system might discover:

```
Move Pattern                           Seen  Win%  Conf  Priority
--------------------------------------------------------------------
knight development opening              45   78%   0.90   70.2
bishop capture middlegame               32   65%   0.64   41.6
pawn quiet endgame                      60   42%   1.00   42.0
queen development opening                8   12%   0.16    1.9
```

**Discoveries:**
- Knights developed in opening → high win rate → prioritize
- Early queen development → low win rate → deprioritize
- Captures in middlegame → good win rate → prioritize
- Quiet pawn moves in endgame → medium win rate → medium priority

**NO hardcoded knowledge** about:
- ❌ Which squares are "center" squares
- ❌ Which ranks are "back ranks"
- ❌ What piece positions are "good" or "bad"

**ONLY observable features:**
- ✅ Piece type that moved
- ✅ Whether it's a capture/check
- ✅ Distance from starting area
- ✅ Game phase (by piece count)

## Philosophy Compliance

### ✓ What System Knows (Allowed)
- Goal: Checkmate opponent king
- Piece values: Discovered from observation
- Move legality: Learned from never seeing illegal moves

### ✓ What System Learns (Observation)
- Which move TYPES lead to wins
- Win rates for piece development
- Success rates for captures vs quiet moves
- Phase-specific move effectiveness

### ✗ What System Does NOT Know (Removed)
- ❌ Specific squares that are "important" (e.g., D4, E4)
- ❌ Positional concepts (center control, back rank)
- ❌ Opening theory (specific square preferences)

## Integration

### Files Modified

1. **`optimized_search.py`**:
   - Removed hardcoded center squares [D4, D5, E4, E5]
   - Removed hardcoded back rank logic [0, 7]
   - Added `LearnableMovePrioritizer` integration
   - Move filtering now uses learned priorities
   - Move ordering uses learned priorities

2. **`test_learning_ai_with_clustering.py`**:
   - Added `LearnableMovePrioritizer` initialization
   - Records all AI moves for learning
   - Updates move statistics after each game

3. **`fast_learning_ai.py`**:
   - Connects AI to move prioritizer
   - Displays learned move pattern statistics
   - Shows top 5 move patterns by priority

### Files Created

4. **`learnable_move_prioritizer.py`**:
   - Classifies moves by observable characteristics
   - Tracks win/loss/draw for each move type
   - Calculates learned priorities
   - Sorts moves by priority during search

## Benefits

1. **No Hardcoded Knowledge**: Discovers effective moves through observation
2. **Adapts to Opponent**: Learns what works against specific play styles
3. **Game-Specific**: Different patterns for opening/middlegame/endgame
4. **Confidence-Weighted**: Trusts patterns more after seeing them often
5. **Outcome-Driven**: Prioritizes moves that actually lead to wins

## Usage

```bash
# Run games - system learns move patterns automatically
python fast_learning_ai.py 20 --stockfish-feedback

# View learned patterns in output:
# Learned Move Patterns (Top 5):
#   knight   development   (dist=2, opening   )
#     Seen  45x | Win rate: 78.0% | Priority: 70.2
```

## Stockfish Feedback Integration

The system can optionally use Stockfish to:
- Identify blunders, mistakes, and inaccuracies
- Learn which move characteristics correlate with errors
- Discover better alternatives to failed patterns

Enable with: `--stockfish-feedback` flag

## Result

**The AI now learns move priorities from outcomes, not rules.**

- Searches moves in learned priority order
- Adapts based on win/loss results
- No assumptions about "good" squares
- Purely statistical: observable features + learned win rates
