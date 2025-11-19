# Pattern Recognition AI - Multi-Game System

## Overview

This system implements observation-based pattern learning for 13 classic board games. The AI learns to play games by discovering patterns through self-play, without hardcoded evaluation rules.

## Supported Games

The system includes 13 board games, each demonstrating different pattern learning challenges:

1. **Chess** - Complex tactics and strategy
2. **Checkers** - Forced captures and king dynamics
3. **Go (9×9)** - Territory control and influence
4. **Othello** - Disc flipping and stable pieces
5. **Connect Four** - Vertical threat patterns
6. **Gomoku** - 5-in-a-row patterns
7. **Hex** - Connection and blocking
8. **Dots and Boxes** - Chain reactions and parity
9. **Breakthrough** - Pawn racing game
10. **Pentago** - Rotation-based 5-in-a-row (AI struggles)
11. **Nine Men's Morris** - Multi-phase mill formation
12. **Lines of Action** - Connectivity with unique movement
13. **Arimaa** - Ultimate AI challenge (designed to be hard for computers)

### Core Architecture

1. **Game-Specific Components** (per game)
   - `{game}_board.py` - Board representation and rules
   - `{game}_game.py` - Game engine and move generation
   - `{game}_scorer.py` - Differential scoring and move categorization
   - `{game}_headless_trainer.py` - Training loop and statistics

2. **Shared Learning System**
   - `learnable_move_prioritizer.py` - Universal pattern learning engine
   - SQLite database per game - Stores learned patterns with win rates
   - Differential scoring - `score = my_advantage - opponent_advantage`
   - Pattern categorization - Game-specific move categories

3. **GUI Tools**
   - `game_launcher_gui.py` - Launch training for all games (requires tkinter)
   - `pattern_viewer_gui.py` - Visualize learned patterns (requires tkinter)
   - `chess_pattern_ai_gui.py` - Chess game viewer with board display

4. **Testing**
   - `test_all_games.py` - Automated tests for all 13 games

### Key Features

1. **Observation-Based Learning**
   - AI discovers patterns through self-play, no hardcoded rules
   - Learns from game outcomes: win/loss/draw for each pattern
   - Adapts move priorities based on success rates

2. **Game-Agnostic Architecture**
   - 85% code reuse across all games
   - Shared pattern learning engine
   - Each game defines its own move categories and scoring

3. **Differential Scoring**
   - `score = my_advantage - opponent_advantage`
   - Game-specific evaluation metrics
   - Pattern categorization for targeted learning

4. **Compact Pattern Databases**
   - SQLite database per game stores learned patterns
   - Tracks: category, distance, phase, win rate, times seen
   - No position memorization, only abstract patterns

## Usage

### Quick Start with GUI (requires tkinter)
```bash
# Launch the game trainer GUI
python3 game_launcher_gui.py

# View learned patterns for a game
python3 pattern_viewer_gui.py chess_training.db
```

### Command Line Training

Train any game from command line:

```bash
# Chess (50 games)
python3 chess_progressive_trainer.py 50

# Checkers (100 games)
python3 checkers/checkers_headless_trainer.py 100

# Go 9×9 (50 games)
python3 go/go_headless_trainer.py 50 --size 9

# Othello (100 games)
python3 othello/othello_headless_trainer.py 100

# Connect Four (200 games)
python3 connect4/connect4_headless_trainer.py 200

# Gomoku (100 games)
python3 gomoku/gomoku_headless_trainer.py 100

# Hex (100 games)
python3 hex/hex_headless_trainer.py 100

# Dots and Boxes (200 games)
python3 dots_boxes/dots_boxes_headless_trainer.py 200

# Breakthrough (100 games)
python3 breakthrough/breakthrough_headless_trainer.py 100

# Pentago (100 games)
python3 pentago/pentago_headless_trainer.py 100

# Nine Men's Morris (50 games)
python3 morris/morris_headless_trainer.py 50

# Lines of Action (50 games)
python3 loa/loa_headless_trainer.py 50

# Arimaa (20 games)
python3 arimaa/arimaa_headless_trainer.py 20
```

### Show Learned Patterns

```bash
# Show top 10 patterns for any game
python3 {game}/{game}_headless_trainer.py --show-patterns 10
```

### Run Full Test Suite

```bash
# Test all 13 games
python3 test_all_games.py
```

## Requirements

### Core (all games)
- Python 3.8+
- sqlite3 (built-in)
- random (built-in)

### Chess only
- python-chess

### GUI tools (optional)
- tkinter (for game_launcher_gui.py and pattern_viewer_gui.py)

Most games have zero external dependencies beyond Python standard library!

## Game Highlights

### Games Where Traditional AI Struggles

**Pentago** - Rotation mechanics make traditional search difficult
- AI learns "rotation trap" patterns (92-96% win rate)
- Discovers that rotating after placement creates winning positions

**Nine Men's Morris** - Multi-phase gameplay
- AI learns mill formation (100% win rate on 3-in-a-row patterns)
- Discovers 2-piece setup patterns (100% win rate)

**Lines of Action** - Complex connectivity requirements
- AI learns piece grouping (94.1% win rate in endgame)
- Discovers that keeping pieces together is critical

**Arimaa** - Specifically designed to be hard for computers
- Branching factor of 16^4 (17,000+ possible moves per turn)
- AI learns rabbit advancement strategies (49.8% success)
- Discovers push/pull tactics and trap control

## Project Structure

```
chess_pattern_ai/
├── game_launcher_gui.py          # Launch training for all games (GUI)
├── pattern_viewer_gui.py          # View learned patterns (GUI)
├── test_all_games.py              # Test all 13 games
├── learnable_move_prioritizer.py  # Shared learning engine
│
├── {game}/                        # Per-game directories:
│   ├── {game}_board.py           # Board representation
│   ├── {game}_game.py            # Game engine
│   ├── {game}_scorer.py          # Scoring and categorization
│   └── {game}_headless_trainer.py # Training loop
│
└── *.db                          # Pattern databases (gitignored)
```

## Version

Pattern Recognition AI v3.0 - Multi-Game System
- 13 games implemented
- Observation-based learning
- Game-agnostic architecture
- GUI launcher for all games
