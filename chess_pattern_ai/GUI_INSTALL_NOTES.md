# GUI Installation Notes

## GUI Application (`ai_gui.py`)

The full GUI provides:
- Visual chess board display
- Real-time metrics and graphs
- Interactive training controls
- Pattern learning visualization

### Requirements

```bash
pip install matplotlib Pillow cairosvg --user
```

**Important**: The GUI also requires **tkinter**, which is not available via pip.

### Installing tkinter

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-tk
```

#### Fedora/RHEL
```bash
sudo dnf install python3-tkinter
```

#### macOS
```bash
# Usually included with Python installation
# If not, install via Homebrew:
brew install python-tk
```

#### Windows
- tkinter is usually included with Python installations from python.org
- If missing, reinstall Python with "tcl/tk and IDLE" option checked

### Verify Installation

```bash
python3 -c "import tkinter; print('✓ tkinter available')"
```

## Alternative: Headless Trainer

If tkinter is not available or you prefer command-line operation, use the **headless trainer**:

```bash
# Train for 50 games
python3 chess_pattern_ai/headless_trainer.py 50

# With detailed output
python3 chess_pattern_ai/headless_trainer.py 20 --verbose

# Export results
python3 chess_pattern_ai/headless_trainer.py 100 --export-json --export-csv

# Show top patterns after training
python3 chess_pattern_ai/headless_trainer.py 50 --show-patterns 10
```

### Headless Trainer Features

✓ **No GUI required** - runs in terminal
✓ **Same learning system** - uses differential scoring and pattern learning
✓ **Progress tracking** - real-time stats during training
✓ **Export results** - JSON and CSV export
✓ **Pattern inspection** - view top learned patterns
✓ **Fast** - ~0.1s per game

### Example Usage

```bash
# Quick test (5 games, show all moves)
python3 chess_pattern_ai/headless_trainer.py 5 --verbose

# Standard training (100 games, updates every 10 games)
python3 chess_pattern_ai/headless_trainer.py 100

# Long training with exports
python3 chess_pattern_ai/headless_trainer.py 500 \
  --progress 50 \
  --export-json \
  --show-patterns 20
```

### Output Example

```
======================================================================
HEADLESS TRAINING - 50 GAMES
======================================================================

--- Game 10/50 (AI plays WHITE) ---
  Result: WIN    | Score:  +1285 | Rounds: 25
  Progress: 6W-3L-1D | Win Rate: 60.0% | Avg Score: +450

======================================================================
TRAINING COMPLETE
======================================================================
Time: 4.2s (0.08s per game)
Results: 32W-14L-4D
Win Rate: 64.0%
Average Score: +542

Learning Statistics:
  Patterns Learned: 47
  Avg Confidence: 0.68
  Avg Win Rate: 58.2%

TOP 10 LEARNED PATTERNS (by priority)
----------------------------------------------------------------------
knight     capture        opening      45    78.0%  0.90   70.2
```

## Choosing Between GUI and Headless

### Use GUI when:
- You want visual feedback
- Training small batches (1-50 games)
- Demonstrating the AI
- Analyzing individual games

### Use Headless when:
- tkinter not available
- Training large batches (100-1000+ games)
- Running on servers/headless systems
- Automating training runs
- Exporting data for analysis

## Both Systems

Both the GUI and headless trainer:
- Use the same differential scoring system
- Learn the same patterns
- Store data in compatible databases
- Can be used interchangeably

The only difference is the interface - the learning algorithm is identical!
