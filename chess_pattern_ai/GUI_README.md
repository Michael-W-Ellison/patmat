# Chess Pattern Recognition AI - GUI Dashboard

## Overview

Interactive GUI for training and monitoring the chess pattern recognition AI that learns through differential scoring and pattern observation.

## Features

### 📊 Real-Time Visualization
- **Live Chess Board**: Visual representation of the current game position
- **Move History**: Scrollable list of all moves played
- **Material Advantage Tracker**: Real-time differential scoring display

### 📈 Training Metrics
- **Game Statistics**: Wins, losses, draws
- **Differential Score Graphs**: Track learning progress over time
- **Cumulative Results**: Visualize win/loss/draw trends
- **Pattern Learning Stats**: Number of patterns learned, confidence levels

### 🎮 Controls

#### Single Game Mode
- **Start Game**: Play a single test game
- **Reset Board**: Clear the board and start fresh

#### Training Mode
- **Number of Games**: Configure how many games to train (1-1000+)
- **Start Training**: Begin automated training session
- **Stop Training**: Halt training at any time

### 📚 Learning Statistics
- **Patterns Learned**: Total number of move patterns discovered
- **Average Confidence**: Statistical confidence in learned patterns
- **Win Rate Tracking**: Performance trends over time

## Installation

### Prerequisites

```bash
# Install GUI dependencies
pip install matplotlib Pillow cairosvg --user
```

All dependencies are listed in `gui_requirements.txt`

### Verify Installation

```bash
python3 -c "import tkinter; import matplotlib; import PIL; import cairosvg; print('✓ All GUI dependencies installed')"
```

## Usage

### Launch the GUI

```bash
python3 chess_pattern_ai/ai_gui.py
```

Or make it executable:

```bash
chmod +x chess_pattern_ai/ai_gui.py
./chess_pattern_ai/ai_gui.py
```

### Quick Start

1. **Single Game**:
   - Click "▶ Start Game"
   - Watch the AI play against a random opponent
   - See real-time material advantage and move history

2. **Training Session**:
   - Enter number of games (e.g., 10, 50, 100)
   - Click "🎓 Start Training"
   - Monitor progress in real-time
   - View learning graphs update automatically

3. **Analyze Results**:
   - Check cumulative win/loss/draw statistics
   - View differential score trends
   - See how patterns improve over time

## GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Chess Pattern Recognition AI - Dashboard                  │
├────────────────────────┬────────────────────────────────────┤
│                        │  Controls                          │
│   Chess Board          │  ▶ Start Game  ⟳ Reset Board      │
│   (Visual Display)     │                                    │
│                        │  Training: [10 games]              │
│   r n b q k b n r     │  🎓 Start  ⏹ Stop                  │
│   p p p p p p p p     │                                    │
│   . . . . . . . .     ├────────────────────────────────────┤
│   . . . . . . . .     │  Live Metrics                      │
│   . . . . . . . .     │  Games: 15    Wins: 10            │
│   . . . . . . . .     │  Losses: 3    Draws: 2            │
│   P P P P P P P P     │                                    │
│   R N B Q K B N R     │  Material Advantage: +580         │
│                        │  Patterns Learned: 42             │
├────────────────────────┤                                    │
│  Move History          ├────────────────────────────────────┤
│                        │  Training Progress                 │
│  1. e4 e5             │  [Score Graph]                    │
│  2. Nf3 Nc6           │  [Win/Loss Trend]                 │
│  3. Bc4 ...           │                                    │
│                        │                                    │
└────────────────────────┴────────────────────────────────────┘
```

## Understanding the Metrics

### Differential Score
- **Positive scores**: AI is ahead (material advantage + performance)
- **High scores (>1000)**: Winning games with good material advantage
- **Negative scores (<-500)**: Losing games
- **Score ranges**:
  - `+1500 to +1600`: Excellent win (quick + ahead)
  - `+1000 to +1200`: Good win
  - `+300 to -300`: Draw
  - `-800`: Loss (fought well)
  - `-1500`: Loss (crushed)

### Material Advantage
- Real-time calculation of `(AI material - Opponent material)`
- **Green**: Ahead in material
- **Red**: Behind in material
- **Black**: Even material

### Pattern Learning
- **Patterns Learned**: Unique move types AI has observed
- **Confidence**: Statistical reliability (0.0 to 1.0)
  - Increases with more observations
  - Max confidence at 50+ games per pattern

## Training Recommendations

### Initial Training (0-50 games)
- Start with 10-20 games
- AI learns basic patterns:
  - Captures are good
  - Development moves
  - Checks create pressure

### Intermediate Training (50-200 games)
- Run sessions of 50 games
- AI refines priorities:
  - Knight forks
  - Pin/skewer tactics
  - Exchange evaluation

### Advanced Training (200+ games)
- Sessions of 100-500 games
- AI discovers meta-patterns:
  - Opening strategies
  - Endgame techniques
  - Position-specific tactics

## Database

Training data is stored in `gui_training.db`:
- Move patterns and priorities
- Statistical confidence
- Historical game scores

To reset learning:
```bash
rm chess_pattern_ai/gui_training.db
```

## Troubleshooting

### GUI won't start
```bash
# Check tkinter
python3 -c "import tkinter; tkinter.Tk()"

# Check other dependencies
python3 -c "import matplotlib, PIL, cairosvg"
```

### Board not displaying
- cairosvg requires Cairo graphics library
- Falls back to text board if SVG rendering fails

### Slow performance
- Reduce number of training games
- Close resource-intensive applications
- Training runs in background thread

## Technical Details

### Architecture
- **Main Thread**: GUI updates and display
- **Worker Thread**: Game playing and training
- **Queue Communication**: Thread-safe updates

### Learning System Integration
- Uses `GameScorer` for differential scoring
- Uses `LearnableMovePrioritizer` for move selection
- Real-time pattern database updates

### Visualization
- **Chess Board**: SVG rendered via python-chess
- **Graphs**: Matplotlib embedded in tkinter
- **Real-time Updates**: Queue-based communication (100ms polling)

## Future Enhancements

Potential additions:
- [ ] Export training data (CSV/JSON)
- [ ] Import/export learned patterns
- [ ] Adjustable AI difficulty (search depth)
- [ ] Play against the AI (manual moves)
- [ ] Pattern inspector (view top patterns)
- [ ] Heatmap of piece activity
- [ ] Opening repertoire statistics

## License

Part of the Chess Pattern Recognition AI project.
