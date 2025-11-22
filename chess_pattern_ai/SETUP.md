# Setup Instructions

## Quick Start

### Install Dependencies

On Windows (PowerShell):
```powershell
pip install python-chess numpy scikit-learn
```

Or using the requirements file:
```powershell
pip install -r requirements.txt
```

On Linux/Mac:
```bash
pip3 install python-chess numpy scikit-learn
```

Or using the requirements file:
```bash
pip3 install -r requirements.txt
```

## Dependencies

**Required**:
- `python-chess` - Chess game logic and board representation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning (for clustering features)

**Optional**:
- `scipy` - Scientific computing (for some advanced ARC features)

All other dependencies are part of Python's standard library.

## Verify Installation

Test that everything is installed:

```python
python3 -c "import chess; import numpy; import sklearn; print('✓ All dependencies installed')"
```

On Windows:
```powershell
python -c "import chess; import numpy; import sklearn; print('✓ All dependencies installed')"
```

## Running the Trainer

### Windows:
```powershell
python chess_pattern_ai/headless_trainer.py 5 --verbose
```

### Linux/Mac:
```bash
python3 chess_pattern_ai/headless_trainer.py 5 --verbose
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'chess'"
**Solution**: Install python-chess
```bash
pip install python-chess
```

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution**: Install numpy
```bash
pip install numpy
```

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: Install scikit-learn
```bash
pip install scikit-learn
```

### Issue: pip not found
**Solution**: Make sure Python and pip are in your PATH, or use:
```bash
python -m pip install python-chess numpy scikit-learn
```

## Project Structure

```
patmat/
├── requirements.txt          # Python dependencies
├── SETUP.md                 # This file
├── arc_dataset/             # ARC puzzle dataset
│   └── data/
├── chess_pattern_ai/        # Main code directory
│   ├── headless_trainer.py  # Command-line chess trainer
│   ├── arc_solver.py        # ARC puzzle solver
│   ├── arc_cross_game_learner.py  # Cross-game pattern learning
│   └── ...
└── *.db                     # SQLite databases (generated)
```

## Quick Test

After installing dependencies, test the system:

```bash
# Test chess AI
python3 chess_pattern_ai/headless_trainer.py 1

# Test ARC solver (if you have the ARC dataset)
python3 chess_pattern_ai/test_arc_learning.py

# Test cross-game pattern extraction
python3 chess_pattern_ai/extract_patterns_from_gameplay.py
```

## Database Files

The system creates several SQLite databases:
- `rule_discovery.db` - Chess learning database (11,533 games)
- `checkers_training.db` - Checkers patterns
- `universal_patterns.db` - Cross-game universal patterns
- `arc_meta_patterns.db` - ARC puzzle patterns

These are auto-generated and don't need manual setup.

## Python Version

Requires Python 3.7 or higher.

Check your version:
```bash
python --version
# or
python3 --version
```

## Getting Help

If you encounter issues:
1. Check you're using Python 3.7+
2. Verify all dependencies are installed: `pip list | grep -E "(chess|numpy|sklearn)"`
3. Make sure you're in the project root directory
4. On Windows, use `python` instead of `python3`
