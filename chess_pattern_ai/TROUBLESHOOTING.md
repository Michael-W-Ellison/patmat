# Troubleshooting Guide - Pattern Recognition AI System

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Training Problems](#training-problems)
4. [GUI Issues](#gui-issues)
5. [Database Issues](#database-issues)
6. [Performance Issues](#performance-issues)
7. [Learning Issues](#learning-issues)
8. [Platform-Specific Issues](#platform-specific-issues)

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'tkinter'`

**Symptom:**
```
Traceback (most recent call last):
  File "game_launcher_gui.py", line 13, in <module>
    import tkinter as tk
ModuleNotFoundError: No module named 'tkinter'
```

**Cause:** tkinter is not installed or not available in your Python installation.

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-tk
```

**macOS:**
```bash
# tkinter comes with Python from python.org
# If using Homebrew Python:
brew install python-tk
```

**Windows:**
```bash
# Reinstall Python from python.org with tcl/tk option checked
# Or use anaconda:
conda install -c anaconda tk
```

**Headless Server (no GUI):**
```bash
# Use command-line trainers instead of GUI
python3 checkers/checkers_headless_trainer.py 100
```

---

### Issue: `ModuleNotFoundError: No module named 'chess'`

**Symptom:**
```
ModuleNotFoundError: No module named 'chess'
```

**Cause:** python-chess library not installed (only needed for chess game).

**Solution:**
```bash
pip3 install python-chess
```

**Note:** Only chess requires external dependencies. All other games use Python standard library only.

---

### Issue: Import errors with relative imports

**Symptom:**
```
ImportError: attempted relative import with no known parent package
```

**Cause:** Running module directly instead of as package.

**Solution:**

**Wrong:**
```bash
cd checkers
python3 checkers_headless_trainer.py
```

**Correct:**
```bash
# Run from chess_pattern_ai directory
cd chess_pattern_ai
python3 checkers/checkers_headless_trainer.py
```

Or add parent directory to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/user/patmat/chess_pattern_ai"
```

---

## Runtime Errors

### Issue: `AttributeError: 'NoneType' object has no attribute 'color'`

**Symptom:**
```
AttributeError: 'NoneType' object has no attribute 'color'
```

**Cause:** Accessing a piece that doesn't exist (empty square).

**Solution:** Always check for None before accessing piece attributes:

**Wrong:**
```python
piece = board.board[row][col]
if piece.color == Color.WHITE:  # Crashes if piece is None
    ...
```

**Correct:**
```python
piece = board.board[row][col]
if piece is not None and piece.color == Color.WHITE:
    ...
```

---

### Issue: Infinite game loops / Games never end

**Symptom:**
- Training hangs on a single game
- Game reaches thousands of moves
- Memory usage grows continuously

**Cause:** No move limit or draw detection.

**Solution:** Add move limit to `is_game_over()`:

```python
def is_game_over(self) -> bool:
    # Check normal win conditions
    if self.board.get_winner() is not None:
        return True

    # Check for no legal moves (stalemate)
    if len(self.get_valid_moves()) == 0:
        return True

    # IMPORTANT: Add move limit to prevent infinite games
    if self.move_count >= 200:  # Adjust per game
        return True

    return False
```

**Game-specific limits:**
- Chess: 100-150 moves
- Checkers: 100 moves
- Go: 200-300 moves
- Pentago: 36 moves (max fills board)
- Connect Four: 42 moves (max fills board)

---

### Issue: `RecursionError: maximum recursion depth exceeded`

**Symptom:**
```
RecursionError: maximum recursion depth exceeded in comparison
```

**Cause:** Circular references or deep recursion in move generation.

**Solution:**

1. **Check for infinite loops in move generation:**

```python
def get_valid_moves(self, color: Color) -> List[Move]:
    moves = []
    visited = set()  # Prevent checking same position twice

    for position in self.get_positions(color):
        if position in visited:
            continue
        visited.add(position)
        # Generate moves...

    return moves
```

2. **Avoid recursive position evaluation:**

```python
# Wrong: Recursive evaluation
def score_position(self, board):
    for move in board.get_valid_moves():
        new_board = board.make_move(move)
        score += self.score_position(new_board)  # Recursion!

# Right: Direct evaluation only
def score_position(self, board):
    score = 0
    score += self._count_pieces(board)
    score += self._evaluate_threats(board)
    return score
```

---

### Issue: `sqlite3.OperationalError: database is locked`

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Cause:** Multiple processes accessing same database file.

**Solutions:**

1. **Use separate database per training session:**
```bash
python3 checkers/checkers_headless_trainer.py 100 --db checkers_session1.db &
python3 checkers/checkers_headless_trainer.py 100 --db checkers_session2.db &
```

2. **Wait for training to complete before viewing:**
```bash
# Training
python3 checkers/checkers_headless_trainer.py 100

# Then view patterns (after training completes)
python3 checkers/checkers_headless_trainer.py --show-patterns 20
```

3. **Increase timeout:**
```python
connection = sqlite3.connect(db_path, timeout=30.0)  # Wait up to 30 seconds
```

---

## Training Problems

### Issue: Win rate stuck at 50% (not learning)

**Symptom:**
- After 1000+ games, win rate stays around 50%
- No improvement over time
- Random move selection

**Diagnosis:**

```bash
# Check if patterns are being learned
python3 my_game/my_game_headless_trainer.py --show-patterns 20
```

If you see:
```
TOP LEARNED PATTERNS
====================================================================
Category        Dist   Phase        Win%     Seen     Priority
--------------------------------------------------------------------
quiet           1      opening      50.0%    1000     50.0
quiet           1      middlegame   50.0%    800      50.0
```

**Causes and Solutions:**

1. **Move categories too generic:**

**Problem:** Everything is categorized as 'quiet'
```python
def get_move_category(self, ...):
    return 'quiet'  # Too generic!
```

**Solution:** Add meaningful categories:
```python
def get_move_category(self, board_before, board_after, move, color):
    if board_after.get_winner() == color:
        return 'winning'
    if self._is_capture(board_before, board_after):
        return 'capture'
    if self._creates_threat(board_after, color):
        return 'threat'
    # Add more categories...
    return 'quiet'
```

2. **Scoring too simplistic:**

**Problem:** All positions score the same
```python
def score_position(self, board, color):
    return 0.0  # No differentiation!
```

**Solution:** Add differential scoring:
```python
def score_position(self, board, color):
    opponent = self._get_opponent(color)

    my_score = self._count_pieces(board, color) * 100
    my_score += self._count_threats(board, color) * 50

    opp_score = self._count_pieces(board, opponent) * 100
    opp_score += self._count_threats(board, opponent) * 50

    return my_score - opp_score  # Differential!
```

3. **Exploration too high:**

**Problem:** Too much randomness in move selection
```python
data['combined_score'] += random.uniform(-50, 50)  # Too much noise!
```

**Solution:** Reduce exploration:
```python
data['combined_score'] += random.uniform(-5, 5)  # Small exploration
```

---

### Issue: AI plays terribly / Makes obvious blunders

**Symptom:**
- AI misses winning moves
- AI hangs pieces/loses material
- Win rate < 20% even after training

**Diagnosis:**

Check top patterns:
```bash
python3 my_game/my_game_headless_trainer.py --show-patterns 20
```

Look for:
- **Winning moves have low priority**: Scoring is backwards
- **Captures have low win rate**: Something is wrong

**Solutions:**

1. **Check score polarity:**

```python
def score_position(self, board, color):
    # CRITICAL: Positive = good for color, Negative = bad for color
    score = self._count_my_pieces(board, color)
    score -= self._count_opponent_pieces(board, color)
    return score  # Must be from color's perspective!
```

**Test with symmetric position:**
```python
def test_score_symmetry(self):
    board = MyGameBoard()  # Starting position
    scorer = MyGameScorer()

    score_white = scorer.score_position(board, Color.WHITE)
    score_black = scorer.score_position(board, Color.BLACK)

    # For symmetric starting position:
    self.assertAlmostEqual(score_white, -score_black, delta=1.0)
```

2. **Ensure winning moves are prioritized:**

```python
def get_move_category(self, board_before, board_after, move, color):
    # ALWAYS check for wins first!
    if board_after.get_winner() == color:
        return 'winning'

    # Then other categories...
```

3. **Verify differential scoring:**

```python
def score_position(self, board, color):
    opponent = self._get_opponent(color)

    # Calculate MY advantages
    my_material = self._count_pieces(board, color)
    my_mobility = len(board.get_valid_moves(color))

    # Calculate OPPONENT advantages
    opp_material = self._count_pieces(board, opponent)
    opp_mobility = len(board.get_valid_moves(opponent))

    # DIFFERENTIAL: my advantages - opponent advantages
    return (my_material - opp_material) * 100 + (my_mobility - opp_mobility) * 10
```

---

### Issue: Training is extremely slow

**Symptom:**
- Less than 1 game per second
- Training 100 games takes hours

**Diagnosis:**

Time each component:
```python
import time

def play_game(self):
    t_total = time.time()

    t_moves = 0
    t_score = 0

    while not game.is_game_over():
        # Time move generation
        t1 = time.time()
        moves = game.get_valid_moves()
        t_moves += time.time() - t1

        # Time scoring
        t2 = time.time()
        for move in moves:
            new_game = game.make_move(move)
            score = self.scorer.score_position(new_game.board, color)
        t_score += time.time() - t2

        # ...

    print(f"Total: {time.time() - t_total:.3f}s | "
          f"Moves: {t_moves:.3f}s | Score: {t_score:.3f}s")
```

**Solutions:**

1. **Optimize move generation** (see Performance Optimization in DEVELOPER_GUIDE.md)

2. **Reduce branching factor:**
```python
def get_valid_moves(self, color):
    all_moves = self._generate_all_moves(color)

    # For training, limit moves to reasonable set
    if len(all_moves) > 50:
        # Prioritize promising moves
        all_moves = self._quick_prune(all_moves)[:50]

    return all_moves
```

3. **Use incremental scoring:**
```python
# Instead of rescoring entire board for each move,
# calculate score delta
def score_move_delta(self, board, move, current_score):
    delta = 0
    if self._is_capture(move):
        delta += 100
    delta += self._position_change(move)
    return current_score + delta
```

---

## GUI Issues

### Issue: GUI launches but shows blank window

**Symptom:**
- Window appears but is empty/white
- No buttons or controls visible

**Cause:** tkinter rendering issue or missing update calls.

**Solution:**

1. **Force update:**
```python
root = tk.Tk()
# ... create widgets ...
root.update()  # Force initial render
root.mainloop()
```

2. **Check geometry:**
```python
root = tk.Tk()
root.geometry("1000x700")  # Set explicit size
```

---

### Issue: "View Patterns" button does nothing

**Symptom:**
- Clicking "View Patterns" doesn't open pattern viewer
- No error messages

**Cause:** pattern_viewer_gui.py not found or database path incorrect.

**Diagnosis:**
```python
# Add debug output to button handler
def view_patterns(self, game_name):
    db_path = os.path.join(os.path.dirname(__file__), config['db'])
    print(f"Debug: DB path = {db_path}")  # Add this
    print(f"Debug: Exists = {os.path.exists(db_path)}")  # Add this
```

**Solution:**

1. **Check pattern viewer exists:**
```bash
ls -la pattern_viewer_gui.py
```

2. **Verify database exists:**
```bash
ls -la *.db
```

3. **Train first to create database:**
```bash
python3 checkers/checkers_headless_trainer.py 10
```

---

### Issue: Training output not appearing in GUI

**Symptom:**
- Start training, but output tab stays empty
- Terminal shows output but GUI doesn't

**Cause:** Output buffering or queue not being processed.

**Solution:**

1. **Flush stdout:**
```python
# In trainer
print("Game 1 complete", flush=True)  # Add flush=True
```

2. **Check queue processing:**
```python
# In GUI
def check_queue(self):
    try:
        while True:
            msg_type, game_name, data = self.update_queue.get_nowait()
            self.process_message(msg_type, game_name, data)
    except queue.Empty:
        pass

    # IMPORTANT: Reschedule
    self.root.after(100, self.check_queue)
```

---

## Database Issues

### Issue: `sqlite3.DatabaseError: database disk image is malformed`

**Symptom:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Cause:** Database corruption (often from interrupted writes or disk full).

**Solutions:**

1. **Delete and recreate:**
```bash
rm my_game_training.db
python3 my_game/my_game_headless_trainer.py 10  # Recreates DB
```

2. **Attempt recovery:**
```bash
sqlite3 my_game_training.db ".recover" | sqlite3 my_game_training_recovered.db
```

3. **Prevention - Use transactions:**
```python
connection.execute("BEGIN TRANSACTION")
try:
    # Multiple inserts/updates
    connection.execute(...)
    connection.execute(...)
    connection.execute("COMMIT")
except:
    connection.execute("ROLLBACK")
    raise
```

---

### Issue: Database grows very large (>1GB)

**Symptom:**
- Database file size in gigabytes
- Training slows down over time

**Cause:** Not aggregating patterns, storing raw data.

**Diagnosis:**
```bash
sqlite3 my_game_training.db "SELECT COUNT(*) FROM learned_move_patterns"
```

If count > 100,000, you have too many patterns.

**Solution:**

The LearnableMovePrioritizer should aggregate patterns automatically. Check that you're using it correctly:

```python
# Correct - aggregates automatically
self.prioritizer._update_move_statistics(
    piece_type='pawn',
    move_category='capture',
    distance=1,
    phase='opening',
    result='win',
    final_score=100
)

# Wrong - don't insert raw moves
cursor.execute("INSERT INTO moves VALUES (?)", (move_str,))
```

**Manual cleanup:**
```bash
# Vacuum database to reclaim space
sqlite3 my_game_training.db "VACUUM"
```

---

## Performance Issues

### Issue: Training uses excessive memory

**Symptom:**
- Memory usage grows to multiple GB
- System becomes unresponsive
- OOM (Out of Memory) errors

**Cause:** Memory leaks in move generation or board copying.

**Diagnosis:**

```python
import tracemalloc

tracemalloc.start()

# Run training
trainer.play_game()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Common Causes:**

1. **Not closing database connections:**
```python
# Wrong
def train(self, num_games):
    for i in range(num_games):
        prioritizer = LearnableMovePrioritizer(db_path)  # Leak!
        self.play_game()

# Correct
def train(self, num_games):
    prioritizer = LearnableMovePrioritizer(db_path)
    try:
        for i in range(num_games):
            self.play_game()
    finally:
        prioritizer.close()  # Always close
```

2. **Storing all game states:**
```python
# Wrong
class Trainer:
    def __init__(self):
        self.all_games = []  # Grows indefinitely!

    def play_game(self):
        game = MyGameGame()
        # ...
        self.all_games.append(game)  # Memory leak!

# Correct - don't store games
class Trainer:
    def play_game(self):
        game = MyGameGame()
        # ... play game ...
        # Let game be garbage collected after method returns
```

3. **Circular references in board:**
```python
# Wrong
class Board:
    def copy(self):
        new_board = Board()
        new_board.parent = self  # Circular reference!
        return new_board

# Correct - break references
class Board:
    def copy(self):
        new_board = Board()
        new_board.board = [row[:] for row in self.board]  # Deep copy only data
        return new_board
```

---

### Issue: High CPU usage (100% on all cores)

**Symptom:**
- CPU at 100% during training
- Fans running at maximum
- Computer becomes slow

**Expected:** Training SHOULD use high CPU - this is normal! AI training is CPU-intensive.

**Problem only if:**
- CPU stays at 100% when NOT training
- Training speed is very slow despite high CPU

**Solutions:**

1. **Reduce parallelism if overheating:**
```bash
# Run single-threaded
python3 checkers/checkers_headless_trainer.py 100

# Don't run multiple trainers simultaneously
```

2. **Add delays if needed:**
```python
def train(self, num_games):
    for i in range(num_games):
        self.play_game()
        if i % 10 == 0:
            time.sleep(0.1)  # Brief pause every 10 games
```

3. **Use lower priority:**
```bash
# Linux/Mac - run with lower priority
nice -n 19 python3 checkers/checkers_headless_trainer.py 1000

# Windows - use START /LOW
start /LOW python checkers/checkers_headless_trainer.py 1000
```

---

## Learning Issues

### Issue: Patterns show 0% or 100% win rate

**Symptom:**
```
Category        Win%     Seen
-----------------------------
capture         100.0%   50
threat          0.0%     30
```

**Cause:** Not enough data, or result tracking is wrong.

**Solutions:**

1. **Train more games:**
```bash
# 100 games is not enough for rare patterns
python3 my_game/my_game_headless_trainer.py 1000
```

2. **Check result tracking:**
```python
def _get_move_result(self, game_result: str, move_color: Color) -> str:
    """Convert game result to move perspective"""
    if game_result == 'draw':
        return 'draw'

    # CRITICAL: Return from move's perspective!
    if move_color == Color.WHITE:
        return 'win' if game_result == 'white_wins' else 'loss'
    else:
        return 'win' if game_result == 'black_wins' else 'loss'
```

**Test:**
```python
def test_result_conversion(self):
    trainer = MyGameHeadlessTrainer(':memory:')

    # White move in white win should be 'win'
    result = trainer._get_move_result('white_wins', Color.WHITE)
    self.assertEqual(result, 'win')

    # White move in black win should be 'loss'
    result = trainer._get_move_result('black_wins', Color.WHITE)
    self.assertEqual(result, 'loss')
```

---

### Issue: Same patterns appear with different win rates

**Symptom:**
```
Category    Dist  Phase       Win%    Seen
-------------------------------------------
capture     1     opening     80.0%   100
capture     1     middlegame  45.0%   80
capture     1     opening     82.0%   50   # Duplicate?
```

**Cause:** Pattern keys not being matched correctly.

**Diagnosis:**

Check database:
```bash
sqlite3 my_game_training.db
> SELECT piece_type, move_category, distance, phase, COUNT(*)
  FROM learned_move_patterns
  GROUP BY piece_type, move_category, distance, phase
  HAVING COUNT(*) > 1;
```

**Solution:**

Ensure consistent pattern keys:
```python
def _update_move_statistics(self, piece_type, move_category, distance, phase, ...):
    # Normalize inputs
    piece_type = piece_type.lower().strip()
    move_category = move_category.lower().strip()
    distance = int(distance)  # Ensure integer
    phase = phase.lower().strip()

    # Update or insert
    cursor.execute("""
        INSERT INTO learned_move_patterns (piece_type, move_category, distance, phase, ...)
        VALUES (?, ?, ?, ?, ...)
        ON CONFLICT(piece_type, move_category, distance, phase)
        DO UPDATE SET ...
    """, (piece_type, move_category, distance, phase, ...))
```

---

## Platform-Specific Issues

### Linux: Permission denied on database

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'my_game_training.db'
```

**Solution:**
```bash
# Check ownership
ls -la *.db

# Fix permissions
chmod 644 my_game_training.db

# Or remove and recreate
rm my_game_training.db
python3 my_game/my_game_headless_trainer.py 10
```

---

### macOS: "Python quit unexpectedly" on GUI launch

**Symptom:**
- GUI crashes immediately on launch
- Error dialog: "Python quit unexpectedly"

**Cause:** tkinter/Tcl version mismatch.

**Solution:**

1. **Use Python from python.org (not Homebrew):**
```bash
# Download and install from python.org
# Then:
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 game_launcher_gui.py
```

2. **Or use Anaconda:**
```bash
conda create -n patmat python=3.11
conda activate patmat
conda install -c anaconda tk
python game_launcher_gui.py
```

---

### Windows: `python3: command not found`

**Symptom:**
```
'python3' is not recognized as an internal or external command
```

**Solution:**

On Windows, use `python` instead of `python3`:

```bash
# Instead of:
python3 checkers/checkers_headless_trainer.py 100

# Use:
python checkers/checkers_headless_trainer.py 100
```

Or create alias:
```bash
doskey python3=python $*
```

---

### Windows: Scripts fail with "FileNotFoundError"

**Symptom:**
```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**Cause:** Windows uses backslashes in paths, but code uses forward slashes.

**Solution:**

Use `os.path.join()` for cross-platform compatibility:

```python
# Wrong - breaks on Windows
db_path = 'checkers/checkers_training.db'

# Correct - works everywhere
db_path = os.path.join('checkers', 'checkers_training.db')
```

---

## Getting More Help

### Enable Debug Logging

Add detailed logging to diagnose issues:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# In your code:
logger.debug(f"Generated {len(moves)} moves")
logger.info(f"Game complete: {result}")
logger.warning(f"Slow move generation: {elapsed:.3f}s")
logger.error(f"Invalid move: {move}")
```

### Common Debug Checkpoints

Add these debug prints to isolate issues:

```python
def play_game(self):
    print(f"DEBUG: Starting game")
    game = MyGameGame()

    move_count = 0
    while not game.is_game_over():
        print(f"DEBUG: Move {move_count}, player {game.current_player}")

        moves = game.get_valid_moves()
        print(f"DEBUG: Generated {len(moves)} moves")

        if not moves:
            print(f"DEBUG: No moves available!")
            break

        # Select and make move
        selected = self._select_move(moves, game)
        print(f"DEBUG: Selected {selected['category']}")

        game = game.make_move(selected['move'])
        move_count += 1

        if move_count > 300:
            print(f"DEBUG: Move limit exceeded!")
            break

    print(f"DEBUG: Game over after {move_count} moves: {game.get_result()}")
```

### Report Issues

If you encounter a bug not covered here:

1. **Simplify**: Create minimal reproduction case
2. **Debug**: Add logging to isolate exact failure point
3. **Document**: Record error messages, stack traces
4. **Test**: Verify issue exists in clean environment
5. **Report**: Open issue at https://github.com/anthropics/claude-code/issues

Include:
- Python version: `python3 --version`
- Platform: `uname -a` (Linux/Mac) or `ver` (Windows)
- Error message with full stack trace
- Minimal code to reproduce
- Expected vs actual behavior

---

## Quick Reference

### Common Commands

```bash
# Run training
python3 {game}/{game}_headless_trainer.py 100

# View patterns
python3 {game}/{game}_headless_trainer.py --show-patterns 20

# Launch GUI
python3 game_launcher_gui.py

# Run tests
python3 test_all_games.py

# Check database
sqlite3 {game}_training.db "SELECT COUNT(*) FROM learned_move_patterns"

# Delete database (fresh start)
rm {game}_training.db
```

### Quick Diagnostics

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check tkinter available
python3 -c "import tkinter; print('OK')"

# Check database size
ls -lh *.db

# Check pattern count
sqlite3 checkers_training.db "SELECT COUNT(*) FROM learned_move_patterns"

# Monitor training speed
time python3 checkers/checkers_headless_trainer.py 10
```

---

This troubleshooting guide covers 95% of issues encountered during development and use of the pattern recognition AI system. For additional help, consult DEVELOPER_GUIDE.md or open an issue.
