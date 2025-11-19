# Developer Guide - Pattern Recognition AI System

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Adding a New Game](#adding-a-new-game)
3. [API Reference](#api-reference)
4. [Testing Guidelines](#testing-guidelines)
5. [Code Conventions](#code-conventions)
6. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### System Design Philosophy

The Pattern Recognition AI system is designed around three core principles:

1. **Game-Agnostic Learning**: 85% of code is reused across all games
2. **Observation-Based Discovery**: AI learns from outcomes, not hardcoded rules
3. **Differential Scoring**: Evaluate positions by comparing advantages

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│              (game_headless_trainer.py)                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────────┐
│  Game Engine │    │    Scorer    │   │ Move Prioritizer │
│  (_game.py)  │    │ (_scorer.py) │   │   (shared)       │
└──────────────┘    └──────────────┘   └──────────────────┘
        │                   │                   │
        ▼                   │                   │
┌──────────────┐            │                   │
│    Board     │            │                   │
│ (_board.py)  │◄───────────┘                   │
└──────────────┘                                │
                                                ▼
                                        ┌──────────────┐
                                        │   SQLite DB  │
                                        │  (patterns)  │
                                        └──────────────┘
```

### Data Flow

1. **Game Execution**: Game engine generates moves using board state
2. **Move Evaluation**: Scorer evaluates each move with differential scoring
3. **Pattern Learning**: Move prioritizer records patterns and outcomes
4. **Adaptation**: Future games use learned patterns to prioritize moves

---

## Adding a New Game

Follow these steps to add a new game to the system. Estimated time: 4-6 hours.

### Step 1: Create Game Directory

```bash
cd chess_pattern_ai
mkdir my_game
```

### Step 2: Implement Board (`my_game_board.py`)

The board class manages game state and rules.

**Required Components:**

```python
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass

class Color(Enum):
    """Player colors - keep names consistent"""
    WHITE = "white"
    BLACK = "black"

@dataclass
class Piece:
    """Game piece representation"""
    color: Color
    position: Tuple[int, int]
    # Add game-specific attributes (type, strength, etc.)

class MyGameBoard:
    """Board state and rules"""

    def __init__(self):
        """Initialize empty or starting position"""
        self.board = []  # Your board representation
        self.move_history = []

    def copy(self) -> 'MyGameBoard':
        """Deep copy for move exploration"""
        # CRITICAL: Must be a true deep copy
        # Game state changes must not affect the original
        pass

    def get_valid_moves(self, color: Color) -> List:
        """Generate all legal moves for color"""
        # Return list of moves (format is game-specific)
        pass

    def make_move(self, move) -> 'MyGameBoard':
        """Apply move and return new board"""
        # Return a NEW board (don't modify self)
        new_board = self.copy()
        # Apply move to new_board
        return new_board

    def is_game_over(self) -> bool:
        """Check if game has ended"""
        pass

    def get_winner(self) -> Optional[Color]:
        """Return winner or None if draw/ongoing"""
        pass

    def __str__(self) -> str:
        """String representation for debugging"""
        pass
```

**Best Practices:**

- Keep move generation efficient (this is called thousands of times)
- Use bitboards for binary games (checkers, chess)
- Cache expensive calculations (connectivity checks, threat detection)
- Write comprehensive `copy()` - bugs here cause subtle training issues

### Step 3: Implement Game Engine (`my_game_game.py`)

The game engine manages turn order and move application.

```python
from typing import List, Optional
from .my_game_board import MyGameBoard, Color

class MyGameGame:
    """Game engine managing turns and moves"""

    def __init__(self, board: Optional[MyGameBoard] = None):
        """Initialize with optional board state"""
        self.board = board if board else MyGameBoard()
        self.current_player = Color.WHITE
        self.move_count = 0

    def copy(self) -> 'MyGameGame':
        """Deep copy entire game state"""
        new_game = MyGameGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def get_valid_moves(self) -> List:
        """Get valid moves for current player"""
        return self.board.get_valid_moves(self.current_player)

    def make_move(self, move) -> 'MyGameGame':
        """Apply move and switch players"""
        new_game = self.copy()
        new_game.board = new_game.board.make_move(move)
        new_game.current_player = self._switch_player(new_game.current_player)
        new_game.move_count += 1
        return new_game

    def is_game_over(self) -> bool:
        """Check if game has ended"""
        # Check for wins, draws, stalemates
        if self.board.is_game_over():
            return True
        # Add move limit check to prevent infinite games
        if self.move_count >= 200:  # Adjust per game
            return True
        return False

    def get_result(self) -> str:
        """Return 'white_wins', 'black_wins', or 'draw'"""
        winner = self.board.get_winner()
        if winner == Color.WHITE:
            return 'white_wins'
        elif winner == Color.BLACK:
            return 'black_wins'
        return 'draw'

    @staticmethod
    def _switch_player(color: Color) -> Color:
        """Switch between players"""
        return Color.BLACK if color == Color.WHITE else Color.WHITE
```

**Important Notes:**

- Always return NEW game objects (immutability is key)
- Include move limit to prevent infinite games
- Handle stalemates/draws correctly

### Step 4: Implement Scorer (`my_game_scorer.py`)

The scorer evaluates positions and categorizes moves.

```python
from typing import Tuple
from .my_game_board import MyGameBoard, Color

class MyGameScorer:
    """Differential scoring and move categorization"""

    def score_position(self, board: MyGameBoard, color: Color) -> float:
        """
        Evaluate position from color's perspective.

        Returns:
            Positive = good for color
            Negative = bad for color

        Key principle: score = my_advantage - opponent_advantage
        """
        my_score = 0.0
        opponent_score = 0.0
        opponent = Color.BLACK if color == Color.WHITE else Color.WHITE

        # Material/piece count
        my_score += self._count_pieces(board, color) * 100
        opponent_score += self._count_pieces(board, opponent) * 100

        # Positional advantages
        my_score += self._evaluate_position(board, color)
        opponent_score += self._evaluate_position(board, opponent)

        # Threats and tactical patterns
        my_score += self._count_threats(board, color) * 50
        opponent_score += self._count_threats(board, opponent) * 50

        # Differential scoring
        return my_score - opponent_score

    def get_move_category(self, board_before: MyGameBoard,
                          board_after: MyGameBoard,
                          move, color: Color) -> str:
        """
        Categorize move for pattern learning.

        Categories should be:
        - Mutually exclusive (each move has one category)
        - Strategically meaningful (captures, threats, development)
        - Ordered by importance (winning > capturing > threatening)

        Returns: category name (str)
        """
        # 1. Check for immediate wins
        if board_after.get_winner() == color:
            return 'winning'

        # 2. Check for captures
        if self._is_capture(board_before, board_after, move):
            return 'capture'

        # 3. Check for threats
        if self._creates_threat(board_after, color):
            return 'threat'

        # 4. Check for defensive moves
        opponent = Color.BLACK if color == Color.WHITE else Color.WHITE
        if self._blocks_threat(board_before, board_after, opponent):
            return 'defensive'

        # 5. Positional categories (game-specific)
        # Examples: 'center_control', 'development', 'grouping'

        # 6. Default quiet move
        return 'quiet'

    def get_game_phase(self, board: MyGameBoard) -> str:
        """
        Determine current game phase.

        Returns: 'opening', 'middlegame', or 'endgame'
        """
        # Example: piece count based
        total_pieces = self._count_all_pieces(board)

        if total_pieces > 24:
            return 'opening'
        elif total_pieces > 10:
            return 'middlegame'
        else:
            return 'endgame'

    # Helper methods
    def _count_pieces(self, board: MyGameBoard, color: Color) -> int:
        """Count pieces for a color"""
        pass

    def _evaluate_position(self, board: MyGameBoard, color: Color) -> float:
        """Evaluate positional factors"""
        pass

    def _count_threats(self, board: MyGameBoard, color: Color) -> int:
        """Count threatening moves available"""
        pass

    def _is_capture(self, before: MyGameBoard, after: MyGameBoard, move) -> bool:
        """Check if move captured a piece"""
        pass

    def _creates_threat(self, board: MyGameBoard, color: Color) -> bool:
        """Check if position creates threats"""
        pass

    def _blocks_threat(self, before: MyGameBoard, after: MyGameBoard,
                       opponent: Color) -> bool:
        """Check if move blocked opponent's threat"""
        pass
```

**Scoring Guidelines:**

- **Differential**: Always compute `my_advantage - opponent_advantage`
- **Consistency**: Use same scale across all evaluations
- **Material First**: Start with piece/territory count
- **Add Context**: Include positional, tactical factors
- **Categories Matter**: Good categorization = better learning

**Common Scoring Metrics:**

- **Material**: Piece count × value
- **Mobility**: Number of legal moves
- **Territory**: Controlled squares/areas
- **Threats**: Attacking moves available
- **Safety**: King safety, piece protection
- **Position**: Center control, piece placement

### Step 5: Implement Trainer (`my_game_headless_trainer.py`)

The trainer runs self-play games and manages learning.

```python
#!/usr/bin/env python3
"""
Headless trainer for MyGame.
Runs self-play games with pattern learning.
"""

import argparse
import random
import time
import os
from typing import List, Tuple
from learnable_move_prioritizer import LearnableMovePrioritizer
from .my_game_game import MyGameGame, Color
from .my_game_scorer import MyGameScorer
from .my_game_board import MyGameBoard

class MyGameHeadlessTrainer:
    """Self-play trainer with pattern learning"""

    def __init__(self, db_path: str = "my_game_training.db"):
        """Initialize trainer with database"""
        self.db_path = db_path
        self.prioritizer = LearnableMovePrioritizer(db_path)
        self.scorer = MyGameScorer()

        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_score = 0

    def play_game(self, verbose: bool = False) -> Tuple[str, int]:
        """
        Play one self-play game.

        Returns:
            (result, final_score) where result is 'white_wins', 'black_wins', or 'draw'
        """
        game = MyGameGame()
        move_history = []  # Track moves for learning

        while not game.is_game_over():
            # Get valid moves
            valid_moves = game.get_valid_moves()

            if not valid_moves:
                break

            # Score and categorize all moves
            move_data = []
            for move in valid_moves:
                # Apply move
                new_game = game.make_move(move)

                # Score position
                score = self.scorer.score_position(new_game.board, game.current_player)

                # Categorize move
                category = self.scorer.get_move_category(
                    game.board, new_game.board, move, game.current_player
                )

                # Get game phase
                phase = self.scorer.get_game_phase(game.board)

                # Calculate distance (game-specific)
                distance = self._calculate_move_distance(move)

                move_data.append({
                    'move': move,
                    'score': score,
                    'category': category,
                    'phase': phase,
                    'distance': distance,
                    'color': game.current_player
                })

            # Select move using learned patterns
            selected_move = self._select_move(move_data)

            # Record move for learning
            move_history.append({
                'move_data': move_data,
                'selected': selected_move,
                'move_number': game.move_count
            })

            # Make move
            game = game.make_move(selected_move['move'])

            if verbose:
                print(f"Move {game.move_count}: {selected_move['category']} "
                      f"(score: {selected_move['score']:.1f})")

        # Get result
        result = game.get_result()
        final_score = self.scorer.score_position(game.board, Color.WHITE)

        # Learn from game
        self._learn_from_game(move_history, result, final_score)

        # Update statistics
        self._update_statistics(result, final_score)

        return result, final_score

    def _select_move(self, move_data: List[dict]) -> dict:
        """
        Select move using learned patterns and exploration.

        Strategy:
        1. Get learned priorities for each move
        2. Add score-based bonus
        3. Add exploration noise
        4. Select best combined score
        """
        # Add learned priorities
        for data in move_data:
            learned_priority = self.prioritizer.get_move_priority(
                piece_type='piece',  # Or piece type if applicable
                move_category=data['category'],
                distance=data['distance'],
                phase=data['phase']
            )
            data['learned_priority'] = learned_priority

        # Combine score with learned priority
        for data in move_data:
            # Normalize score to 0-100 range
            normalized_score = self._normalize_score(data['score'], move_data)

            # Combine: 50% learned patterns, 50% position score
            data['combined_score'] = (
                0.5 * data['learned_priority'] +
                0.5 * normalized_score
            )

            # Add exploration noise (10% randomness)
            data['combined_score'] += random.uniform(-10, 10)

        # Select best move
        selected = max(move_data, key=lambda x: x['combined_score'])
        return selected

    def _normalize_score(self, score: float, all_moves: List[dict]) -> float:
        """Normalize score to 0-100 range"""
        scores = [m['score'] for m in all_moves]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return 50.0

        return ((score - min_score) / (max_score - min_score)) * 100

    def _learn_from_game(self, move_history: List[dict],
                         result: str, final_score: int):
        """Update pattern database with game outcome"""
        for move_record in move_history:
            selected = move_record['selected']

            # Convert result to win/loss/draw from move's perspective
            move_result = self._get_move_result(result, selected['color'])

            # Update pattern statistics
            self.prioritizer._update_move_statistics(
                piece_type='piece',
                move_category=selected['category'],
                distance=selected['distance'],
                phase=selected['phase'],
                result=move_result,
                final_score=final_score
            )

    def _get_move_result(self, game_result: str, move_color: Color) -> str:
        """Convert game result to move perspective"""
        if game_result == 'draw':
            return 'draw'

        if move_color == Color.WHITE:
            return 'win' if game_result == 'white_wins' else 'loss'
        else:
            return 'win' if game_result == 'black_wins' else 'loss'

    def _calculate_move_distance(self, move) -> int:
        """Calculate move distance (game-specific)"""
        # Example: Manhattan distance
        # Implement based on your move representation
        return 1

    def _update_statistics(self, result: str, final_score: int):
        """Update game statistics"""
        self.games_played += 1

        if result == 'white_wins':
            self.white_wins += 1
            self.total_score += abs(final_score)
        elif result == 'black_wins':
            self.black_wins += 1
            self.total_score -= abs(final_score)
        else:
            self.draws += 1

    def train(self, num_games: int, progress_interval: int = 10):
        """Train by playing multiple games"""
        print("=" * 70)
        print("MY GAME HEADLESS TRAINING")
        print("=" * 70)
        print(f"Games to play: {num_games}")
        print(f"Database: {self.db_path}")
        print(f"Progress updates every {progress_interval} games")
        print("=" * 70)
        print()

        start_time = time.time()

        for i in range(num_games):
            result, score = self.play_game()

            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed
                print(f"Game {i + 1}/{num_games} | "
                      f"W: {self.white_wins} L: {self.black_wins} D: {self.draws} | "
                      f"Speed: {games_per_sec:.2f} games/sec")

        elapsed = time.time() - start_time
        self._print_final_statistics(elapsed)

    def _print_final_statistics(self, elapsed_time: float):
        """Print training summary"""
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total Games: {self.games_played}")
        print(f"Results: {self.white_wins}W - {self.black_wins}L - {self.draws}D")

        if self.games_played > 0:
            win_rate = (self.white_wins / self.games_played) * 100
            avg_score = self.total_score / self.games_played
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Average Score: {avg_score:+.0f}")

        print(f"Total Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"Speed: {self.games_played/elapsed_time:.2f} games/sec")
        print("=" * 70)

    def show_top_patterns(self, limit: int = 20):
        """Display top learned patterns"""
        patterns = self.prioritizer.get_top_patterns(limit)

        print("\n" + "=" * 70)
        print("TOP LEARNED PATTERNS")
        print("=" * 70)
        print(f"{'Category':<15} {'Dist':<6} {'Phase':<12} {'Win%':<8} {'Seen':<8} {'Priority':<10}")
        print("-" * 70)

        for p in patterns:
            print(f"{p['move_category']:<15} {p['distance']:<6} {p['phase']:<12} "
                  f"{p['win_rate']:.1f}% {p['times_seen']:<8} {p['priority']:.1f}")

        print("=" * 70)

    def close(self):
        """Close database connection"""
        self.prioritizer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MyGame Headless Trainer')
    parser.add_argument('num_games', type=int, nargs='?', default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--db', type=str, default='my_game_training.db',
                        help='Database path')
    parser.add_argument('--progress', type=int, default=10,
                        help='Progress update interval')
    parser.add_argument('--show-patterns', type=int, metavar='N',
                        help='Show top N patterns and exit')

    args = parser.parse_args()

    trainer = MyGameHeadlessTrainer(args.db)

    try:
        if args.show_patterns:
            trainer.show_top_patterns(args.show_patterns)
        else:
            trainer.train(args.num_games, args.progress)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
```

### Step 6: Add Tests

Add test cases to `test_all_games.py`:

```python
# Import your game
from my_game.my_game_board import MyGameBoard, Color
from my_game.my_game_game import MyGameGame
from my_game.my_game_scorer import MyGameScorer

class TestMyGame(unittest.TestCase):
    """Test MyGame implementation"""

    def test_initial_board(self):
        """Test board initialization"""
        board = MyGameBoard()
        # Assert initial state is correct

    def test_move_generation(self):
        """Test that moves are generated correctly"""
        board = MyGameBoard()
        moves = board.get_valid_moves(Color.WHITE)
        self.assertGreater(len(moves), 0)

    def test_move_application(self):
        """Test that moves are applied correctly"""
        game = MyGameGame()
        moves = game.get_valid_moves()
        new_game = game.make_move(moves[0])
        # Assert state changed correctly

    def test_win_detection(self):
        """Test win condition detection"""
        # Set up winning position
        # Assert is_game_over() returns True
        # Assert get_winner() returns correct color

    def test_differential_scoring(self):
        """Test scorer differential scoring"""
        board = MyGameBoard()
        scorer = MyGameScorer()
        score_white = scorer.score_position(board, Color.WHITE)
        score_black = scorer.score_position(board, Color.BLACK)
        # For symmetric starting positions:
        self.assertAlmostEqual(score_white, -score_black, delta=1.0)
```

### Step 7: Update `.gitignore`

```bash
echo "my_game_training.db" >> .gitignore
```

### Step 8: Update `game_launcher_gui.py`

Add your game to the GAMES dictionary:

```python
GAMES = {
    # ... existing games ...
    'My Game': {
        'trainer': 'python3 my_game/my_game_headless_trainer.py',
        'db': 'my_game_training.db',
        'default_games': 100
    }
}
```

### Step 9: Test Your Implementation

```bash
# Run basic game
cd chess_pattern_ai
python3 my_game/my_game_headless_trainer.py 10

# Check patterns learned
python3 my_game/my_game_headless_trainer.py --show-patterns 10

# Run unit tests
python3 -m pytest test_all_games.py::TestMyGame -v
```

---

## API Reference

### LearnableMovePrioritizer

The shared learning engine used by all games.

#### Constructor

```python
LearnableMovePrioritizer(db_path: str)
```

**Parameters:**
- `db_path`: Path to SQLite database file

#### Methods

##### `get_move_priority(piece_type: str, move_category: str, distance: int, phase: str) -> float`

Get learned priority for a move pattern.

**Parameters:**
- `piece_type`: Type of piece (or 'piece' for games without types)
- `move_category`: Move category from scorer
- `distance`: Move distance (1-10 typically)
- `phase`: Game phase ('opening', 'middlegame', 'endgame')

**Returns:**
- Priority score (0-100, higher is better)

**Example:**
```python
priority = prioritizer.get_move_priority(
    piece_type='pawn',
    move_category='capture',
    distance=1,
    phase='middlegame'
)
# priority = 87.5 (good move based on past games)
```

##### `_update_move_statistics(piece_type: str, move_category: str, distance: int, phase: str, result: str, final_score: int)`

Update pattern statistics after a game.

**Parameters:**
- `result`: 'win', 'loss', or 'draw'
- `final_score`: Final position score

**Called by:** Trainer after each game

##### `get_top_patterns(limit: int = 20) -> List[dict]`

Get top patterns by priority.

**Returns:**
```python
[
    {
        'move_category': 'capture',
        'distance': 1,
        'phase': 'middlegame',
        'win_rate': 87.5,
        'times_seen': 120,
        'priority': 92.3
    },
    # ...
]
```

##### `close()`

Close database connection. Always call in `finally` block.

---

## Testing Guidelines

### Unit Tests

Every game should have these tests:

1. **Board initialization**: Verify starting position
2. **Move generation**: Check legal moves are generated
3. **Move application**: Verify moves change state correctly
4. **Win detection**: Test win/loss/draw conditions
5. **Differential scoring**: Verify symmetric scores for symmetric positions
6. **Copy correctness**: Ensure deep copies are independent

### Integration Tests

Test the full training loop:

```python
def test_training_integration(self):
    """Test full training workflow"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        trainer = MyGameHeadlessTrainer(db_path)
        result, score = trainer.play_game()

        # Verify result is valid
        self.assertIn(result, ['white_wins', 'black_wins', 'draw'])

        # Verify patterns were learned
        patterns = trainer.prioritizer.get_top_patterns(limit=5)
        self.assertGreater(len(patterns), 0)

        trainer.close()
    finally:
        os.unlink(db_path)
```

### Performance Tests

Measure training speed:

```python
def test_training_speed(self):
    """Verify game plays quickly enough"""
    trainer = MyGameHeadlessTrainer(':memory:')

    start = time.time()
    for _ in range(10):
        trainer.play_game()
    elapsed = time.time() - start

    games_per_sec = 10 / elapsed
    self.assertGreater(games_per_sec, 5)  # At least 5 games/sec
```

---

## Code Conventions

### Naming

- **Classes**: PascalCase (`MyGameBoard`, `MyGameScorer`)
- **Functions**: snake_case (`get_valid_moves`, `score_position`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_MOVES`, `BOARD_SIZE`)
- **Private methods**: Leading underscore (`_calculate_distance`)

### File Organization

```
my_game/
├── __init__.py              # Empty or minimal imports
├── my_game_board.py         # 200-400 lines
├── my_game_game.py          # 150-300 lines
├── my_game_scorer.py        # 200-300 lines
└── my_game_headless_trainer.py  # 350-450 lines
```

### Documentation

Every public method needs a docstring:

```python
def get_valid_moves(self, color: Color) -> List[Move]:
    """
    Generate all legal moves for the given color.

    Args:
        color: Color to generate moves for

    Returns:
        List of valid moves in current position

    Example:
        >>> board = MyGameBoard()
        >>> moves = board.get_valid_moves(Color.WHITE)
        >>> len(moves)
        20
    """
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Optional, Tuple, Dict

def score_position(self, board: MyGameBoard, color: Color) -> float:
    pass

def get_winner(self) -> Optional[Color]:
    pass

def get_move_data(self) -> Dict[str, any]:
    pass
```

---

## Performance Optimization

### Bottleneck: Move Generation

Move generation is called thousands of times per game.

**Optimization strategies:**

1. **Lazy evaluation**: Only generate moves when needed
2. **Incremental updates**: Update legal moves instead of regenerating
3. **Bitboards**: For binary games, use bit manipulation
4. **Caching**: Cache expensive calculations (board hash → valid moves)

**Example: Caching**

```python
class MyGameBoard:
    def __init__(self):
        self._move_cache = {}

    def get_valid_moves(self, color: Color) -> List[Move]:
        board_hash = self._hash_position()

        if board_hash in self._move_cache:
            return self._move_cache[board_hash]

        moves = self._generate_moves(color)
        self._move_cache[board_hash] = moves
        return moves

    def _hash_position(self) -> int:
        # Create unique hash for position
        pass
```

### Bottleneck: Position Scoring

Scoring is called for every legal move.

**Optimization strategies:**

1. **Incremental scoring**: Update score delta instead of recalculating
2. **Early termination**: Stop scoring if move is obviously bad
3. **Lazy evaluation**: Only calculate expensive features if needed

**Example: Incremental Scoring**

```python
def score_move_delta(self, board: MyGameBoard, move: Move,
                     current_score: float) -> float:
    """
    Calculate score change from move instead of full rescore.
    Much faster than score_position().
    """
    delta = 0.0

    # Only recalculate affected features
    if self._is_capture(move):
        delta += 100  # Captured piece value

    # Position change delta
    delta += self._position_delta(board, move)

    return current_score + delta
```

### Bottleneck: Database Access

Database I/O can slow training.

**Optimization strategies:**

1. **Batch updates**: Update patterns in batches, not one-by-one
2. **Connection pooling**: Reuse database connections
3. **In-memory during training**: Use `:memory:` for speed, save to disk after

**Example: Batch Updates**

```python
class LearnableMovePrioritizer:
    def __init__(self, db_path: str):
        self._pending_updates = []
        self._batch_size = 100

    def _update_move_statistics(self, **kwargs):
        self._pending_updates.append(kwargs)

        if len(self._pending_updates) >= self._batch_size:
            self._flush_updates()

    def _flush_updates(self):
        # Batch insert/update
        cursor = self.connection.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO learned_move_patterns ...",
            self._pending_updates
        )
        self.connection.commit()
        self._pending_updates = []
```

### Target Performance

Aim for these benchmarks:

- **Training speed**: 10+ games/second for simple games, 1+ for complex
- **Move generation**: <1ms for typical position
- **Position scoring**: <0.1ms per move
- **Database updates**: <10ms per game

---

## Advanced Topics

### Multi-Phase Games

Games like Nine Men's Morris have distinct phases.

**Implementation:**

```python
class GamePhase(Enum):
    PLACEMENT = "placement"
    MOVEMENT = "movement"
    FLYING = "flying"

class MyGameGame:
    def get_game_phase(self) -> GamePhase:
        if self.pieces_placed < 9:
            return GamePhase.PLACEMENT
        elif self.piece_count <= 3:
            return GamePhase.FLYING
        return GamePhase.MOVEMENT

    def get_valid_moves(self) -> List[Move]:
        phase = self.get_game_phase()

        if phase == GamePhase.PLACEMENT:
            return self._get_placement_moves()
        elif phase == GamePhase.FLYING:
            return self._get_flying_moves()
        return self._get_normal_moves()
```

### Multi-Step Turns

Games like Arimaa allow multiple actions per turn.

**Implementation:**

```python
class MyGameGame:
    def __init__(self):
        self.steps_remaining = 4  # Steps per turn
        self.current_turn_steps = []

    def is_turn_complete(self) -> bool:
        return self.steps_remaining == 0

    def make_step(self, step: Step) -> 'MyGameGame':
        """Apply one step of a multi-step turn"""
        new_game = self.copy()
        new_game.board = new_game.board.apply_step(step)
        new_game.steps_remaining -= 1
        new_game.current_turn_steps.append(step)

        if new_game.is_turn_complete():
            new_game._complete_turn()

        return new_game
```

### Stochastic Games

Games with randomness (dice, card draws).

**Not currently supported** - All games are deterministic. Adding stochastic games would require:

1. Expectimax search instead of minimax
2. Averaging scores over random outcomes
3. Different learning approach (Q-learning or policy gradients)

---

This completes the developer guide. For troubleshooting, see `TROUBLESHOOTING.md`.
