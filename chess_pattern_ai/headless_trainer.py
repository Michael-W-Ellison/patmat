#!/usr/bin/env python3
"""
Headless Chess AI Trainer - No GUI Required

Provides similar functionality to the GUI but runs in terminal:
- Train multiple games
- Real-time metrics display
- Progress tracking
- Export results to JSON/CSV
"""

import chess
import json
import csv
import time
import random
from datetime import datetime
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer
from general_move_patterns import GeneralPatternLearner, evaluate, universal_db_path
from chess_adapter import ChessAdapter
from mate_patterns import MatePatternLearner
from position_patterns import PositionPatternLearner
from sequence_patterns import SequencePatternLearner
from pattern_search import PatternSearch


class HeadlessTrainer:
    """Command-line chess AI trainer"""

    use_sequences = True        # multi-move patterns (sequence_patterns) in move choice
    SEQUENCE_WEIGHT = 20        # score per material unit a move's sequence patterns expect (a pawn is 10)
    SEQUENCE_EXTRA = 5          # moves the sequence patterns rate best, considered beyond the 15 by priority
    use_search = True           # choose moves by a look-ahead guided by the sequence patterns (pattern_search)
    SEARCH_SECONDS = 0.5        # time for that look-ahead per move
    use_position_values = True  # the look-ahead values positions by learned position patterns (position_patterns)
    SEARCH_SELECTIVE = True     # look-ahead reads forcing moves + a few others below the root (pattern_search)
    SEARCH_DEPTH = 3            # plies the look-ahead goes to (None: as deep as its time allows)
    SEARCH_BONUS_CAP = 3.0      # limit on the pattern bonus added to the look-ahead's value of a move
    SEARCH_THREATS = False      # look-ahead recognises threats: considers escapes, a second threat costs (pattern_search)
    use_mate_values = False     # the look-ahead values kings' surroundings by how often they preceded mate
    use_mobility_values = False # root moves get the learned value of their mobility class (general patterns)
    MOBILITY_WEIGHT = 1.0       # material units (a pawn is 10) per point of that value (scale -50..+50)
    SELF_PLAY_RANDOM_PLIES = 4  # self-play: plies played at random from the start, so that games differ
    use_judge = False           # Stockfish judges every agent move; the general patterns learn from it (move_judges)
    JUDGE_DEPTH = 12            # the judge's search depth
    judge_positions = False     # with the judge: position patterns learn the judge's evaluations (position_patterns)
    use_general_values = False  # root moves get the learned value of their general pattern (general_move_patterns)
    GENERAL_WEIGHT = 0.2        # material units (a pawn is 10) per point of that value (scale -50..+50)
    use_refutations = False     # learn (with the judge) which replies refute blunders; the search reads them
    use_move_history = False    # the search orders moves by killer moves and history scores (move_history)
    use_tactics = False         # general patterns get tactical parts: threats left after the best reply
                                # (tactical_patterns)

    def __init__(self, db_path='headless_training.db', opponent='random', stockfish_level=0,
                 stockfish_path='/usr/games/stockfish', stockfish_time=0.1, screen=None, screen_side=False):
        self.scorer = GameScorer()
        # screen: a screen_reading.ScreenMemory (or True: a new one): the agent then knows the game only from
        # the screen (screen_reading.render of the board, the opponent's last move shown), followed in
        # context from the starting screen; it decides from its own board (play_game)
        self.screen = None
        self.screen_side = screen_side      # the screen shows the board from the agent's side (flipped as black)
        if screen is not None and screen is not False:
            from screen_reading import ScreenMemory
            self.screen = ScreenMemory() if screen is True else screen
            self.screen_stats = {'turns': 0, 'true board': 0, 'lost': 0, 'read alone': 0, 'illegal': 0}
        self.prioritizer = LearnableMovePrioritizer(db_path)
        # Multi-move patterns, learned from both players' moves
        self.sequences = SequencePatternLearner(db_path)
        # Position values, learned from every position of every game, whichever player is to move
        self.positions = PositionPatternLearner(db_path, self.sequences.pieces)
        # Kings' surroundings before mate, learned from both players' mates
        self.mates = MatePatternLearner(db_path)
        self.search = PatternSearch(self.sequences, self.SEARCH_SECONDS,
                                    positions=self.positions if self.use_position_values else None,
                                    selective=self.SEARCH_SELECTIVE, max_depth=self.SEARCH_DEPTH,
                                    bonus_cap=self.SEARCH_BONUS_CAP,
                                    mates=self.mates if self.use_mate_values else None,
                                    threats=self.SEARCH_THREATS)
        # Opponent: 'random' (capture-biased), 'stockfish' at a skill level (0-20), or 'self' (the agent's own
        # search; self_play can also be switched per game when a Stockfish opponent is configured)
        self.self_play = opponent == 'self'
        self.learn = True                   # False: games are played but nothing is recorded (evaluation)
        self.engine = None
        self.stockfish_time = stockfish_time
        if opponent == 'stockfish':
            import chess.engine
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.engine.configure({'Skill Level': stockfish_level})
        # Game-independent patterns, shared with the other games' trainers
        self.general = GeneralPatternLearner(universal_db_path(db_path), 'chess')
        self.adapter = ChessAdapter()
        self.last_general_features = None
        self.judge = None
        if self.use_judge:
            from move_judges import StockfishJudge
            self.judge = StockfishJudge(stockfish_path, self.JUDGE_DEPTH)
            self.general.judge = self.judge
        if self.use_refutations:
            self.general.LEARN_REFUTATIONS = True
            self.search.refutations = self.general.refutations
            self.search.adapter = self.adapter
        if self.use_tactics:
            self.general.TACTICS = True
        if self.use_move_history:
            from move_history import MoveHistory
            self.search.move_history = MoveHistory()

        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []
        self.game_results = []

        # Track draw types
        self.draw_types = {
            'stalemate': 0,
            'insufficient_material': 0,
            'fifty_moves': 0,
            'repetition': 0,
            'avoidable': 0,
            'unavoidable': 0
        }

    def play_ai_move(self, board, ai_color):
        """AI move selection using learned patterns + position evaluation"""
        if board.is_game_over():
            return None

        if self.use_search:
            # the game-independent patterns of every legal move, each compared with the others (described
            # alone, a move's gain and mobility classes would compare it with nothing)
            moves = list(board.legal_moves)
            features = {m: a['features'] for m, a in
                        zip(moves, self.general.analyze(self.adapter, board, ai_color, moves,
                                                        max_replies=float('inf')))}
            extra = {m: 0.0 for m in moves}
            if self.use_mobility_values:
                # e.g. 'mob=restricts': whether the move leaves the opponent fewer replies than most moves
                for m, f in features.items():
                    extra[m] += self.MOBILITY_WEIGHT * self.general.part_value(f.split('|')[2])
            if self.use_general_values:
                for m, f in features.items():
                    extra[m] += self.GENERAL_WEIGHT * self.general.value(f)
            move = self.search.choose(board, extra)
            # (the game-independent patterns learn from the move chosen)
            self.last_general_features = self.general.note_move(self.adapter, board, ai_color, move, features[move])
            return move

        legal_moves = list(board.legal_moves)
        legal_moves = self.prioritizer.sort_moves_by_priority(board, legal_moves)

        # Evaluate moves using BOTH the position evaluation AND learned patterns
        best_move = None
        best_score = float('-inf')

        candidates = legal_moves[:15]
        sequence_values = {}
        if self.use_sequences:
            context = self.sequences.context(board)
            sequence_values = {m: self.sequences.value(board, m, context) for m in legal_moves}
            extra = sorted(legal_moves[15:], key=lambda m: -sequence_values[m])[:self.SEQUENCE_EXTRA]
            candidates = candidates + extra
        # One-ply lookahead over every opponent reply: material, progress toward
        # checkmate, and finished games valued by their result (see ChessAdapter)
        analysis = self.general.analyze(self.adapter, board, ai_color, candidates,
                                        max_replies=float('inf'))
        value_now = evaluate(self.adapter, board, ai_color)

        for move, result in zip(candidates, analysis):
            features = result['features']
            # Evaluation after the opponent's best reply
            position_score = value_now + result['net']

            # Get learned priority for this move (0-100 scale)
            # Higher priority = historically better outcomes
            # The priority now includes observable game state features:
            # - repetition_count (AI will learn: count=2 → score=-10000 → avoid)
            # - moves_since_progress (AI will learn: 40+ → draw → make progress)
            # - total_material_level (AI will learn: low → draw → avoid trades)
            priority = self.prioritizer.get_move_priority(board, move)

            # Game-independent pattern value, same scale as the priority (-50..+50)
            priority += self.general.value(features)

            # Combine position evaluation + learned patterns
            # Priority scaled by 20 means a priority-72 move gets +1440 bonus
            score = position_score + (priority * 20)

            # Multi-move patterns: the material this move has been followed by, over the next plies
            score += self.SEQUENCE_WEIGHT * sequence_values.get(move, 0.0)

            if score > best_score:
                best_score = score
                best_move = move
                self.last_general_features = features

        self.general.note_move(self.adapter, board, ai_color, best_move, self.last_general_features)
        return best_move

    def play_opponent_move(self, board):
        """The agent's own search in self-play, else Stockfish if configured, else a random opponent with 30%
        capture bias"""
        if board.is_game_over():
            return None

        if self.self_play:
            return self.search.choose(board)

        if self.engine is not None:
            import chess.engine
            return self.engine.play(board, chess.engine.Limit(time=self.stockfish_time)).move

        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]

        if captures and random.random() < 0.3:
            return random.choice(captures)

        return random.choice(legal_moves)

    def _follow_screen(self, board, seen, ai_color):
        """
        The agent's board before its move: the screen shown now (the true board drawn, the opponent's last
        move highlighted), followed in context from the agent's board (the opponent's one move since); if it
        cannot be followed, the screen read alone, square by square (the agent keeps its side to move and
        castling rights as it knows them)
        """
        from screen_reading import board_fen, render
        stats = self.screen_stats
        stats['turns'] += 1
        if len(seen.move_stack) < len(board.move_stack):          # the opponent has moved since
            shown = render(board, lastmove=board.peek(), flipped=self.screen_side and ai_color == chess.BLACK)
            moves = self.screen.follow(seen, shown, plies=1, flipped=self.screen_flipped)
            if moves is not None:
                seen.push(moves[0])
            else:
                stats['lost'] += 1
                stats['read alone'] += 1
                seen = seen.copy()
                seen.set_board_fen(board_fen(self.screen.read(shown, self.screen_flipped)))
                seen.turn = ai_color
        stats['true board'] += seen.board_fen() == board.board_fen() and seen.turn == board.turn
        return seen

    def play_game(self, ai_color, verbose=False):
        """Play a complete game until natural conclusion"""
        board = chess.Board()
        game_moves = []
        general_features = []
        move_count = 0
        if self.self_play:
            # the search is deterministic: without random first plies, self-play games would repeat
            for _ in range(self.SELF_PLAY_RANDOM_PLIES):
                board.push(random.choice(list(board.legal_moves)))

        # Play until game naturally ends (checkmate, stalemate, etc.)
        # No artificial move limit - let games reach their conclusion
        if self.screen is not None:
            from screen_reading import render
            assert not self.self_play, 'screen play is for games against an opponent'
            flip = self.screen_side and ai_color == chess.BLACK
            first = self.screen.learn_start(render(chess.Board(), flipped=flip), 'the starting screen') \
                if not self.screen.examples else None
            # which way the game's screens face: from the rank labels of the starting screen (or, unknown,
            # found in context on every screen: screen_reading.ScreenMemory.follow)
            from screen_reading import orientation
            self.screen_flipped = first['flipped'] if first and 'flipped' in first else \
                orientation(render(chess.Board(), flipped=flip))
            seen = chess.Board()            # the agent's own board: what it has made of the screens

        while not board.is_game_over():
            fen_before = board.fen()

            if board.turn == ai_color:
                if self.screen is not None:
                    seen = self._follow_screen(board, seen, ai_color)
                    move = self.play_ai_move(seen, ai_color)
                    if move is not None:
                        seen.push(move)
                        if move not in board.legal_moves:      # decided from a board read wrong
                            self.screen_stats['illegal'] += 1
                            move = random.choice(list(board.legal_moves))
                else:
                    move = self.play_ai_move(board, ai_color)
                if move:
                    move_san = board.san(move)
                    board.push(move)
                    game_moves.append((fen_before, move.uci(), move_san))
                    general_features.append(self.last_general_features)

                    if verbose:
                        ai_mat = self.scorer._calculate_material(board, ai_color)
                        opp_mat = self.scorer._calculate_material(board, not ai_color)
                        advantage = ai_mat - opp_mat
                        print(f"  AI: {move_san:8s} | Advantage: {advantage:+6.0f}")
            else:
                move = self.play_opponent_move(board)
                if move:
                    # Get SAN notation BEFORE pushing the move
                    move_san = board.san(move)
                    board.push(move)
                    move_count += 1
                    if verbose:
                        print(f"  Opp: {move_san:8s}")

        # Calculate final score
        rounds_played = board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(board, ai_color, rounds_played)

        # Track draw types for analysis
        if result == 'draw':
            # Calculate material to determine if draw was avoidable
            ai_mat = self.scorer._calculate_material(board, ai_color)
            opp_mat = self.scorer._calculate_material(board, not ai_color)
            material_advantage = ai_mat - opp_mat

            if board.is_stalemate():
                self.draw_types['stalemate'] += 1
                if material_advantage > 10:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1

            elif board.is_insufficient_material():
                self.draw_types['insufficient_material'] += 1
                self.draw_types['unavoidable'] += 1

            elif board.is_fifty_moves():
                self.draw_types['fifty_moves'] += 1
                if material_advantage > 20:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1

            elif board.is_repetition():
                self.draw_types['repetition'] += 1
                if material_advantage > -20:
                    self.draw_types['avoidable'] += 1
                else:
                    self.draw_types['unavoidable'] += 1

        # Update statistics
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        self.score_history.append(final_score)

        # Record for learning (sequences, positions and mates learn from both players' moves, so a self-play game
        # is learned from both colours)
        if self.learn:
            self.prioritizer.record_game_moves(game_moves, ai_color, result, final_score)
            self.general.record_game(general_features, result, 'chess')
            self.sequences.record_game(board.move_stack, ai_color, result)
            self.positions.record_game(board.move_stack, ai_color, result,
                                       judge=self.judge if self.judge_positions else None)
            self.mates.record_game(board.move_stack)

        # Store result details
        self.game_results.append({
            'game_num': self.games_played,
            'result': result,
            'score': final_score,
            'rounds': rounds_played,
            'ai_color': 'white' if ai_color == chess.WHITE else 'black'
        })

        return result, final_score, rounds_played

    def train(self, num_games, verbose=False, progress_interval=10):
        """Train for multiple games"""
        print("=" * 70)
        print(f"HEADLESS TRAINING - {num_games} GAMES")
        print("=" * 70)

        start_time = time.time()

        for i in range(num_games):
            ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "WHITE" if ai_color == chess.WHITE else "BLACK"

            if verbose or (i + 1) % progress_interval == 0:
                print(f"\n--- Game {i+1}/{num_games} (AI plays {color_str}) ---")

            result, score, rounds = self.play_game(ai_color, verbose=verbose)

            if (i + 1) % progress_interval == 0 or verbose:
                win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0
                avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
                print(f"  Result: {result.upper():6s} | Score: {score:+7.0f} | Rounds: {rounds:2d}")
                print(f"  Progress: {self.wins}W-{self.losses}L-{self.draws}D | Win Rate: {win_rate:.1f}% | Avg Score: {avg_score:+.0f}")

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
        print(f"Results: {self.wins}W-{self.losses}L-{self.draws}D")
        print(f"Win Rate: {(self.wins/self.games_played*100):.1f}%")
        print(f"Average Score: {sum(self.score_history)/len(self.score_history):+.0f}")
        print(f"Best Score: {max(self.score_history):+.0f}")
        print(f"Worst Score: {min(self.score_history):+.0f}")

        # Draw type analysis
        if self.draws > 0:
            print(f"\nDraw Analysis ({self.draws} total draws):")
            print(f"  Stalemate:              {self.draw_types['stalemate']:3d} ({self.draw_types['stalemate']/self.draws*100:5.1f}%)")
            print(f"  Insufficient Material:  {self.draw_types['insufficient_material']:3d} ({self.draw_types['insufficient_material']/self.draws*100:5.1f}%)")
            print(f"  Fifty-Move Rule:        {self.draw_types['fifty_moves']:3d} ({self.draw_types['fifty_moves']/self.draws*100:5.1f}%)")
            print(f"  Threefold Repetition:   {self.draw_types['repetition']:3d} ({self.draw_types['repetition']/self.draws*100:5.1f}%)")
            print(f"  ---")
            print(f"  AVOIDABLE (AI's fault): {self.draw_types['avoidable']:3d} ({self.draw_types['avoidable']/self.draws*100:5.1f}%)")
            print(f"  Unavoidable:            {self.draw_types['unavoidable']:3d} ({self.draw_types['unavoidable']/self.draws*100:5.1f}%)")

        # Learning statistics
        stats = self.prioritizer.get_statistics()
        print(f"\nLearning Statistics:")
        print(f"  Patterns Learned: {stats['patterns_learned']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg Win Rate: {stats['avg_win_rate']:.1%}")

        print("\n" + "=" * 70)

    def export_results(self, filename=None):
        """Export results to JSON"""
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'summary': {
                'games_played': self.games_played,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'win_rate': self.wins / self.games_played if self.games_played > 0 else 0,
                'avg_score': sum(self.score_history) / len(self.score_history) if self.score_history else 0
            },
            'score_history': self.score_history,
            'games': self.game_results
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results exported to {filename}")

    def export_csv(self, filename=None):
        """Export results to CSV"""
        if filename is None:
            filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['game_num', 'result', 'score', 'rounds', 'ai_color'])
            writer.writeheader()
            writer.writerows(self.game_results)

        print(f"✓ Results exported to {filename}")

    def show_top_patterns(self, limit=10):
        """Display top learned patterns"""
        print("\n" + "=" * 70)
        print(f"TOP {limit} LEARNED PATTERNS (by priority)")
        print("=" * 70)

        patterns = self.prioritizer.get_top_patterns(limit)

        if not patterns:
            print("No patterns learned yet!")
            return

        print(f"{'Piece':<10} {'Category':<14} {'Distance':<10} {'Seen':<6} {'Win%':<8} {'Conf':<6} {'Priority':<8}")
        print("-" * 70)

        for row in patterns:
            piece, category, dist, seen, win_rate, conf, priority = row
            print(f"{piece:<10} {category:<14} {dist:<10} {seen:<6} {win_rate:<7.1%} {conf:<6.2f} {priority:<8.1f}")

        for title, avoid in (('PURSUE', False), ('AVOID', True)):
            print(f"\nMULTI-MOVE PATTERNS TO {title} (value: expected material gain, a pawn = 10)")
            for pattern, value, seen in self.sequences.top_patterns(limit, avoid=avoid):
                print(f"  {value:+7.1f}  seen {seen:6.1f}  {pattern}")

        for title, worst in (('BEST', False), ('WORST', True)):
            print(f"\nPOSITION PATTERNS, {title} FOR THE SIDE TO MOVE (m: its piece, t: the other side's)")
            for pattern, weight, seen in self.positions.top_patterns(limit, worst=worst):
                print(f"  {weight:+7.1f}  seen {seen:7.0f}  {pattern}")

        print("\nKING SURROUNDINGS MOST OFTEN FOLLOWED BY MATE (free/blocked/attacked/off-board squares, + check)")
        for pattern, rate, seen in self.mates.top_patterns(limit):
            print(f"  {rate:6.1%}  seen {seen:7.0f}  {pattern}")

        print("=" * 70)

    def close(self):
        """Clean up resources"""
        self.prioritizer.close()
        self.general.close()
        self.mates.close()
        self.positions.close()
        self.sequences.close()
        if self.engine is not None:
            self.engine.quit()
        if self.judge is not None:
            self.judge.close()


def main():
    """Run headless trainer"""
    import argparse

    parser = argparse.ArgumentParser(description='Chess AI Headless Trainer')
    parser.add_argument('num_games', type=int, help='Number of games to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed game output')
    parser.add_argument('--progress', type=int, default=10, help='Progress update interval (default: 10)')
    parser.add_argument('--export-json', action='store_true', help='Export results to JSON')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--show-patterns', type=int, default=0, metavar='N', help='Show top N patterns after training')
    parser.add_argument('--opponent', choices=['random', 'stockfish', 'self'], default='random',
                        help="Opponent: 'self' is the agent's own search (default: random)")
    parser.add_argument('--level', type=int, default=0, help='Stockfish skill level 0-20 (default: 0)')
    parser.add_argument('--stockfish-path', default='/usr/games/stockfish', help='Stockfish binary')
    parser.add_argument('--db', default='headless_training.db', help='Training database (default: headless_training.db)')
    parser.add_argument('--screen', action='store_true',
                        help='The agent knows the game only from the screen (screen_reading), read from the start')
    parser.add_argument('--screen-side', action='store_true',
                        help='With --screen: the screen shows the board from the agent\'s side (flipped as black)')

    args = parser.parse_args()

    trainer = HeadlessTrainer(args.db, opponent=args.opponent, stockfish_level=args.level,
                              stockfish_path=args.stockfish_path, screen=True if args.screen else None,
                              screen_side=args.screen_side)

    try:
        trainer.train(args.num_games, verbose=args.verbose, progress_interval=args.progress)

        if args.export_json:
            trainer.export_results()

        if args.export_csv:
            trainer.export_csv()

        if args.show_patterns > 0:
            trainer.show_top_patterns(args.show_patterns)

    finally:
        trainer.close()


if __name__ == '__main__':
    main()
