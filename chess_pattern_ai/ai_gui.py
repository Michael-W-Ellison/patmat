#!/usr/bin/env python3
"""
Chess Pattern Recognition AI - GUI Interface

Features:
- Visual chess board showing current game
- Real-time metrics and statistics
- Training controls (single game or multiple games)
- Learning progress graphs
- Pattern statistics viewer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import chess
import chess.svg
import chess.engine
from PIL import Image, ImageTk
import io
import cairosvg
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import time


class ChessAIGUI:
    """Main GUI application for Chess Pattern Recognition AI"""

    def __init__(self, root):
        self.root = root
        self.root.title("Chess Pattern Recognition AI - Dashboard")
        self.root.geometry("1400x900")

        # Initialize components
        self.scorer = GameScorer()
        self.prioritizer = LearnableMovePrioritizer('gui_training.db')

        # Game state
        self.current_board = chess.Board()
        self.game_history = []
        self.ai_color = chess.WHITE
        self.game_active = False
        self.training_active = False
        self.training_thread = None

        # Statistics tracking
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.score_history = []
        self.material_history = []

        # Stockfish integration
        self.stockfish_path = '/usr/games/stockfish'
        self.stockfish_level = 5  # Default skill level (0-20)
        self.use_stockfish = False  # Toggle between Stockfish and random
        self.engine = None
        self.consecutive_wins = 0  # For progressive difficulty

        # Try to initialize Stockfish
        self._init_stockfish()

        # Queue for thread communication
        self.update_queue = queue.Queue()

        # Create GUI
        self.create_widgets()

        # Start update loop
        self.process_queue()

    def _init_stockfish(self):
        """Initialize Stockfish engine"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({"Skill Level": self.stockfish_level})
            self.use_stockfish = True
            print(f"✓ Stockfish initialized at level {self.stockfish_level}")
            print(f"  Path: {self.stockfish_path}")
        except FileNotFoundError:
            print(f"⚠ Stockfish not found at: {self.stockfish_path}")
            print("  Trying common alternative paths...")
            # Try alternative paths
            for path in ['/usr/local/bin/stockfish', 'stockfish', '/opt/homebrew/bin/stockfish']:
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(path)
                    self.engine.configure({"Skill Level": self.stockfish_level})
                    self.stockfish_path = path
                    self.use_stockfish = True
                    print(f"✓ Stockfish found at: {path}")
                    return
                except:
                    continue
            print("  Stockfish not available - will use random opponent")
            self.use_stockfish = False
        except Exception as e:
            print(f"⚠ Error initializing Stockfish: {e}")
            print("  Will use random opponent instead")
            self.use_stockfish = False

    def update_stockfish_level(self):
        """Update Stockfish skill level"""
        if self.engine and self.use_stockfish:
            try:
                new_level = int(self.stockfish_level_var.get())
                self.stockfish_level = max(0, min(20, new_level))
                self.engine.configure({"Skill Level": self.stockfish_level})
                self.consecutive_wins = 0  # Reset progression
                print(f"✓ Stockfish level updated to {self.stockfish_level}")
            except Exception as e:
                print(f"⚠ Error updating Stockfish level: {e}")

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main container with left and right panels
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Chess board and game info
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)

        # Right panel - Metrics and controls
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)

        # === LEFT PANEL ===
        self.create_board_panel(left_panel)
        self.create_game_info_panel(left_panel)

        # === RIGHT PANEL ===
        self.create_controls_panel(right_panel)
        self.create_metrics_panel(right_panel)
        self.create_graphs_panel(right_panel)

    def create_board_panel(self, parent):
        """Create chess board display"""
        board_frame = ttk.LabelFrame(parent, text="Chess Board", padding=10)
        board_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Board canvas
        self.board_label = tk.Label(board_frame, bg='white')
        self.board_label.pack()

        # Update board display
        self.update_board_display()

    def create_game_info_panel(self, parent):
        """Create game information display"""
        info_frame = ttk.LabelFrame(parent, text="Game Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Move history
        ttk.Label(info_frame, text="Move History:").pack(anchor=tk.W)
        self.move_history_text = scrolledtext.ScrolledText(
            info_frame, height=10, width=40, wrap=tk.WORD
        )
        self.move_history_text.pack(fill=tk.BOTH, expand=True)

    def create_controls_panel(self, parent):
        """Create control buttons"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Single game controls
        single_frame = ttk.Frame(control_frame)
        single_frame.pack(fill=tk.X, pady=5)

        self.start_game_btn = ttk.Button(
            single_frame, text="▶ Start Game", command=self.start_single_game
        )
        self.start_game_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(
            single_frame, text="⟳ Reset Board", command=self.reset_board
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Training controls
        train_frame = ttk.LabelFrame(control_frame, text="Training", padding=5)
        train_frame.pack(fill=tk.X, pady=10)

        ttk.Label(train_frame, text="Number of games:").pack(side=tk.LEFT, padx=5)
        self.num_games_var = tk.StringVar(value="10")
        num_games_entry = ttk.Entry(train_frame, textvariable=self.num_games_var, width=10)
        num_games_entry.pack(side=tk.LEFT, padx=5)

        self.start_training_btn = ttk.Button(
            train_frame, text="🎓 Start Training", command=self.start_training
        )
        self.start_training_btn.pack(side=tk.LEFT, padx=5)

        self.stop_training_btn = ttk.Button(
            train_frame, text="⏹ Stop Training", command=self.stop_training, state=tk.DISABLED
        )
        self.stop_training_btn.pack(side=tk.LEFT, padx=5)

        # Stockfish controls
        stockfish_frame = ttk.LabelFrame(control_frame, text="Opponent Settings", padding=5)
        stockfish_frame.pack(fill=tk.X, pady=10)

        # Opponent type selector
        ttk.Label(stockfish_frame, text="Opponent:").pack(side=tk.LEFT, padx=5)

        self.opponent_var = tk.StringVar(value="stockfish" if self.use_stockfish else "random")
        opponent_combo = ttk.Combobox(
            stockfish_frame,
            textvariable=self.opponent_var,
            values=["random", "stockfish"],
            state="readonly",
            width=10
        )
        opponent_combo.pack(side=tk.LEFT, padx=5)
        opponent_combo.bind("<<ComboboxSelected>>", lambda e: self._update_opponent_type())

        # Stockfish level control
        ttk.Label(stockfish_frame, text="Stockfish Level (0-20):").pack(side=tk.LEFT, padx=5)
        self.stockfish_level_var = tk.StringVar(value=str(self.stockfish_level))
        level_spin = ttk.Spinbox(
            stockfish_frame,
            from_=0, to=20,
            textvariable=self.stockfish_level_var,
            width=5,
            command=self.update_stockfish_level
        )
        level_spin.pack(side=tk.LEFT, padx=5)

        # Progressive mode checkbox
        self.progressive_var = tk.BooleanVar(value=False)
        progressive_check = ttk.Checkbutton(
            stockfish_frame,
            text="Progressive (auto-advance after 10 wins)",
            variable=self.progressive_var
        )
        progressive_check.pack(side=tk.LEFT, padx=10)

        # Status label
        status_text = "✓ Stockfish available" if self.use_stockfish else "⚠ Stockfish not found"
        self.stockfish_status_label = ttk.Label(
            stockfish_frame,
            text=status_text,
            foreground="green" if self.use_stockfish else "orange"
        )
        self.stockfish_status_label.pack(side=tk.LEFT, padx=5)

    def _update_opponent_type(self):
        """Update opponent type when combobox changes"""
        opponent = self.opponent_var.get()
        if opponent == "stockfish" and not self.use_stockfish:
            messagebox.showwarning(
                "Stockfish Not Available",
                "Stockfish is not installed or not found.\n\n"
                "Will use random opponent instead.\n\n"
                "To use Stockfish:\n"
                "  - Install: sudo apt-get install stockfish\n"
                "  - Or download from: https://stockfishchess.org/"
            )
            self.opponent_var.set("random")

    def create_metrics_panel(self, parent):
        """Create real-time metrics display"""
        metrics_frame = ttk.LabelFrame(parent, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Statistics
        stats_frame = ttk.Frame(metrics_frame)
        stats_frame.pack(fill=tk.X, pady=5)

        # Games played
        ttk.Label(stats_frame, text="Games Played:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.games_played_label = ttk.Label(stats_frame, text="0", font=('Arial', 12, 'bold'))
        self.games_played_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Win rate
        ttk.Label(stats_frame, text="Wins:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.wins_label = ttk.Label(stats_frame, text="0", foreground='green')
        self.wins_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(stats_frame, text="Losses:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.losses_label = ttk.Label(stats_frame, text="0", foreground='red')
        self.losses_label.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(stats_frame, text="Draws:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.draws_label = ttk.Label(stats_frame, text="0", foreground='blue')
        self.draws_label.grid(row=3, column=1, sticky=tk.W, padx=5)

        # Current game metrics
        current_frame = ttk.LabelFrame(metrics_frame, text="Current Game", padding=5)
        current_frame.pack(fill=tk.X, pady=10)

        ttk.Label(current_frame, text="Material Advantage:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.material_label = ttk.Label(current_frame, text="+0", font=('Arial', 11, 'bold'))
        self.material_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(current_frame, text="Move Number:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.move_num_label = ttk.Label(current_frame, text="0")
        self.move_num_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Learning metrics
        learning_frame = ttk.LabelFrame(metrics_frame, text="Learning Progress", padding=5)
        learning_frame.pack(fill=tk.X, pady=10)

        stats = self.prioritizer.get_statistics()

        ttk.Label(learning_frame, text="Patterns Learned:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.patterns_label = ttk.Label(learning_frame, text=str(stats['patterns_learned']))
        self.patterns_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(learning_frame, text="Avg Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.confidence_label = ttk.Label(learning_frame, text=f"{stats['avg_confidence']:.2f}")
        self.confidence_label.grid(row=1, column=1, sticky=tk.W, padx=5)

    def create_graphs_panel(self, parent):
        """Create graphs for metrics visualization"""
        graph_frame = ttk.LabelFrame(parent, text="Training Progress", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=80)

        # Score history subplot
        self.score_ax = self.fig.add_subplot(211)
        self.score_ax.set_title("Differential Scores")
        self.score_ax.set_xlabel("Game Number")
        self.score_ax.set_ylabel("Score")
        self.score_ax.grid(True, alpha=0.3)

        # Win rate subplot
        self.winrate_ax = self.fig.add_subplot(212)
        self.winrate_ax.set_title("Win/Loss/Draw Distribution")
        self.winrate_ax.set_xlabel("Game Number")
        self.winrate_ax.set_ylabel("Count")
        self.winrate_ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_board_display(self):
        """Update the chess board visualization"""
        try:
            # Generate SVG
            svg_data = chess.svg.board(self.current_board, size=400)

            # Convert SVG to PNG
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

            # Load into PIL
            image = Image.open(io.BytesIO(png_data))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update label
            self.board_label.configure(image=photo)
            self.board_label.image = photo  # Keep reference

        except Exception as e:
            print(f"Error updating board: {e}")
            # Fallback to text representation
            board_text = str(self.current_board)
            self.board_label.configure(text=board_text)

    def update_metrics(self):
        """Update all metric displays"""
        # Update statistics
        self.games_played_label.configure(text=str(self.games_played))
        self.wins_label.configure(text=str(self.wins))
        self.losses_label.configure(text=str(self.losses))
        self.draws_label.configure(text=str(self.draws))

        # Update current game material
        if self.current_board:
            ai_mat = self.scorer._calculate_material(self.current_board, self.ai_color)
            opp_mat = self.scorer._calculate_material(self.current_board, not self.ai_color)
            advantage = ai_mat - opp_mat

            color = 'green' if advantage > 0 else 'red' if advantage < 0 else 'black'
            self.material_label.configure(text=f"{advantage:+.0f}", foreground=color)

        self.move_num_label.configure(text=str(self.current_board.fullmove_number))

        # Update learning statistics
        stats = self.prioritizer.get_statistics()
        self.patterns_label.configure(text=str(stats['patterns_learned']))
        self.confidence_label.configure(text=f"{stats['avg_confidence']:.2f}")

    def update_graphs(self):
        """Update progress graphs"""
        if not self.score_history:
            return

        # Clear axes
        self.score_ax.clear()
        self.winrate_ax.clear()

        # Score history
        self.score_ax.plot(range(1, len(self.score_history) + 1), self.score_history,
                          marker='o', linestyle='-', color='blue', alpha=0.7)
        self.score_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.score_ax.set_title("Differential Scores Over Time")
        self.score_ax.set_xlabel("Game Number")
        self.score_ax.set_ylabel("Score")
        self.score_ax.grid(True, alpha=0.3)

        # Win/Loss/Draw counts
        games = range(1, self.games_played + 1)
        wins_cumulative = []
        losses_cumulative = []
        draws_cumulative = []

        w, l, d = 0, 0, 0
        for i, score in enumerate(self.score_history):
            # Estimate result from score (simplified)
            if score > 500:
                w += 1
            elif score < -500:
                l += 1
            else:
                d += 1
            wins_cumulative.append(w)
            losses_cumulative.append(l)
            draws_cumulative.append(d)

        if games:
            self.winrate_ax.plot(games, wins_cumulative, label='Wins', color='green', linewidth=2)
            self.winrate_ax.plot(games, losses_cumulative, label='Losses', color='red', linewidth=2)
            self.winrate_ax.plot(games, draws_cumulative, label='Draws', color='blue', linewidth=2)
            self.winrate_ax.legend()
            self.winrate_ax.set_title("Cumulative Results")
            self.winrate_ax.set_xlabel("Game Number")
            self.winrate_ax.set_ylabel("Count")
            self.winrate_ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def play_ai_move(self):
        """Make an AI move"""
        if self.current_board.is_game_over():
            return None

        legal_moves = list(self.current_board.legal_moves)

        # Sort by learned priority
        legal_moves = self.prioritizer.sort_moves_by_priority(self.current_board, legal_moves)

        # Simple evaluation: pick best move based on material
        best_move = None
        best_score = -999999

        for move in legal_moves[:15]:  # Limit search
            self.current_board.push(move)

            # Evaluate
            ai_mat = self.scorer._calculate_material(self.current_board, self.ai_color)
            opp_mat = self.scorer._calculate_material(self.current_board, not self.ai_color)
            score = ai_mat - opp_mat

            self.current_board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def play_opponent_move(self):
        """Make opponent move (Stockfish or random)"""
        if self.current_board.is_game_over():
            return None

        # Use Stockfish if available and selected
        opponent = self.opponent_var.get() if hasattr(self, 'opponent_var') else "random"

        if opponent == "stockfish" and self.engine and self.use_stockfish:
            try:
                result = self.engine.play(self.current_board, chess.engine.Limit(time=0.1))
                return result.move
            except Exception as e:
                print(f"⚠ Stockfish error: {e}, falling back to random")
                # Fall through to random move

        # Random opponent (fallback or selected)
        legal_moves = list(self.current_board.legal_moves)
        if not legal_moves:
            return None

        # 30% chance of capture if available
        captures = [m for m in legal_moves if self.current_board.is_capture(m)]
        if captures and random.random() < 0.3:
            return random.choice(captures)

        return random.choice(legal_moves)

    def play_full_game(self):
        """Play a complete game"""
        self.current_board = chess.Board()
        game_moves = []
        move_count = 0

        while not self.current_board.is_game_over() and move_count < 60:
            fen_before = self.current_board.fen()

            if self.current_board.turn == self.ai_color:
                # AI move
                move = self.play_ai_move()
                if move:
                    move_san = self.current_board.san(move)
                    self.current_board.push(move)
                    game_moves.append((fen_before, move.uci(), move_san))

                    # Update display
                    self.update_queue.put(('board', None))
                    self.update_queue.put(('move', f"{move_count//2 + 1}. {move_san}"))
                    time.sleep(0.1)  # Slow down for visualization
            else:
                # Opponent move
                move = self.play_opponent_move()
                if move:
                    self.current_board.push(move)
                    move_count += 1
                    self.update_queue.put(('board', None))
                    time.sleep(0.05)

        # Game over - record results
        rounds_played = self.current_board.fullmove_number
        final_score, result = self.scorer.calculate_final_score(
            self.current_board, self.ai_color, rounds_played
        )

        # Update statistics
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        self.score_history.append(final_score)

        # Progressive difficulty (if enabled)
        if self.progressive_var.get() if hasattr(self, 'progressive_var') else False:
            if result == 'win':
                self.consecutive_wins += 1
                if self.consecutive_wins >= 10 and self.stockfish_level < 20:
                    self.stockfish_level += 1
                    self.consecutive_wins = 0
                    if self.engine and self.use_stockfish:
                        self.engine.configure({"Skill Level": self.stockfish_level})
                        self.stockfish_level_var.set(str(self.stockfish_level))
                        self.update_queue.put(('move', f"\n🎉 Level up! Now Stockfish level {self.stockfish_level}\n"))
            else:
                self.consecutive_wins = 0

        # Record for learning
        self.prioritizer.record_game_moves(game_moves, self.ai_color, result, final_score)

        # Update displays
        self.update_queue.put(('metrics', None))
        self.update_queue.put(('graphs', None))
        self.update_queue.put(('move', f"\n=== Game Over: {result.upper()} ==="))
        self.update_queue.put(('move', f"Score: {final_score:.0f}\n"))

        return result, final_score

    def start_single_game(self):
        """Start a single game"""
        if self.game_active or self.training_active:
            messagebox.showwarning("Game Active", "A game or training session is already running")
            return

        self.game_active = True
        self.start_game_btn.configure(state=tk.DISABLED)

        def game_thread():
            self.reset_board()
            self.play_full_game()
            self.game_active = False
            self.update_queue.put(('enable_start', None))

        threading.Thread(target=game_thread, daemon=True).start()

    def start_training(self):
        """Start training multiple games"""
        if self.game_active or self.training_active:
            messagebox.showwarning("Training Active", "A game or training session is already running")
            return

        try:
            num_games = int(self.num_games_var.get())
            if num_games < 1:
                raise ValueError()
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid number of games (≥ 1)")
            return

        self.training_active = True
        self.start_training_btn.configure(state=tk.DISABLED)
        self.stop_training_btn.configure(state=tk.NORMAL)
        self.start_game_btn.configure(state=tk.DISABLED)

        def training_thread():
            for i in range(num_games):
                if not self.training_active:
                    break

                self.update_queue.put(('move', f"\n{'='*50}\nGame {i+1}/{num_games}\n{'='*50}\n"))
                self.reset_board()
                self.play_full_game()

            self.training_active = False
            self.update_queue.put(('training_complete', None))

        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stop the training session"""
        self.training_active = False
        self.stop_training_btn.configure(state=tk.DISABLED)

    def reset_board(self):
        """Reset the chess board"""
        self.current_board = chess.Board()
        self.game_history = []
        self.update_board_display()
        self.move_history_text.delete(1.0, tk.END)

    def process_queue(self):
        """Process update queue from worker threads"""
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()

                if msg_type == 'board':
                    self.update_board_display()
                elif msg_type == 'move':
                    self.move_history_text.insert(tk.END, data + "\n")
                    self.move_history_text.see(tk.END)
                elif msg_type == 'metrics':
                    self.update_metrics()
                elif msg_type == 'graphs':
                    self.update_graphs()
                elif msg_type == 'enable_start':
                    self.start_game_btn.configure(state=tk.NORMAL)
                elif msg_type == 'training_complete':
                    self.start_training_btn.configure(state=tk.NORMAL)
                    self.stop_training_btn.configure(state=tk.DISABLED)
                    self.start_game_btn.configure(state=tk.NORMAL)
                    messagebox.showinfo("Training Complete",
                                      f"Completed {self.games_played} games!\n"
                                      f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

    def on_closing(self):
        """Clean up when closing"""
        self.training_active = False
        self.game_active = False
        self.prioritizer.close()

        # Cleanup Stockfish engine
        if self.engine:
            try:
                self.engine.quit()
                print("✓ Stockfish engine closed")
            except:
                pass

        self.root.destroy()


def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = ChessAIGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
