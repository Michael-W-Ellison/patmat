#!/usr/bin/env python3
"""
Game Launcher GUI - Unified training interface for all 13 games

Features:
- Launch training for any game with custom parameters
- Real-time statistics (win rate, games played, average score)
- Progress tracking
- View learned patterns
- No board rendering needed - just stats!
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import subprocess
import os
import sqlite3
from pathlib import Path


class GameLauncherGUI:
    """Unified GUI for launching and monitoring game training"""

    # Game configurations
    GAMES = {
        'Chess': {
            'trainer': 'python3 chess_progressive_trainer.py',
            'db': 'chess_training.db',
            'default_games': 50
        },
        'Checkers': {
            'trainer': 'python3 checkers/checkers_headless_trainer.py',
            'db': 'checkers_training.db',
            'default_games': 100
        },
        'Go (9x9)': {
            'trainer': 'python3 go/go_headless_trainer.py --size 9',
            'db': 'go_training.db',
            'default_games': 50
        },
        'Othello': {
            'trainer': 'python3 othello/othello_headless_trainer.py',
            'db': 'othello_training.db',
            'default_games': 100
        },
        'Connect Four': {
            'trainer': 'python3 connect4/connect4_headless_trainer.py',
            'db': 'connect4_training.db',
            'default_games': 200
        },
        'Gomoku': {
            'trainer': 'python3 gomoku/gomoku_headless_trainer.py',
            'db': 'gomoku_training.db',
            'default_games': 100
        },
        'Hex': {
            'trainer': 'python3 hex/hex_headless_trainer.py',
            'db': 'hex_training.db',
            'default_games': 100
        },
        'Dots and Boxes': {
            'trainer': 'python3 dots_boxes/dots_boxes_headless_trainer.py',
            'db': 'dots_boxes_training.db',
            'default_games': 200
        },
        'Breakthrough': {
            'trainer': 'python3 breakthrough/breakthrough_headless_trainer.py',
            'db': 'breakthrough_training.db',
            'default_games': 100
        },
        'Pentago': {
            'trainer': 'python3 pentago/pentago_headless_trainer.py',
            'db': 'pentago_training.db',
            'default_games': 100
        },
        'Nine Men\'s Morris': {
            'trainer': 'python3 morris/morris_headless_trainer.py',
            'db': 'morris_training.db',
            'default_games': 50
        },
        'Lines of Action': {
            'trainer': 'python3 loa/loa_headless_trainer.py',
            'db': 'loa_training.db',
            'default_games': 50
        },
        'Arimaa': {
            'trainer': 'python3 arimaa/arimaa_headless_trainer.py',
            'db': 'arimaa_training.db',
            'default_games': 20
        }
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Pattern Recognition AI - Game Launcher")
        self.root.geometry("1200x800")

        # State tracking
        self.active_trainings = {}  # game_name -> process
        self.update_queue = queue.Queue()

        # Create UI
        self.create_widgets()

        # Start update loop
        self.update_ui()

    def create_widgets(self):
        """Create GUI widgets"""

        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(main_frame, text="Pattern Recognition AI - Game Launcher",
                         font=('Arial', 16, 'bold'))
        title.pack(pady=(0, 10))

        # Subtitle
        subtitle = ttk.Label(main_frame,
                           text="Launch training for any of 13 games and monitor progress",
                           font=('Arial', 10))
        subtitle.pack(pady=(0, 20))

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Game Launcher
        self.launcher_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.launcher_tab, text="Launch Training")
        self.create_launcher_tab()

        # Tab 2: Active Training
        self.monitor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.monitor_tab, text="Training Monitor")
        self.create_monitor_tab()

        # Tab 3: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Game Statistics")
        self.create_stats_tab()

    def create_launcher_tab(self):
        """Create game launcher tab"""

        # Instructions
        ttk.Label(self.launcher_tab,
                 text="Select a game and click 'Start Training' to begin",
                 font=('Arial', 10)).pack(pady=10)

        # Scrollable frame for games
        canvas = tk.Canvas(self.launcher_tab)
        scrollbar = ttk.Scrollbar(self.launcher_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Create game cards
        for game_name, config in self.GAMES.items():
            self.create_game_card(scrollable_frame, game_name, config)

    def create_game_card(self, parent, game_name, config):
        """Create a card for each game"""

        card = ttk.LabelFrame(parent, text=game_name, padding=15)
        card.pack(fill=tk.X, pady=5, padx=5)

        # Info frame
        info_frame = ttk.Frame(card)
        info_frame.pack(fill=tk.X)

        # Database path
        ttk.Label(info_frame, text=f"Database: {config['db']}",
                 font=('Arial', 9)).pack(anchor=tk.W)

        # Check if database exists
        db_path = os.path.join(os.path.dirname(__file__), config['db'])
        if os.path.exists(db_path):
            # Get pattern count
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM learned_move_patterns")
                count = cursor.fetchone()[0]
                conn.close()
                ttk.Label(info_frame, text=f"Patterns learned: {count}",
                         foreground='green', font=('Arial', 9)).pack(anchor=tk.W)
            except:
                pass
        else:
            ttk.Label(info_frame, text="No training data yet",
                     foreground='orange', font=('Arial', 9)).pack(anchor=tk.W)

        # Controls frame
        controls = ttk.Frame(card)
        controls.pack(fill=tk.X, pady=(10, 0))

        # Number of games
        ttk.Label(controls, text="Games:").pack(side=tk.LEFT, padx=(0, 5))
        games_var = tk.IntVar(value=config['default_games'])
        games_spinbox = ttk.Spinbox(controls, from_=1, to=10000,
                                    textvariable=games_var, width=10)
        games_spinbox.pack(side=tk.LEFT, padx=(0, 20))

        # Start button
        start_btn = ttk.Button(controls, text="Start Training",
                              command=lambda: self.start_training(game_name, games_var.get()))
        start_btn.pack(side=tk.LEFT, padx=5)

        # Stop button
        stop_btn = ttk.Button(controls, text="Stop",
                             command=lambda: self.stop_training(game_name),
                             state=tk.DISABLED)
        stop_btn.pack(side=tk.LEFT, padx=5)

        # View patterns button
        view_btn = ttk.Button(controls, text="View Patterns",
                             command=lambda: self.view_patterns(config['db']))
        view_btn.pack(side=tk.LEFT, padx=5)

        # Store button references
        if not hasattr(self, 'game_buttons'):
            self.game_buttons = {}
        self.game_buttons[game_name] = {
            'start': start_btn,
            'stop': stop_btn,
            'games_var': games_var
        }

    def create_monitor_tab(self):
        """Create training monitor tab"""

        ttk.Label(self.monitor_tab, text="Active Training Sessions",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(self.monitor_tab,
                                                     height=30,
                                                     width=120,
                                                     font=('Courier', 9))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tags for colored output
        self.output_text.tag_config('game_name', foreground='blue', font=('Courier', 9, 'bold'))
        self.output_text.tag_config('success', foreground='green')
        self.output_text.tag_config('error', foreground='red')
        self.output_text.tag_config('info', foreground='black')

    def create_stats_tab(self):
        """Create statistics tab"""

        ttk.Label(self.stats_tab, text="Training Statistics Summary",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        # Refresh button
        ttk.Button(self.stats_tab, text="Refresh Statistics",
                  command=self.refresh_stats).pack(pady=5)

        # Stats display
        self.stats_text = scrolledtext.ScrolledText(self.stats_tab,
                                                    height=30,
                                                    width=120,
                                                    font=('Courier', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial stats load
        self.refresh_stats()

    def start_training(self, game_name, num_games):
        """Start training for a game"""

        if game_name in self.active_trainings:
            messagebox.showwarning("Already Running",
                                  f"{game_name} training is already running!")
            return

        config = self.GAMES[game_name]

        # Build command
        cmd = f"{config['trainer']} {num_games} --progress 10"

        # Update UI
        self.game_buttons[game_name]['start'].config(state=tk.DISABLED)
        self.game_buttons[game_name]['stop'].config(state=tk.NORMAL)

        # Log start
        self.log_output(f"\n{'='*80}\n", 'info')
        self.log_output(f"Starting {game_name} training - {num_games} games\n", 'game_name')
        self.log_output(f"{'='*80}\n", 'info')

        # Start training in background thread
        def run_training():
            try:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(__file__)
                )

                self.active_trainings[game_name] = process

                # Read output line by line
                for line in process.stdout:
                    self.update_queue.put(('output', game_name, line))

                # Wait for completion
                process.wait()

                # Remove from active
                if game_name in self.active_trainings:
                    del self.active_trainings[game_name]

                # Update UI
                self.update_queue.put(('complete', game_name, process.returncode))

            except Exception as e:
                self.update_queue.put(('error', game_name, str(e)))

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

    def stop_training(self, game_name):
        """Stop training for a game"""

        if game_name not in self.active_trainings:
            return

        process = self.active_trainings[game_name]
        process.terminate()

        self.log_output(f"\n[STOPPED] {game_name} training stopped by user\n", 'error')

        # Update UI
        self.game_buttons[game_name]['start'].config(state=tk.NORMAL)
        self.game_buttons[game_name]['stop'].config(state=tk.DISABLED)

        del self.active_trainings[game_name]

    def view_patterns(self, db_name):
        """Launch pattern viewer for a database"""

        db_path = os.path.join(os.path.dirname(__file__), db_name)

        if not os.path.exists(db_path):
            messagebox.showinfo("No Data",
                               "No training data exists yet. Train the game first!")
            return

        # Launch pattern viewer
        cmd = f"python3 pattern_viewer_gui.py {db_path}"
        subprocess.Popen(cmd, shell=True, cwd=os.path.dirname(__file__))

    def log_output(self, text, tag='info'):
        """Add output to monitor"""
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)

    def refresh_stats(self):
        """Refresh statistics display"""

        self.stats_text.delete('1.0', tk.END)

        self.stats_text.insert(tk.END, "="*100 + "\n")
        self.stats_text.insert(tk.END, "PATTERN RECOGNITION AI - TRAINING STATISTICS\n")
        self.stats_text.insert(tk.END, "="*100 + "\n\n")

        self.stats_text.insert(tk.END, f"{'Game':<25} {'Patterns':<12} {'Database Size':<15} {'Status'}\n")
        self.stats_text.insert(tk.END, "-"*100 + "\n")

        for game_name, config in self.GAMES.items():
            db_path = os.path.join(os.path.dirname(__file__), config['db'])

            if os.path.exists(db_path):
                try:
                    # Get pattern count
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM learned_move_patterns")
                    pattern_count = cursor.fetchone()[0]

                    # Get database size
                    size_bytes = os.path.getsize(db_path)
                    size_kb = size_bytes / 1024

                    conn.close()

                    status = "✓ Trained"
                    self.stats_text.insert(tk.END,
                        f"{game_name:<25} {pattern_count:<12} {size_kb:>10.1f} KB   {status}\n")

                except Exception as e:
                    self.stats_text.insert(tk.END,
                        f"{game_name:<25} {'ERROR':<12} {'N/A':<15} {str(e)[:30]}\n")
            else:
                self.stats_text.insert(tk.END,
                    f"{game_name:<25} {'0':<12} {'N/A':<15} Not trained yet\n")

        self.stats_text.insert(tk.END, "\n" + "="*100 + "\n")

    def update_ui(self):
        """Update UI from queue"""

        try:
            while True:
                msg_type, game_name, data = self.update_queue.get_nowait()

                if msg_type == 'output':
                    self.log_output(f"[{game_name}] {data}", 'info')

                elif msg_type == 'complete':
                    if data == 0:
                        self.log_output(f"\n[COMPLETE] {game_name} training finished successfully!\n\n", 'success')
                    else:
                        self.log_output(f"\n[ERROR] {game_name} training failed with code {data}\n\n", 'error')

                    self.game_buttons[game_name]['start'].config(state=tk.NORMAL)
                    self.game_buttons[game_name]['stop'].config(state=tk.DISABLED)

                elif msg_type == 'error':
                    self.log_output(f"\n[ERROR] {game_name}: {data}\n\n", 'error')
                    self.game_buttons[game_name]['start'].config(state=tk.NORMAL)
                    self.game_buttons[game_name]['stop'].config(state=tk.DISABLED)

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_ui)

    def on_closing(self):
        """Clean up when closing"""
        # Stop all active trainings
        for game_name in list(self.active_trainings.keys()):
            self.stop_training(game_name)

        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = GameLauncherGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
