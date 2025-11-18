#!/usr/bin/env python3
"""
Pattern Database Viewer - GUI for examining learned patterns

Shows:
- All learned patterns with stats
- Individual pattern performance history
- Trend analysis (improving/declining)
- Filter and sort options
"""

import tkinter as tk
from tkinter import ttk
import sqlite3


class PatternViewerGUI:
    """GUI for viewing pattern database details"""

    def __init__(self, root, db_path):
        self.root = root
        self.db_path = db_path
        self.root.title("Pattern Database Viewer")
        self.root.geometry("1000x700")

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.create_widgets()
        self.refresh_patterns()

    def create_widgets(self):
        """Create GUI widgets"""

        # Top frame - filters and controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text="Filter:").pack(side=tk.LEFT, padx=5)

        # Piece filter
        ttk.Label(control_frame, text="Piece:").pack(side=tk.LEFT)
        self.piece_filter = ttk.Combobox(control_frame, width=10,
                                         values=['All', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'])
        self.piece_filter.set('All')
        self.piece_filter.pack(side=tk.LEFT, padx=5)

        # Category filter
        ttk.Label(control_frame, text="Category:").pack(side=tk.LEFT, padx=(20, 0))
        self.category_filter = ttk.Combobox(control_frame, width=12,
                                            values=['All', 'capture', 'development', 'quiet', 'check'])
        self.category_filter.set('All')
        self.category_filter.pack(side=tk.LEFT, padx=5)

        # Sort option
        ttk.Label(control_frame, text="Sort:").pack(side=tk.LEFT, padx=(20, 0))
        self.sort_option = ttk.Combobox(control_frame, width=12,
                                        values=['Priority', 'Win Rate', 'Games Played', 'Recent'])
        self.sort_option.set('Priority')
        self.sort_option.pack(side=tk.LEFT, padx=5)

        # Refresh button
        ttk.Button(control_frame, text="🔄 Refresh", command=self.refresh_patterns).pack(side=tk.LEFT, padx=20)

        # Stats frame
        stats_frame = ttk.LabelFrame(self.root, text="Database Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="Loading...")
        self.stats_label.pack()

        # Pattern list frame
        list_frame = ttk.LabelFrame(self.root, text="Learned Patterns", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Treeview for patterns
        columns = ('piece', 'category', 'phase', 'games', 'wins', 'losses', 'win%', 'avg_score', 'priority')
        self.pattern_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=20)

        # Column headings
        self.pattern_tree.heading('piece', text='Piece')
        self.pattern_tree.heading('category', text='Category')
        self.pattern_tree.heading('phase', text='Phase')
        self.pattern_tree.heading('games', text='Games')
        self.pattern_tree.heading('wins', text='Wins')
        self.pattern_tree.heading('losses', text='Losses')
        self.pattern_tree.heading('win%', text='Win %')
        self.pattern_tree.heading('avg_score', text='Avg Score')
        self.pattern_tree.heading('priority', text='Priority')

        # Column widths
        self.pattern_tree.column('piece', width=80)
        self.pattern_tree.column('category', width=120)
        self.pattern_tree.column('phase', width=100)
        self.pattern_tree.column('games', width=60)
        self.pattern_tree.column('wins', width=60)
        self.pattern_tree.column('losses', width=60)
        self.pattern_tree.column('win%', width=70)
        self.pattern_tree.column('avg_score', width=90)
        self.pattern_tree.column('priority', width=80)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pattern_tree.yview)
        self.pattern_tree.configure(yscrollcommand=scrollbar.set)

        self.pattern_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind double-click to show details
        self.pattern_tree.bind('<Double-1>', self.show_pattern_details)

        # Bottom frame - details
        details_frame = ttk.LabelFrame(self.root, text="Pattern Details (double-click pattern above)", padding=10)
        details_frame.pack(fill=tk.X, padx=10, pady=5)

        self.details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def refresh_patterns(self):
        """Refresh pattern list from database"""
        # Clear existing items
        for item in self.pattern_tree.get_children():
            self.pattern_tree.delete(item)

        # Build query
        query = '''
            SELECT piece_type, move_category, game_phase,
                   times_seen, games_won, games_lost, games_drawn,
                   win_rate, avg_score, priority_score
            FROM learned_move_patterns
            WHERE 1=1
        '''

        params = []

        # Apply filters
        if self.piece_filter.get() != 'All':
            query += ' AND piece_type = ?'
            params.append(self.piece_filter.get())

        if self.category_filter.get() != 'All':
            query += ' AND move_category = ?'
            params.append(self.category_filter.get())

        # Apply sorting
        sort_by = self.sort_option.get()
        if sort_by == 'Priority':
            query += ' ORDER BY priority_score DESC'
        elif sort_by == 'Win Rate':
            query += ' ORDER BY win_rate DESC'
        elif sort_by == 'Games Played':
            query += ' ORDER BY times_seen DESC'
        elif sort_by == 'Recent':
            query += ' ORDER BY updated_at DESC'

        # Execute query
        self.cursor.execute(query, params)
        patterns = self.cursor.fetchall()

        # Populate tree
        for pattern in patterns:
            piece, category, phase, times, wins, losses, draws, win_rate, avg_score, priority = pattern

            total_games = wins + losses + draws
            win_pct = f"{win_rate*100:.1f}%"

            self.pattern_tree.insert('', tk.END, values=(
                piece, category, phase, total_games, wins, losses,
                win_pct, f"{avg_score:+.0f}", f"{priority:.1f}"
            ))

        # Update statistics
        self.update_statistics()

    def update_statistics(self):
        """Update database statistics"""
        # Total patterns
        self.cursor.execute('SELECT COUNT(*) FROM learned_move_patterns')
        total_patterns = self.cursor.fetchone()[0]

        # Average priority
        self.cursor.execute('SELECT AVG(priority_score) FROM learned_move_patterns')
        avg_priority = self.cursor.fetchone()[0] or 0

        # Average confidence
        self.cursor.execute('SELECT AVG(confidence) FROM learned_move_patterns')
        avg_confidence = self.cursor.fetchone()[0] or 0

        # Total games recorded
        self.cursor.execute('SELECT SUM(times_seen) FROM learned_move_patterns')
        total_observations = self.cursor.fetchone()[0] or 0

        # Overall win rate
        self.cursor.execute('SELECT SUM(games_won), SUM(games_lost), SUM(games_drawn) FROM learned_move_patterns')
        wins, losses, draws = self.cursor.fetchone()
        total = wins + losses + draws
        overall_wr = (wins / total * 100) if total > 0 else 0

        stats_text = (f"Total Patterns: {total_patterns}  |  "
                     f"Total Observations: {total_observations}  |  "
                     f"Avg Priority: {avg_priority:.1f}  |  "
                     f"Avg Confidence: {avg_confidence:.2f}  |  "
                     f"Overall Win Rate: {overall_wr:.1f}%")

        self.stats_label.configure(text=stats_text)

    def show_pattern_details(self, event):
        """Show detailed information about selected pattern"""
        selection = self.pattern_tree.selection()
        if not selection:
            return

        item = self.pattern_tree.item(selection[0])
        values = item['values']

        piece, category, phase = values[0], values[1], values[2]
        total_games, wins, losses = values[3], values[4], values[5]
        win_pct, avg_score, priority = values[6], values[7], values[8]

        # Get additional details
        self.cursor.execute('''
            SELECT times_seen, games_won, games_lost, games_drawn,
                   total_score, confidence, updated_at
            FROM learned_move_patterns
            WHERE piece_type = ? AND move_category = ? AND game_phase = ?
        ''', (piece, category, phase))

        result = self.cursor.fetchone()
        if not result:
            return

        times, w, l, d, total_score, confidence, updated = result

        details = f"""
Pattern: {piece.upper()} - {category.title()} ({phase})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance:
  Games Played: {total_games}  |  W-L-D: {wins}-{losses}-{d}  |  Win Rate: {win_pct}
  Average Score: {avg_score} (differential)
  Total Score: {total_score:.0f}

Learning Stats:
  Priority: {priority} / 100
  Confidence: {confidence:.2f} / 1.00  (increases with observations)
  Last Updated: {updated}

Interpretation:
  Priority {priority:.0f}: {"HIGH - Frequently selected" if float(priority) > 60 else "MEDIUM - Sometimes selected" if float(priority) > 40 else "LOW - Rarely selected"}
  Score {avg_score}: {"Excellent pattern" if float(avg_score) > 500 else "Good pattern" if float(avg_score) > 0 else "Losing pattern"}
  Confidence {confidence:.2f}: {"High confidence" if confidence > 0.7 else "Medium confidence" if confidence > 0.3 else "Low confidence (needs more games)"}
        """

        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details.strip())

    def close(self):
        """Clean up"""
        self.conn.close()
        self.root.destroy()


def main():
    """Run pattern viewer"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 pattern_viewer_gui.py <database_path>")
        print("\nExample:")
        print("  python3 pattern_viewer_gui.py progressive_training.db")
        sys.exit(1)

    db_path = sys.argv[1]

    root = tk.Tk()
    app = PatternViewerGUI(root, db_path)

    def on_closing():
        app.close()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
