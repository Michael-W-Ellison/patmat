import sqlite3

conn = sqlite3.connect('headless_training.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*), SUM(times_seen), SUM(games_won), SUM(games_lost), SUM(games_drawn) FROM learned_move_patterns')
count, seen, won, lost, drawn = cursor.fetchone()
total = won + lost + drawn

print(f'Patterns: {count:,}')
print(f'Observations: {seen:,}')
print(f'Win rate: {won/total*100:.1f}% ({won:,}W-{lost:,}L-{drawn:,}D)')
print()

cursor.execute('SELECT piece_type, move_category, game_phase, times_seen, win_rate, priority_score FROM learned_move_patterns ORDER BY priority_score DESC LIMIT 5')
print("Top 5 patterns:")
for row in cursor.fetchall():
    print(f'  {row[0]:8} {row[1]:12} {row[2]:10} seen={row[3]:>10,} wr={row[4]:>5.1%} pri={row[5]:>5.1f}')

conn.close()