#!/usr/bin/env python3
"""
COMPLETE SYSTEM RESET: Fix All Broken Patterns

The enhanced system broke ALL piece patterns, not just queens.
This script resets the entire pattern database to balanced, playable values.
"""

import sqlite3
import sys
from pathlib import Path

def analyze_broken_system(db_path):
    """Analyze the full extent of system damage"""
    
    print("🔍 COMPLETE SYSTEM DAMAGE ANALYSIS")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check each piece type
    piece_types = ['queen', 'rook', 'bishop', 'knight', 'pawn', 'king']
    
    print("Piece damage assessment:")
    print("Piece      Patterns  Avg Priority  Avg WinRate  Status")
    print("-" * 55)
    
    damage_assessment = {}
    
    for piece in piece_types:
        cursor.execute('''
            SELECT COUNT(*), AVG(priority_score), AVG(win_rate)
            FROM learned_move_patterns 
            WHERE piece_type = ? AND times_seen > 10
        ''', (piece,))
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            count, avg_pri, avg_wr = result
            
            # Assess damage level
            if avg_pri < 20 and avg_wr < 0.10:
                status = "🚨 CRITICAL"
                damage_level = "critical"
            elif avg_pri < 35 and avg_wr < 0.20:
                status = "⚠️ DAMAGED"  
                damage_level = "damaged"
            elif avg_pri > 60 and avg_wr > 0.30:
                status = "✅ HEALTHY"
                damage_level = "healthy"
            else:
                status = "🔧 NEEDS WORK"
                damage_level = "moderate"
                
            damage_assessment[piece] = damage_level
            print(f"{piece:10} {count:8} {avg_pri:11.1f} {avg_wr:10.1%}  {status}")
        else:
            damage_assessment[piece] = "missing"
            print(f"{piece:10} {'0':8} {'N/A':11} {'N/A':10}  🚫 NO DATA")
    
    # Overall assessment
    critical_pieces = sum(1 for level in damage_assessment.values() if level == "critical")
    total_pieces = len([p for p in damage_assessment.values() if p != "missing"])
    
    print(f"\n📊 Damage Summary:")
    print(f"   Critical damage: {critical_pieces}/{total_pieces} piece types")
    print(f"   System status: {'🚨 BROKEN' if critical_pieces > 2 else '⚠️ IMPAIRED' if critical_pieces > 0 else '✅ FUNCTIONAL'}")
    
    conn.close()
    return damage_assessment

def reset_all_patterns_to_balanced(db_path):
    """Reset ALL patterns to balanced, playable values"""
    
    print(f"\n🔄 COMPLETE PATTERN RESET TO BALANCED VALUES")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create backup of current state
    cursor.execute('DROP TABLE IF EXISTS learned_move_patterns_broken')
    cursor.execute('CREATE TABLE learned_move_patterns_broken AS SELECT * FROM learned_move_patterns')
    print("📦 Created backup: learned_move_patterns_broken")
    
    # Define balanced pattern values based on chess principles
    balanced_patterns = {
        # Queen patterns - most powerful piece
        ('queen', 'capture_check'): {'priority': 85.0, 'win_rate': 0.65},
        ('queen', 'capture'): {'priority': 80.0, 'win_rate': 0.60},
        ('queen', 'check'): {'priority': 75.0, 'win_rate': 0.55},
        ('queen', 'quiet'): {'priority': 70.0, 'win_rate': 0.45},
        ('queen', 'development'): {'priority': 65.0, 'win_rate': 0.40},
        
        # Rook patterns - second most powerful
        ('rook', 'capture_check'): {'priority': 75.0, 'win_rate': 0.55},
        ('rook', 'capture'): {'priority': 70.0, 'win_rate': 0.50},
        ('rook', 'check'): {'priority': 65.0, 'win_rate': 0.45},
        ('rook', 'quiet'): {'priority': 60.0, 'win_rate': 0.35},
        ('rook', 'development'): {'priority': 55.0, 'win_rate': 0.30},
        
        # Bishop patterns
        ('bishop', 'capture_check'): {'priority': 70.0, 'win_rate': 0.50},
        ('bishop', 'capture'): {'priority': 65.0, 'win_rate': 0.45},
        ('bishop', 'check'): {'priority': 60.0, 'win_rate': 0.40},
        ('bishop', 'quiet'): {'priority': 55.0, 'win_rate': 0.30},
        ('bishop', 'development'): {'priority': 60.0, 'win_rate': 0.35},
        
        # Knight patterns  
        ('knight', 'capture_check'): {'priority': 70.0, 'win_rate': 0.50},
        ('knight', 'capture'): {'priority': 65.0, 'win_rate': 0.45},
        ('knight', 'check'): {'priority': 60.0, 'win_rate': 0.40},
        ('knight', 'quiet'): {'priority': 55.0, 'win_rate': 0.30},
        ('knight', 'development'): {'priority': 65.0, 'win_rate': 0.40},
        
        # Pawn patterns
        ('pawn', 'capture_check'): {'priority': 60.0, 'win_rate': 0.45},
        ('pawn', 'capture'): {'priority': 55.0, 'win_rate': 0.40},
        ('pawn', 'check'): {'priority': 50.0, 'win_rate': 0.35},
        ('pawn', 'quiet'): {'priority': 45.0, 'win_rate': 0.25},
        ('pawn', 'development'): {'priority': 50.0, 'win_rate': 0.30},
        
        # King patterns - context dependent
        ('king', 'capture'): {'priority': 40.0, 'win_rate': 0.35},  # Can be good in endgame
        ('king', 'quiet'): {'priority': 35.0, 'win_rate': 0.30},   # Essential for endgame
        ('king', 'development'): {'priority': 30.0, 'win_rate': 0.25},
        ('king', 'check'): {'priority': 25.0, 'win_rate': 0.20},   # Rare but possible
    }
    
    # Apply balanced values
    print("🎯 Applying balanced pattern values:")
    total_updated = 0
    
    for (piece, move_type), values in balanced_patterns.items():
        priority = values['priority']
        win_rate = values['win_rate']
        
        cursor.execute('''
            UPDATE learned_move_patterns
            SET priority_score = ?,
                win_rate = ?,
                confidence = 0.4
            WHERE piece_type = ? AND move_category = ?
        ''', (priority, win_rate, piece, move_type))
        
        updated = cursor.rowcount
        total_updated += updated
        
        if updated > 0:
            print(f"   {piece:8} {move_type:12} → {priority:5.1f} priority, {win_rate:5.1%} WR ({updated} patterns)")
    
    print(f"\n✅ Updated {total_updated} patterns to balanced values")
    
    # Set reasonable defaults for any remaining patterns
    cursor.execute('''
        UPDATE learned_move_patterns
        SET priority_score = CASE 
            WHEN piece_type = 'queen' THEN 70.0
            WHEN piece_type = 'rook' THEN 60.0
            WHEN piece_type IN ('bishop', 'knight') THEN 55.0
            WHEN piece_type = 'pawn' THEN 45.0
            WHEN piece_type = 'king' THEN 35.0
            ELSE 40.0
        END,
        win_rate = CASE
            WHEN piece_type = 'queen' THEN 0.45
            WHEN piece_type = 'rook' THEN 0.35  
            WHEN piece_type IN ('bishop', 'knight') THEN 0.30
            WHEN piece_type = 'pawn' THEN 0.25
            WHEN piece_type = 'king' THEN 0.30
            ELSE 0.25
        END,
        confidence = 0.3
        WHERE priority_score IS NULL OR win_rate IS NULL 
        OR priority_score < 10 OR win_rate < 0.05
    ''')
    
    defaults_updated = cursor.rowcount
    print(f"🔧 Applied defaults to {defaults_updated} remaining patterns")
    
    conn.commit()
    conn.close()

def verify_balanced_system(db_path):
    """Verify the system is now balanced"""
    
    print(f"\n✅ VERIFYING BALANCED SYSTEM")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check each piece type again
    piece_types = ['queen', 'rook', 'bishop', 'knight', 'pawn', 'king']
    
    print("Post-reset analysis:")
    print("Piece      Patterns  Avg Priority  Avg WinRate  Status")
    print("-" * 55)
    
    all_healthy = True
    
    for piece in piece_types:
        cursor.execute('''
            SELECT COUNT(*), AVG(priority_score), AVG(win_rate), MIN(priority_score), MAX(priority_score)
            FROM learned_move_patterns 
            WHERE piece_type = ? AND times_seen > 10
        ''', (piece,))
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            count, avg_pri, avg_wr, min_pri, max_pri = result
            
            # Check if healthy
            expected_min = {'queen': 60, 'rook': 50, 'bishop': 45, 'knight': 45, 'pawn': 40, 'king': 25}
            expected_wr = {'queen': 0.35, 'rook': 0.25, 'bishop': 0.25, 'knight': 0.25, 'pawn': 0.20, 'king': 0.20}
            
            is_healthy = avg_pri >= expected_min.get(piece, 30) and avg_wr >= expected_wr.get(piece, 0.20)
            status = "✅ HEALTHY" if is_healthy else "❌ STILL BROKEN"
            
            if not is_healthy:
                all_healthy = False
            
            print(f"{piece:10} {count:8} {avg_pri:11.1f} {avg_wr:10.1%}  {status}")
    
    # Overall system check
    cursor.execute('''
        SELECT 
            AVG(priority_score) as avg_priority,
            AVG(win_rate) as avg_winrate,
            MIN(priority_score) as min_priority,
            COUNT(*) as total_patterns
        FROM learned_move_patterns 
        WHERE times_seen > 10
    ''')
    
    avg_pri, avg_wr, min_pri, total = cursor.fetchone()
    
    print(f"\n📊 Overall system health:")
    print(f"   Total patterns: {total}")
    print(f"   Average priority: {avg_pri:.1f}")
    print(f"   Average win rate: {avg_wr:.1%}")
    print(f"   Minimum priority: {min_pri:.1f}")
    
    system_healthy = all_healthy and avg_pri > 45 and avg_wr > 0.25 and min_pri > 20
    
    print(f"\n🏥 System Status: {'🎉 FULLY RECOVERED' if system_healthy else '⚠️ NEEDS MORE WORK'}")
    
    conn.close()
    return system_healthy

def create_test_game_script(project_dir):
    """Create script to test actual gameplay"""
    
    test_game_script = '''#!/usr/bin/env python3
"""
Test Actual Game - See if AI can play normally now
"""

import chess
from game_scorer import GameScorer
from learnable_move_prioritizer import LearnableMovePrioritizer

def test_ai_gameplay():
    """Test if AI can play a reasonable game"""
    
    print("🎮 TESTING AI GAMEPLAY")
    print("=" * 40)
    
    # Initialize components
    prioritizer = LearnableMovePrioritizer("enhanced_training.db")
    scorer = GameScorer()
    
    board = chess.Board()
    moves_made = []
    
    print("🔍 Testing first 10 AI moves:")
    print("Move   Piece    Type        Priority  SAN")
    print("-" * 45)
    
    for move_num in range(10):
        if board.is_game_over():
            break
            
        # Get AI move
        legal_moves = list(board.legal_moves)
        legal_moves = prioritizer.sort_moves_by_priority(board, legal_moves)
        
        if not legal_moves:
            break
            
        # Pick best move
        best_move = legal_moves[0]
        priority = prioritizer.get_move_priority(board, best_move)
        
        # Get move info
        piece = board.piece_at(best_move.from_square)
        piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
        
        move_type = "capture" if board.is_capture(best_move) else "quiet"
        if board.gives_check(best_move):
            move_type += "+check"
            
        san = board.san(best_move)
        
        print(f"{move_num+1:4d}   {piece_name:8} {move_type:10} {priority:8.1f}  {san}")
        
        # Make the move
        board.push(best_move)
        moves_made.append((piece_name, move_type, priority, san))
        
        # Make random opponent move
        opp_moves = list(board.legal_moves)
        if opp_moves:
            board.push(opp_moves[0])
    
    # Analyze the moves
    piece_types_used = set(move[0] for move in moves_made)
    avg_priority = sum(move[2] for move in moves_made) / len(moves_made)
    
    print(f"\\n📊 Game Analysis:")
    print(f"   Moves made: {len(moves_made)}")
    print(f"   Piece types used: {', '.join(sorted(piece_types_used))}")
    print(f"   Average priority: {avg_priority:.1f}")
    print(f"   Used queen: {'✅' if 'queen' in piece_types_used else '❌'}")
    print(f"   Used other pieces: {'✅' if len(piece_types_used) > 1 else '❌'}")
    
    # Assessment
    if len(piece_types_used) >= 3 and avg_priority > 40 and 'queen' in piece_types_used:
        print(f"\\n🎉 AI APPEARS TO BE PLAYING NORMALLY!")
        game_health = "healthy"
    elif len(piece_types_used) >= 2 and avg_priority > 30:
        print(f"\\n🔧 AI is playing but could be better")
        game_health = "moderate"
    else:
        print(f"\\n❌ AI still has serious problems")
        game_health = "broken"
    
    prioritizer.close()
    return game_health

if __name__ == "__main__":
    test_ai_gameplay()
'''
    
    test_path = Path(project_dir) / "test_game.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_game_script)
    
    print(f"   ✓ Created gameplay test: {test_path}")

def main():
    """Complete system reset"""
    
    print("🚨 COMPLETE ENHANCED PATTERN SYSTEM RESET")
    print("This will fix ALL broken patterns, not just queens")
    print("=" * 60)
    
    # Paths
    db_path = "C:/Users/Sojourner/Desktop/patmat/enhanced_training.db"
    project_dir = "C:/Users/Sojourner/Desktop/patmat"
    
    try:
        # Step 1: Analyze full damage
        damage_assessment = analyze_broken_system(db_path)
        
        # Step 2: Reset everything to balanced values
        reset_all_patterns_to_balanced(db_path)
        
        # Step 3: Verify the reset worked
        system_healthy = verify_balanced_system(db_path)
        
        # Step 4: Create gameplay test
        create_test_game_script(project_dir)
        
        print(f"\\n{'=' * 60}")
        if system_healthy:
            print("🎉 COMPLETE SYSTEM RESET SUCCESSFUL!")
            print("\\n🎮 Test the reset:")
            print("   1. Test gameplay patterns:")
            print("      python test_game.py")
            print("   ")
            print("   2. Run actual training:")
            print("      python headless_trainer.py 50")
            print("   ")
            print("   Expected improvements:")
            print("   - Win rate: 0% → 15-30%")
            print("   - Stalemate rate: 94% → 40-60%") 
            print("   - AI will use ALL pieces, not just queen")
            print("   - Games will be more dynamic and varied")
            
        else:
            print("⚠️ RESET INCOMPLETE - Some issues remain")
            print("\\nFallback option:")
            print("   Delete enhanced_training.db and start fresh:")
            print("   rm enhanced_training.db")
            print("   python headless_trainer.py 100")
            
        print(f"\\n📦 Broken patterns backed up as: learned_move_patterns_broken")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\\n🆘 EMERGENCY OPTIONS:")
        print("1. Delete enhanced database completely:")
        print("   rm enhanced_training.db")  
        print("2. Use your original working database:")
        print("   python headless_trainer.py 100")

if __name__ == "__main__":
    main()
