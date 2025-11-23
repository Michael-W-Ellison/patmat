#!/usr/bin/env python3
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
    
    print(f"\n📊 Game Analysis:")
    print(f"   Moves made: {len(moves_made)}")
    print(f"   Piece types used: {', '.join(sorted(piece_types_used))}")
    print(f"   Average priority: {avg_priority:.1f}")
    print(f"   Used queen: {'✅' if 'queen' in piece_types_used else '❌'}")
    print(f"   Used other pieces: {'✅' if len(piece_types_used) > 1 else '❌'}")
    
    # Assessment
    if len(piece_types_used) >= 3 and avg_priority > 40 and 'queen' in piece_types_used:
        print(f"\n🎉 AI APPEARS TO BE PLAYING NORMALLY!")
        game_health = "healthy"
    elif len(piece_types_used) >= 2 and avg_priority > 30:
        print(f"\n🔧 AI is playing but could be better")
        game_health = "moderate"
    else:
        print(f"\n❌ AI still has serious problems")
        game_health = "broken"
    
    prioritizer.close()
    return game_health

if __name__ == "__main__":
    test_ai_gameplay()
