"""Enhanced chess feature engineering module.

Calculates advanced chess features by replaying move sequences from SAN/UCI strings:
- Pawn structures (doubled, isolated, passed, backward pawns, pawn islands)
- Piece-Square Tables (PST) opening positional scores
- King safety (pawn shield, enemy attackers, open files near king)
- Piece mobility (attack coverage for minor and major pieces)
- Piece development (minor pieces moved, back rank counts)
"""

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm

# Piece Square Tables (PST) - Opening/Midgame values
# Positive for white, perspective of white player. Mirror for black.
# High values encourage center occupancy, development, king safety.
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

PST_ROOK = [
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MIDDLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PST_MAP = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MIDDLE
}

def get_pst_score(board: chess.Board, color: chess.Color) -> float:
    """Calculate the Piece-Square Table score for a color."""
    score = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            pst_table = PST_MAP[piece.piece_type]
            # White uses index directly, Black mirrors vertically
            idx = square if color == chess.WHITE else chess.square_mirror(square)
            score += pst_table[idx]
    return score

def get_pawn_structure_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Calculate pawn structure statistics for a color."""
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    # Files of our pawns
    pawn_files = [chess.square_file(sq) for sq in pawns]
    pawn_file_counts = {f: pawn_files.count(f) for f in range(8)}
    
    doubled = sum(count - 1 for count in pawn_file_counts.values() if count > 1)
    
    isolated = 0
    passed = 0
    backward = 0
    
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # Isolated: no friendly pawns on adjacent files
        adj_files = [f - 1, f + 1]
        has_adj_pawn = any(pawn_file_counts.get(af, 0) > 0 for af in adj_files if 0 <= af < 8)
        if not has_adj_pawn:
            isolated += 1
            
        # Passed: no enemy pawns in front of it on same or adjacent files
        is_passed = True
        files_to_check = [f - 1, f, f + 1]
        for ef in files_to_check:
            if 0 <= ef < 8:
                for er in range(8):
                    # Check ranks in front of pawn
                    in_front = er > r if color == chess.WHITE else er < r
                    if in_front and board.piece_at(chess.square(ef, er)) == chess.Piece(chess.PAWN, not color):
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            passed += 1
            
        # Backward pawn: behind friendly pawns on adjacent files, and cannot be defended by friendly pawns
        # Simple heuristic: no friendly pawn on adjacent files at the same or lower rank
        is_backward = True
        has_adj_files = False
        for af in adj_files:
            if 0 <= af < 8:
                has_adj_files = True
                # Find if any friendly pawns on adjacent file af are at rank <= r (for white) or >= r (for black)
                for ar in range(8):
                    if board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, color):
                        behind_or_beside = ar <= r if color == chess.WHITE else ar >= r
                        if behind_or_beside:
                            is_backward = False
                            break
            if not is_backward:
                break
        if is_backward and has_adj_files and pawn_file_counts[f] > 0:
            backward += 1

    # Pawn Islands count
    present_files = [int(pawn_file_counts[f] > 0) for f in range(8)]
    pawn_islands = 0
    in_island = False
    for f in range(8):
        if present_files[f]:
            if not in_island:
                pawn_islands += 1
                in_island = True
        else:
            in_island = False
            
    return {
        "doubled": doubled,
        "isolated": isolated,
        "passed": passed,
        "backward": backward,
        "pawn_islands": pawn_islands
    }

def get_king_safety_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Calculate king safety metrics."""
    king_square = board.king(color)
    if king_square is None:
        return {"pawn_shield": 0, "attackers_near_king": 0, "open_files_near_king": 0}
        
    kf = chess.square_file(king_square)
    kr = chess.square_rank(king_square)
    
    # Pawn shield
    # Look at 3 files in front of king
    shield_rank = kr + 1 if color == chess.WHITE else kr - 1
    pawn_shield = 0
    if 0 <= shield_rank < 8:
        for f in [kf - 1, kf, kf + 1]:
            if 0 <= f < 8:
                piece = board.piece_at(chess.square(f, shield_rank))
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shield += 1
                    
    # Attackers near king
    # Count enemy pieces attacking any adjacent square to king
    adj_squares = []
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf, nr = kf + df, kr + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                adj_squares.append(chess.square(nf, nr))
                
    enemy_attackers = set()
    for sq in adj_squares:
        attackers = board.attackers(not color, sq)
        for attacker_sq in attackers:
            enemy_attackers.add(attacker_sq)
    attackers_near_king = len(enemy_attackers)
    
    # Open files near king
    # Check files kf-1, kf, kf+1. If no pawns of our color are on a file, it's open.
    open_files = 0
    for f in [kf - 1, kf, kf + 1]:
        if 0 <= f < 8:
            has_pawn = False
            for r in range(8):
                piece = board.piece_at(chess.square(f, r))
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
                
    return {
        "pawn_shield": pawn_shield,
        "attackers_near_king": attackers_near_king,
        "open_files_near_king": open_files
    }

def get_mobility_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Calculate piece mobility metrics (attack coverage)."""
    # Count how many squares are attacked by different piece types
    knight_attacks = set()
    bishop_attacks = set()
    rook_attacks = set()
    queen_attacks = set()
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            attacks = board.attacks(sq)
            if piece.piece_type == chess.KNIGHT:
                knight_attacks.update(attacks)
            elif piece.piece_type == chess.BISHOP:
                bishop_attacks.update(attacks)
            elif piece.piece_type == chess.ROOK:
                rook_attacks.update(attacks)
            elif piece.piece_type == chess.QUEEN:
                queen_attacks.update(attacks)
                
    return {
        "knight_mobility": len(knight_attacks),
        "bishop_mobility": len(bishop_attacks),
        "rook_mobility": len(rook_attacks),
        "queen_mobility": len(queen_attacks),
        "total_attack_coverage": len(knight_attacks | bishop_attacks | rook_attacks | queen_attacks)
    }

def get_development_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Calculate piece development metrics."""
    # Starting squares for knights/bishops
    starting_squares = [chess.B1, chess.C1, chess.F1, chess.G1] if color == chess.WHITE else [chess.B8, chess.C8, chess.F8, chess.G8]
    minor_developed = 0
    for sq in starting_squares:
        piece = board.piece_at(sq)
        # If square is empty or has another piece, it's developed
        if piece is None or piece.color != color or piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
            minor_developed += 1
            
    # Back rank count
    back_rank = 0 if color == chess.WHITE else 7
    back_rank_count = 0
    for f in range(8):
        piece = board.piece_at(chess.square(f, back_rank))
        if piece and piece.color == color:
            back_rank_count += 1
            
    return {
        "minor_developed": minor_developed,
        "back_rank_count": back_rank_count
    }

def extract_all_enhanced(board: chess.Board, color: chess.Color, prefix: str) -> dict[str, float | int]:
    """Extract all enhanced features for a given board state and color."""
    features = {}
    
    # PST Score
    pst_val = get_pst_score(board, color)
    features[f"{prefix}pst_score"] = pst_val
    
    # Pawn Structure
    pawns = get_pawn_structure_features(board, color)
    for k, v in pawns.items():
        features[f"{prefix}pawns_{k}"] = v
        
    # King Safety
    king = get_king_safety_features(board, color)
    for k, v in king.items():
        features[f"{prefix}king_{k}"] = v
        
    # Mobility
    mob = get_mobility_features(board, color)
    for k, v in mob.items():
        features[f"{prefix}{k}"] = v
        
    # Development
    dev = get_development_features(board, color)
    for k, v in dev.items():
        features[f"{prefix}dev_{k}"] = v
        
    return features

def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features to the DataFrame by replaying games."""
    print("Replaying games to engineer enhanced features...")
    df = df.copy()
    
    # We will collect list of dictionaries of enhanced features
    enhanced_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Reconstruct the board
        moves_text = str(row["first_10_moves_text"])
        tokens = moves_text.split()
        
        # UCI moves are at odd indices (1, 3, 5, ...)
        uci_moves = tokens[1::2]
        
        board = chess.Board()
        board_m3 = None
        board_m10 = None
        
        for i, uci in enumerate(uci_moves):
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    break
            except Exception:
                break
                
            if i == 5:  # After 6 plies (3 full moves)
                board_m3 = board.copy(stack=False)
            if i == 19:  # After 20 plies (10 full moves)
                board_m10 = board.copy(stack=False)
                break
        
        # If boards are missing (should not happen if eligible, but fallback)
        if board_m3 is None:
            board_m3 = board.copy(stack=False)
        if board_m10 is None:
            board_m10 = board.copy(stack=False)
            
        # Extract features for white and black
        feats = {}
        
        # Move 3 features
        w3 = extract_all_enhanced(board_m3, chess.WHITE, "m3_white_")
        b3 = extract_all_enhanced(board_m3, chess.BLACK, "m3_black_")
        feats.update(w3)
        feats.update(b3)
        # PST diff
        feats["m3_pst_diff"] = w3["m3_white_pst_score"] - b3["m3_black_pst_score"]
        feats["m3_mobility_diff"] = w3["m3_white_total_attack_coverage"] - b3["m3_black_total_attack_coverage"]
        
        # Move 10 features
        w10 = extract_all_enhanced(board_m10, chess.WHITE, "m10_white_")
        b10 = extract_all_enhanced(board_m10, chess.BLACK, "m10_black_")
        feats.update(w10)
        feats.update(b10)
        # PST diff
        feats["m10_pst_diff"] = w10["m10_white_pst_score"] - b10["m10_black_pst_score"]
        feats["m10_mobility_diff"] = w10["m10_white_total_attack_coverage"] - b10["m10_black_total_attack_coverage"]
        
        enhanced_records.append(feats)
        
    # Create DataFrame and concat
    enhanced_df = pd.DataFrame(enhanced_records, index=df.index)
    res_df = pd.concat([df, enhanced_df], axis=1)
    
    # Also add Elo baseline expected probability
    # Expected score using Elo formula: 1 / (1 + 10 ** (elo_diff / 400))
    # Note: elo_diff is white_elo - black_elo. But wait, in expected score:
    # E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400)) = 1 / (1 + 10 ** (-elo_diff / 400))
    if "elo_diff" in res_df.columns:
        res_df["elo_expected_prob"] = 1.0 / (1.0 + 10.0 ** (-res_df["elo_diff"] / 400.0))
        
    return res_df
