"""
Data Loader utilities for chessboard classification.
Handles FEN parsing and label extraction from input data.
"""

import numpy as np
from typing import List, Union

# 13 classes total:
# 0-5: White pieces, 6-11: Black pieces, 12: Empty square
PIECE_TO_ID = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    'empty': 12
}

# Inverse mapping: ID -> piece character (for FEN reconstruction)
ID_TO_PIECE = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k',
    12: None  # Empty square
}

NUM_CLASSES = 13


def parse_fen_to_grid(fen: str) -> List[List[str]]:
    """
    Converts a FEN board string into an 8x8 grid of piece identifiers.
    Only processes the first field (piece placement).
    """
    fen_board = fen.split(' ')[0].strip()
    ranks = fen_board.split('/')

    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen_board}")

    valid_pieces = set('KQRBNPkqrbnp')
    grid: List[List[str]] = []

    for rank_idx, rank in enumerate(ranks):
        grid_row: List[str] = []
        i = 0
        while i < len(rank):
            ch = rank[i]
            if ch.isdigit():
                num_str = ""
                while i < len(rank) and rank[i].isdigit():
                    num_str += rank[i]
                    i += 1
                num_empty = int(num_str)
                if not (1 <= num_empty <= 8):
                    raise ValueError(f"Invalid empty count {num_empty} in rank {8-rank_idx}: {rank}")
                grid_row.extend(['empty'] * num_empty)
                continue
            elif ch in valid_pieces:
                grid_row.append(ch)
            else:
                raise ValueError(f"Invalid char '{ch}' in rank {8-rank_idx}: {rank}")
            i += 1

        if len(grid_row) != 8:
            raise ValueError(f"Rank {8-rank_idx} must have 8 squares, got {len(grid_row)}: {rank}")

        grid.append(grid_row)

    return grid


def fen_to_labels(fen: str, flatten: bool = False) -> np.ndarray:
    """
    Converts FEN string to numpy labels (8x8) or (64,)
    """
    grid = parse_fen_to_grid(fen)
    labels = np.array([[PIECE_TO_ID[p] for p in row] for row in grid], dtype=np.int64)
    return labels.flatten() if flatten else labels


def grid_to_fen(grid: np.ndarray) -> str:
    """
    Converts an 8x8 grid of class IDs (or 'unknown') back to a FEN string.
    
    Args:
        grid: 8x8 numpy array where each cell is:
              - An integer (0-12) representing a piece class
              - The string 'unknown' for occluded squares
    
    Returns:
        FEN string (board placement only, no castling/en-passant info)
        
    Note: 'unknown' and empty (12) are both treated as empty squares in FEN.
    """
    fen_ranks = []
    
    for row in grid:
        rank_str = ""
        empty_count = 0
        
        for cell in row:
            # Handle 'unknown' string or empty class (12)
            if cell == 'unknown' or cell == 12:
                empty_count += 1
            else:
                # It's a piece (0-11)
                piece_char = ID_TO_PIECE.get(int(cell))
                if piece_char is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += piece_char
        
        # Flush remaining empty squares
        if empty_count > 0:
            rank_str += str(empty_count)
        
        fen_ranks.append(rank_str)
    
    return "/".join(fen_ranks)
