"""
Data Loader utilities for chessboard classification.
Handles FEN parsing and label extraction from input data.
"""

import numpy as np
from typing import List, Union


# Mapping pieces to class IDs (13 classes total)
# 0-5: White pieces, 6-11: Black pieces, 12: Empty square
PIECE_TO_ID = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,     # White
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,   # Black
    'empty': 12
}

# Pre-calculated lookup table for optimized ID mapping
# This avoids dictionary lookups in nested loops for better performance
_PIECE_TO_ID_LOOKUP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    'empty': 12
}
# For even faster access, we can use a list-based lookup if needed


def parse_fen_to_grid(fen: str) -> List[List[str]]:
    """
    Converts a FEN board string into an 8x8 grid of piece identifiers.
    
    FEN (Forsyth-Edwards Notation) piece placement rules:
    - Ranks are separated by '/' and listed from rank 8 (top) to rank 1 (bottom)
    - Each rank describes files a-h (left to right)
    - Uppercase letters = white pieces: K, Q, R, B, N, P
    - Lowercase letters = black pieces: k, q, r, b, n, p
    - Numbers represent consecutive empty squares (e.g., "8" = 8 empty squares)
    - Each rank must total exactly 8 squares
    
    Full FEN format has 6 fields separated by spaces:
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    This function only processes the first field (piece placement).
    
    Args:
        fen: FEN string (can be full FEN or just piece placement part)
        
    Returns:
        8x8 grid (list of lists) where:
        - grid[0] is rank 8 (top, black's back rank)
        - grid[7] is rank 1 (bottom, white's back rank)
        - Each element is either a piece character ('P', 'r', etc.) or 'empty'
        
    Raises:
        ValueError: If FEN format is invalid (wrong number of ranks, 
                   invalid characters, or rank doesn't sum to 8 squares)
    
    Example:
        >>> parse_fen_to_grid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],  # rank 8
         ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  # rank 7
         ['empty', 'empty', ...],                    # rank 6
         ...]
    """
    # Extract only the piece placement field (first field before space)
    # Full FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # We need: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    fen_board = fen.split(' ')[0].strip()
    
    # Split into 8 ranks (rank 8 to rank 1, separated by '/')
    ranks = fen_board.split('/')
    
    if len(ranks) != 8:
        raise ValueError(
            f"FEN must have exactly 8 ranks separated by '/', got {len(ranks)}. "
            f"FEN: {fen_board}"
        )
    
    grid = []
    valid_pieces = set('KQRBNPkqrbnp')
    
    for rank_idx, rank in enumerate(ranks):
        grid_row = []
        i = 0
        
        # Process each character in the rank string
        while i < len(rank):
            char = rank[i]
            
            if char.isdigit():
                # Handle numbers (standard FEN uses single digits 1-8, but we support multi-digit)
                # Collect all consecutive digits to form the number
                num_str = ""
                while i < len(rank) and rank[i].isdigit():
                    num_str += rank[i]
                    i += 1
                
                num_empty = int(num_str)
                if num_empty == 0:
                    raise ValueError(f"FEN cannot have 0 empty squares. Rank {8-rank_idx}: {rank}")
                if num_empty > 8:
                    raise ValueError(
                        f"FEN cannot have more than 8 empty squares in a rank. "
                        f"Got {num_empty} in rank {8-rank_idx}: {rank}"
                    )
                
                grid_row.extend(['empty'] * num_empty)
                # Don't increment i here - we already advanced it in the while loop
                continue
                
            elif char in valid_pieces:
                # Valid piece character
                grid_row.append(char)
            else:
                # Invalid character
                raise ValueError(
                    f"Invalid character '{char}' in FEN rank {8-rank_idx}. "
                    f"Rank: {rank}. Valid characters: KQRBNPkqrbnp or digits 1-8"
                )
            
            i += 1
        
        # Validate that the rank has exactly 8 squares
        if len(grid_row) != 8:
            raise ValueError(
                f"FEN rank {8-rank_idx} must have exactly 8 squares, got {len(grid_row)}. "
                f"Rank string: '{rank}'"
            )
        
        grid.append(grid_row)
    
    return grid


def fen_to_labels(fen: str, flatten: bool = False) -> np.ndarray:
    """
    Converts FEN string directly to a numpy array of class IDs.
    Optimized for performance using pre-calculated lookup table.
    
    Args:
        fen: FEN board position string
        flatten: If True, returns flattened array of shape (64,).
                 If False, returns 2D array of shape (8, 8).
                 Default: False (2D grid format for per-square classification)
    
    Returns:
        numpy array with class IDs (0-12):
        - Shape (8, 8) if flatten=False (default) - suitable for per-square CrossEntropyLoss
        - Shape (64,) if flatten=True - suitable for whole-board classification
    
    Example:
        >>> labels = fen_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        >>> labels.shape
        (8, 8)
        >>> labels_flat = fen_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", flatten=True)
        >>> labels_flat.shape
        (64,)
    """
    grid = parse_fen_to_grid(fen)
    
    # Optimized: Use pre-calculated lookup table instead of dictionary access in nested loop
    # This is faster when processing millions of FEN strings
    labels = np.array([[_PIECE_TO_ID_LOOKUP[piece] for piece in row] for row in grid], dtype=np.int64)
    
    if flatten:
        return labels.flatten()
    return labels


# Test functions for FEN parser
if __name__ == "__main__":
    # Test with standard starting position
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print("Testing FEN parser (input only)...")
    print(f"Input FEN: {test_fen}")
    
    grid = parse_fen_to_grid(test_fen)
    print("\nParsed grid (rank 8 to rank 1):")
    for i, row in enumerate(grid):
        print(f"Rank {8-i}: {row}")
    
    # Test with labels (2D format - default)
    labels = fen_to_labels(test_fen, flatten=False)
    print(f"\nLabels shape (2D): {labels.shape}")
    print(f"Labels (rank 8 to rank 1):")
    for i in range(8):
        print(f"Rank {8-i}: {labels[i]}")
    
    # Test with labels (flattened format)
    labels_flat = fen_to_labels(test_fen, flatten=True)
    print(f"\nLabels shape (flattened): {labels_flat.shape}")
    print(f"First 16 values: {labels_flat[:16]}")
    
    # Test with a more complex position
    complex_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    print(f"\n\nTesting complex FEN: {complex_fen}")
    grid2 = parse_fen_to_grid(complex_fen)
    print("Grid:")
    for i, row in enumerate(grid2):
        print(f"Rank {8-i}: {row}")

