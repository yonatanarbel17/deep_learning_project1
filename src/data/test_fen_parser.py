"""
Standalone test script for FEN parser (no external dependencies).
"""

# Copy the FEN parser functions here for testing
PIECE_TO_ID = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    'empty': 12
}


def parse_fen_to_grid(fen: str):
    """Converts FEN string to 8x8 grid."""
    fen_board = fen.split(' ')[0]
    grid = []
    ranks = fen_board.split('/')
    
    if len(ranks) != 8:
        raise ValueError(f"FEN must have exactly 8 ranks, got {len(ranks)}")
    
    for rank in ranks:
        grid_row = []
        for char in rank:
            if char.isdigit():
                num_empty = int(char)
                grid_row.extend(['empty'] * num_empty)
            elif char in 'KQRBNPkqrbnp':
                grid_row.append(char)
            else:
                raise ValueError(f"Invalid character in FEN: '{char}'")
        
        if len(grid_row) != 8:
            raise ValueError(f"Rank must have exactly 8 squares, got {len(grid_row)}. Rank: {rank}")
        
        grid.append(grid_row)
    
    return grid


def grid_to_fen(grid):
    """Converts 8x8 grid back to FEN."""
    ranks = []
    for row in grid:
        fen_rank = ""
        empty_count = 0
        
        for square in row:
            if square == 'empty' or square == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += square
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        ranks.append(fen_rank)
    
    return '/'.join(ranks)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FEN Parser")
    print("=" * 60)
    
    # Test 1: Standard starting position
    print("\n1. Testing standard starting position:")
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print(f"   Input FEN: {test_fen}")
    
    grid = parse_fen_to_grid(test_fen)
    print(f"   Parsed successfully: {len(grid)} ranks, {len(grid[0])} files per rank")
    
    # Show first rank (rank 8 - black pieces)
    print(f"   Rank 8 (top): {grid[0]}")
    print(f"   Rank 1 (bottom): {grid[7]}")
    
    # Test round-trip
    reconstructed = grid_to_fen(grid)
    print(f"   Reconstructed: {reconstructed}")
    print(f"   Round-trip match: {test_fen == reconstructed}")
    
    # Test 2: Complex position with numbers
    print("\n2. Testing complex position:")
    complex_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    print(f"   Input FEN: {complex_fen}")
    
    grid2 = parse_fen_to_grid(complex_fen)
    print(f"   Parsed successfully")
    print(f"   Rank 8: {grid2[0]}")
    print(f"   Rank 5: {grid2[3]}")  # Should have some empty squares
    
    reconstructed2 = grid_to_fen(grid2)
    print(f"   Reconstructed: {reconstructed2}")
    print(f"   Round-trip match: {complex_fen == reconstructed2}")
    
    # Test 3: All empty board
    print("\n3. Testing empty board:")
    empty_fen = "8/8/8/8/8/8/8/8"
    print(f"   Input FEN: {empty_fen}")
    
    grid3 = parse_fen_to_grid(empty_fen)
    all_empty = all(square == 'empty' for row in grid3 for square in row)
    print(f"   All squares empty: {all_empty}")
    
    # Test 4: Check piece mapping
    print("\n4. Testing piece to ID mapping:")
    test_grid = parse_fen_to_grid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    unique_pieces = set()
    for row in test_grid:
        for square in row:
            if square != 'empty':
                unique_pieces.add(square)
    
    print(f"   Unique pieces found: {sorted(unique_pieces)}")
    print(f"   All pieces have mappings: {all(p in PIECE_TO_ID for p in unique_pieces)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

