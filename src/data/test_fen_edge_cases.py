"""
Test edge cases for FEN parser.
"""

# Import the functions directly (standalone test)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import only what we need (avoid numpy dependency for now)
def parse_fen_to_grid(fen: str):
    """Standalone version for testing."""
    fen_board = fen.split(' ')[0].strip()
    ranks = fen_board.split('/')
    
    if len(ranks) != 8:
        raise ValueError(f"FEN must have exactly 8 ranks, got {len(ranks)}")
    
    grid = []
    valid_pieces = set('KQRBNPkqrbnp')
    
    for rank_idx, rank in enumerate(ranks):
        grid_row = []
        i = 0
        
        while i < len(rank):
            char = rank[i]
            
            if char.isdigit():
                num_str = ""
                while i < len(rank) and rank[i].isdigit():
                    num_str += rank[i]
                    i += 1
                
                num_empty = int(num_str)
                if num_empty == 0 or num_empty > 8:
                    raise ValueError(f"Invalid number of empty squares: {num_empty}")
                
                grid_row.extend(['empty'] * num_empty)
                continue
                
            elif char in valid_pieces:
                grid_row.append(char)
            else:
                raise ValueError(f"Invalid character '{char}' in FEN")
            
            i += 1
        
        if len(grid_row) != 8:
            raise ValueError(f"Rank must have exactly 8 squares, got {len(grid_row)}")
        
        grid.append(grid_row)
    
    return grid

def grid_to_fen(grid):
    """Standalone version for testing."""
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

def test_edge_cases():
    print("=" * 60)
    print("Testing FEN Parser Edge Cases")
    print("=" * 60)
    
    # Test 1: Full FEN string (with all 6 fields)
    print("\n1. Testing full FEN string (with all fields):")
    full_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    grid = parse_fen_to_grid(full_fen)
    print(f"   Successfully parsed full FEN")
    print(f"   Rank 8: {grid[0]}")
    
    # Test 2: Complex position with mixed numbers
    print("\n2. Testing complex position:")
    complex_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    grid2 = parse_fen_to_grid(complex_fen)
    print(f"   Successfully parsed")
    print(f"   Rank 8: {grid2[0]}")
    print(f"   Rank 5: {grid2[3]}")
    
    # Test 3: Position with single pieces and many empties
    print("\n3. Testing position with isolated pieces:")
    isolated_fen = "8/3k4/8/3K4/8/8/8/8"
    grid3 = parse_fen_to_grid(isolated_fen)
    print(f"   Successfully parsed")
    print(f"   Rank 7 (should have king at file d): {grid3[1]}")
    print(f"   Rank 4 (should have king at file d): {grid3[4]}")
    
    # Test 4: Round-trip conversion
    print("\n4. Testing round-trip conversion:")
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        "8/8/8/8/8/8/8/8",
        "8/3k4/8/3K4/8/8/8/8"
    ]
    
    for fen in test_fens:
        grid = parse_fen_to_grid(fen)
        reconstructed = grid_to_fen(grid)
        match = (fen == reconstructed)
        print(f"   FEN: {fen[:40]}...")
        print(f"   Match: {match}")
        if not match:
            print(f"   Original:  {fen}")
            print(f"   Reconstructed: {reconstructed}")
    
    # Test 5: Error cases (should raise ValueError)
    print("\n5. Testing error handling:")
    error_cases = [
        ("7/8/8/8/8/8/8/8", "Rank with 7 squares (should be 8)"),
        ("9/8/8/8/8/8/8/8", "Rank with 9 squares (should be 8)"),
        ("8/8/8/8/8/8/8", "Only 7 ranks (should be 8)"),
        ("8/8/8/8/8/8/8/8/8", "9 ranks (should be 8)"),
    ]
    
    for invalid_fen, description in error_cases:
        try:
            parse_fen_to_grid(invalid_fen)
            print(f"   ERROR: Should have raised ValueError for: {description}")
        except ValueError as e:
            print(f"   ✓ Correctly raised ValueError for: {description}")
            print(f"     Error: {str(e)[:60]}...")
    
    print("\n" + "=" * 60)
    print("Edge case tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_edge_cases()

