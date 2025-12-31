"""
Test script to parse a FEN string and output the grid to a file.
"""

# Standalone FEN parser (no dependencies)
def parse_fen_to_grid(fen: str):
    """Converts FEN string to 8x8 grid."""
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

# Test FEN string
test_fen = "r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1"

print(f"Parsing FEN: {test_fen}")
print("=" * 60)

# Parse FEN to grid
grid = parse_fen_to_grid(test_fen)

# Create output file
output_file = "fen_grid_output.txt"

with open(output_file, 'w') as f:
    f.write("FEN String:\n")
    f.write(f"{test_fen}\n\n")
    f.write("=" * 60 + "\n")
    f.write("Parsed Grid (8x8):\n")
    f.write("=" * 60 + "\n")
    f.write("Rank 8 (top, black's back rank) to Rank 1 (bottom, white's back rank)\n")
    f.write("Files: a  b  c  d  e  f  g  h\n")
    f.write("-" * 60 + "\n\n")
    
    # Write grid with rank labels
    for i, row in enumerate(grid):
        rank_num = 8 - i
        f.write(f"Rank {rank_num}: ")
        # Format each square nicely
        formatted_row = []
        for square in row:
            if square == 'empty':
                formatted_row.append('.')
            else:
                formatted_row.append(square)
        f.write("  ".join(formatted_row))
        f.write("\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("Visual Representation:\n")
    f.write("=" * 60 + "\n\n")
    
    # Create a visual board representation
    f.write("    a  b  c  d  e  f  g  h\n")
    f.write("  +--+--+--+--+--+--+--+--+\n")
    
    for i, row in enumerate(grid):
        rank_num = 8 - i
        f.write(f"{rank_num} |")
        for square in row:
            if square == 'empty':
                f.write(" .|")
            else:
                f.write(f" {square}|")
        f.write(f" {rank_num}\n")
        f.write("  +--+--+--+--+--+--+--+--+\n")
    
    f.write("    a  b  c  d  e  f  g  h\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("Piece Count:\n")
    f.write("=" * 60 + "\n")
    
    # Count pieces
    piece_count = {}
    for row in grid:
        for square in row:
            if square != 'empty':
                piece_count[square] = piece_count.get(square, 0) + 1
    
    for piece in sorted(piece_count.keys()):
        piece_name = {
            'K': 'White King', 'Q': 'White Queen', 'R': 'White Rook',
            'B': 'White Bishop', 'N': 'White Knight', 'P': 'White Pawn',
            'k': 'Black King', 'q': 'Black Queen', 'r': 'Black Rook',
            'b': 'Black Bishop', 'n': 'Black Knight', 'p': 'Black Pawn'
        }.get(piece, piece)
        f.write(f"{piece} ({piece_name}): {piece_count[piece]}\n")
    
    f.write(f"\nEmpty squares: {sum(1 for row in grid for square in row if square == 'empty')}\n")
    f.write(f"Total squares: {len(grid) * len(grid[0])}\n")

print(f"\nGrid parsed successfully!")
print(f"Output saved to: {output_file}")
print("\nGrid preview:")
print("Rank 8 (top) to Rank 1 (bottom):")
for i, row in enumerate(grid):
    rank_num = 8 - i
    formatted_row = [sq if sq != 'empty' else '.' for sq in row]
    print(f"Rank {rank_num}: {'  '.join(formatted_row)}")
