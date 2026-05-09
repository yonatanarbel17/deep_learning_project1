#!/usr/bin/env python3
"""
Demo script: Run inference on a chess board image and output the predicted FEN.

Usage:
    python demo.py --image path/to/board_image.jpg
    python demo.py --image path/to/board_image.jpg --output_dir results
    python demo.py --game_dir data/game2_per_frame --frame 200

The script will:
1. Load the trained EfficientNet-B2 model
2. Detect and rectify the chess board
3. Extract and classify all 64 squares
4. Output the predicted FEN string
5. Save a visualization to the output directory
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.board_detection import get_chess_board_upper_look
from data.data_loader import ID_TO_PIECE
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device


def grid_to_chess_board(pred_grid):
    """Convert prediction grid to a python-chess Board, returning board and list of occluded squares."""
    import chess
    # Map our class IDs to python-chess piece objects
    id_to_piece = {
        0: chess.Piece(chess.PAWN, chess.WHITE),
        1: chess.Piece(chess.KNIGHT, chess.WHITE),
        2: chess.Piece(chess.BISHOP, chess.WHITE),
        3: chess.Piece(chess.ROOK, chess.WHITE),
        4: chess.Piece(chess.QUEEN, chess.WHITE),
        5: chess.Piece(chess.KING, chess.WHITE),
        6: chess.Piece(chess.PAWN, chess.BLACK),
        7: chess.Piece(chess.KNIGHT, chess.BLACK),
        8: chess.Piece(chess.BISHOP, chess.BLACK),
        9: chess.Piece(chess.ROOK, chess.BLACK),
        10: chess.Piece(chess.QUEEN, chess.BLACK),
        11: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board(fen=None)  # empty board
    occluded_squares = []
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)  # chess lib: a1=0, row 0 = rank 1
            cell = pred_grid[row, col]
            if cell == 'unknown' or cell == 13:
                occluded_squares.append(square)
            elif cell in id_to_piece:
                board.set_piece_at(square, id_to_piece[cell])
    return board, occluded_squares


def visualize_prediction(original_img, pred_grid, pred_fen, confidences, output_path,
                         true_grid=None, true_fen=None, threshold=0.1):
    """Create chess diagram SVG with red X on occluded squares."""
    import chess
    import chess.svg

    # Build chess board from predictions
    board, occluded_squares = grid_to_chess_board(pred_grid)

    # Render board as SVG
    svg_str = chess.svg.board(board, size=800, coordinates=True)

    # python-chess SVG uses viewBox="0 0 390 390", margin=15, square=45x45
    # Square at file f, rank r: x = 15 + f*45, y = 15 + (7-r)*45
    margin = 15
    sq_size = 45
    x_marks_svg = ""
    for sq in occluded_squares:
        f = chess.square_file(sq)   # 0=a, 7=h
        r = chess.square_rank(sq)   # 0=rank1, 7=rank8
        x = margin + f * sq_size
        y = margin + (7 - r) * sq_size
        pad = 4  # pixels padding from edge
        x_marks_svg += (
            f'<line x1="{x+pad}" y1="{y+pad}" x2="{x+sq_size-pad}" y2="{y+sq_size-pad}" '
            f'stroke="red" stroke-width="7" stroke-linecap="round"/>\n'
            f'<line x1="{x+sq_size-pad}" y1="{y+pad}" x2="{x+pad}" y2="{y+sq_size-pad}" '
            f'stroke="red" stroke-width="7" stroke-linecap="round"/>\n'
        )

    if x_marks_svg:
        svg_str = svg_str.replace('</svg>', x_marks_svg + '</svg>')

    # Save SVG
    svg_path = output_path if output_path.endswith('.svg') else output_path.replace('.png', '.svg')
    with open(svg_path, 'w') as f:
        f.write(svg_str)
    print(f"Board diagram saved to: {svg_path}")


def run_inference(image_path, model_path, output_dir, game_dir=None, frame=None,
                  threshold=0.1, temperature=1.2444):
    """Run inference on a single chess board image."""
    device = get_device()
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model(backbone="efficientnet_b2", pretrained=False, freeze_ratio=0.0, dropout=0.3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create predictor
    predictor = BoardPredictor(model=model, device=device, threshold=threshold, temperature=temperature)

    # Load and process image
    print(f"Processing: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_rectified = get_chess_board_upper_look(img, output_size=512, use_perspective=True)

    # Extract squares
    squares = extract_squares_with_padding(img_rectified, board_size=512, square_size=224, pad_ratio=0.7)
    transform = get_default_transforms(is_training=False)
    squares_tensor = torch.stack([transform(sq) for sq in squares])

    # Predict
    grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)

    # Output
    print(f"\nPredicted FEN: {fen}")

    # Count pieces
    unknown_count = sum(1 for r in range(8) for c in range(8) if grid[r, c] == 'unknown')
    if unknown_count > 0:
        print(f"Occluded/unknown squares: {unknown_count}")

    # Try to load ground truth if game_dir and frame are provided
    true_grid = None
    true_fen = None
    if game_dir and frame is not None:
        import pandas as pd
        from data.data_loader import fen_to_labels
        game_path = Path(game_dir)
        csv_files = list(game_path.glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            col = 'from_frame' if 'from_frame' in df.columns else 'frame'
            if col in df.columns:
                true_row = df[df[col] == frame]
                if len(true_row) > 0:
                    true_fen = true_row.iloc[0]['fen']
                    true_grid = fen_to_labels(true_fen, flatten=False)
                    print(f"Ground Truth FEN: {true_fen}")
                    print(f"FEN Match: {fen == true_fen}")

                    # Calculate accuracy
                    correct = 0
                    total = 0
                    for r in range(8):
                        for c in range(8):
                            total += 1
                            if grid[r, c] != 'unknown' and grid[r, c] == true_grid[r, c]:
                                correct += 1
                    print(f"Square-level Accuracy: {correct/total*100:.2f}% ({correct}/{total})")

    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    img_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{img_name}_prediction.svg")
    visualize_prediction(img, grid, fen, confidences, output_path,
                         true_grid=true_grid, true_fen=true_fen, threshold=threshold)

    return fen, grid


def main():
    parser = argparse.ArgumentParser(description="Chess board inference demo")
    parser.add_argument("--image", type=str, help="Path to a board image file")
    parser.add_argument("--game_dir", type=str, help="Path to a game directory (e.g., data/game2_per_frame)")
    parser.add_argument("--frame", type=int, help="Frame number (use with --game_dir)")
    parser.add_argument("--model", type=str, default="outputs/best_model.pth", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
    elif args.game_dir and args.frame is not None:
        game_dir = Path(args.game_dir)
        images_dir = game_dir / "tagged_images"
        if not images_dir.exists():
            images_dir = game_dir / "images"
        image_path = images_dir / f"frame_{args.frame:06d}.jpg"
        if not image_path.exists():
            image_path = images_dir / f"frame_{args.frame:06d}.png"
        if not image_path.exists():
            print(f"ERROR: Frame {args.frame} not found in {game_dir}")
            sys.exit(1)
        image_path = str(image_path)
    else:
        parser.error("Provide either --image or both --game_dir and --frame")

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    run_inference(image_path, args.model, args.output_dir,
                  game_dir=args.game_dir, frame=args.frame)


if __name__ == "__main__":
    main()
