#!/usr/bin/env python3
"""
Required evaluation function for Project 1: Chessboard State Prediction.

Usage:
    from predict import predict_board
    result = predict_board(image)  # image: np.ndarray (H, W, 3), uint8, RGB

    # Or from command line:
    python predict.py --image path/to/board.jpg
    python predict.py --game_dir data/game2_per_frame --frame 588
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.board_detection import get_chess_board_upper_look
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device

# ============================================================
# Class ID mapping: Our model IDs → Professor's required IDs
# ============================================================
# Our model:     P=0, N=1, B=2, R=3, Q=4, K=5, p=6, n=7, b=8, r=9, q=10, k=11, empty=12, occluded=13
# Professor's:   WP=0, WR=1, WN=2, WB=3, WQ=4, WK=5, BP=6, BR=7, BN=8, BB=9, BQ=10, BK=11, empty=12, OOD=13
OUR_TO_PROF = {
    0: 0,    # White Pawn → White Pawn
    1: 2,    # White Knight → White Knight
    2: 3,    # White Bishop → White Bishop
    3: 1,    # White Rook → White Rook
    4: 4,    # White Queen → White Queen
    5: 5,    # White King → White King
    6: 6,    # Black Pawn → Black Pawn
    7: 8,    # Black Knight → Black Knight
    8: 9,    # Black Bishop → Black Bishop
    9: 7,    # Black Rook → Black Rook
    10: 10,  # Black Queen → Black Queen
    11: 11,  # Black King → Black King
    12: 12,  # Empty → Empty
    13: 13,  # Occluded → OOD
}

# Singleton model cache to avoid reloading on every call
_model_cache = {"model": None, "predictor": None, "device": None}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pth")


def _load_model():
    """Load model once and cache it."""
    if _model_cache["model"] is not None:
        return _model_cache["predictor"], _model_cache["device"]

    device = get_device()
    model = create_model(backbone="efficientnet_b2", pretrained=False, freeze_ratio=0.0, dropout=0.3)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictor = BoardPredictor(model=model, device=device, threshold=0.1, temperature=1.2444)

    _model_cache["model"] = model
    _model_cache["predictor"] = predictor
    _model_cache["device"] = device

    return predictor, device


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Args:
        image: np.ndarray, shape (H, W, 3), dtype uint8, RGB, values [0, 255]

    Returns:
        torch.Tensor, shape (8, 8), dtype torch.int64, device CPU
        Values in [0, 14] per the required class encoding:
            0=White Pawn, 1=White Rook, 2=White Knight, 3=White Bishop,
            4=White Queen, 5=White King, 6=Black Pawn, 7=Black Rook,
            8=Black Knight, 9=Black Bishop, 10=Black Queen, 11=Black King,
            12=Empty, 13=OOD/Unknown
    """
    predictor, device = _load_model()

    # Convert numpy RGB to PIL Image
    pil_img = Image.fromarray(image)

    # Board detection and perspective correction
    img_rectified = get_chess_board_upper_look(pil_img, output_size=512, use_perspective=True)

    # Extract 64 squares with contextual padding
    squares = extract_squares_with_padding(img_rectified, board_size=512, square_size=224, pad_ratio=0.7)
    transform = get_default_transforms(is_training=False)
    squares_tensor = torch.stack([transform(sq) for sq in squares])

    # Run model prediction
    grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)

    # Convert our model's class IDs to professor's required encoding
    output = torch.zeros(8, 8, dtype=torch.int64)
    for row in range(8):
        for col in range(8):
            cell = grid[row, col]
            if cell == 'unknown':
                output[row, col] = 13  # OOD
            else:
                our_id = int(cell)
                output[row, col] = OUR_TO_PROF.get(our_id, 13)

    # Save output image to ./results/
    _save_visualization(pil_img, grid, fen, confidences)

    return output  # (8, 8), int64, CPU


def _save_visualization(original_img, pred_grid, pred_fen, confidences):
    """Save chess diagram SVG with red X on occluded squares to ./results/."""
    import chess
    import chess.svg

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

    board = chess.Board(fen=None)
    occluded_squares = []
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            cell = pred_grid[row, col]
            if cell == 'unknown' or cell == 13:
                occluded_squares.append(square)
            elif cell in id_to_piece:
                board.set_piece_at(square, id_to_piece[cell])

    # Render SVG with X marks on occluded squares
    svg_str = chess.svg.board(
        board,
        squares=chess.SquareSet(occluded_squares),
        size=800,
        coordinates=True
    )

    # Recolor X marks from black to red
    svg_str = svg_str.replace(
        'fill="#000" stroke="#fff" stroke-width="1.688"',
        'fill="red" stroke="red" stroke-width="0.5"'
    )

    os.makedirs("results", exist_ok=True)
    svg_path = os.path.join("results", "prediction.svg")
    with open(svg_path, 'w') as f:
        f.write(svg_str)


# ============================================================
# Command-line interface
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess board state prediction")
    parser.add_argument("--image", type=str, help="Path to a board image file")
    parser.add_argument("--game_dir", type=str, help="Path to a game directory")
    parser.add_argument("--frame", type=int, help="Frame number (use with --game_dir)")
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
        image_path = str(image_path)
    else:
        parser.error("Provide either --image or both --game_dir and --frame")

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # Load image as numpy RGB uint8
    img = np.array(Image.open(image_path).convert("RGB"))

    # Run prediction
    result = predict_board(img)

    print(f"\nPredicted board state (professor's encoding):")
    print(result)
    print(f"\nShape: {result.shape}, Dtype: {result.dtype}, Device: {result.device}")
    print(f"Visualization saved to: results/prediction.svg")
