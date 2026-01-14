#!/usr/bin/env python3
"""
Find and visualize a frame where the model classified at least one cell as "unknown".

Usage:
    python find_unknown_frame.py --model outputs/best_model.pth --data_root /path/to/data --threshold 0.5
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.data_loader import grid_to_fen, fen_to_labels
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device
from utils.visualization import visualize_prediction


def load_game_data(data_root: str, game_num: int):
    """Load data for a specific game."""
    data_path = Path(data_root)
    game_folder = data_path / f"game{game_num}"
    
    if not game_folder.exists():
        # Try alternative naming
        game_folders = [f for f in data_path.iterdir() if f.is_dir() and f"game{game_num}" in f.name.lower()]
        if game_folders:
            game_folder = game_folders[0]
        else:
            raise FileNotFoundError(f"Game {game_num} not found in {data_root}")
    
    # Find CSV
    csv_files = list(game_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {game_folder}")
    
    df = pd.read_csv(csv_files[0])
    
    # Find images directory
    images_dir = game_folder / "tagged_images"
    if not images_dir.exists():
        images_dir = game_folder / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory in {game_folder}")
    
    return df, images_dir


def find_frame_with_unknown(
    model_path: str,
    data_root: str,
    threshold: float = 0.5,
    max_frames: int = 100,
    board_size: int = 512,
    square_size: int = 224
):
    """
    Search for a frame where at least one square is classified as "unknown".
    
    Args:
        model_path: Path to trained model
        data_root: Root directory with game data
        threshold: OOD confidence threshold
        max_frames: Maximum number of frames to check
        board_size: Board resize size
        square_size: Square size
    """
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model(backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create predictor
    predictor = BoardPredictor(model, device, threshold=threshold)
    
    # Load data (try validation games first - typically higher game numbers)
    data_path = Path(data_root)
    game_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and "game" in f.name.lower()])
    
    if not game_folders:
        print(f"ERROR: No game folders found in {data_root}")
        return
    
    print(f"\nSearching through {len(game_folders)} games for frames with 'unknown' classifications...")
    print(f"Using threshold: {threshold}")
    print("-" * 60)
    
    transform = get_default_transforms(is_training=False)
    frames_checked = 0
    
    # Try games in reverse order (validation games are usually last)
    for game_folder in reversed(game_folders):
        try:
            # Extract game number
            game_name = game_folder.name
            game_num = int(''.join(filter(str.isdigit, game_name)))
        except ValueError:
            continue
        
        try:
            df, images_dir = load_game_data(data_root, game_num)
        except FileNotFoundError as e:
            print(f"Skipping {game_folder.name}: {e}")
            continue
        
        print(f"\nChecking Game {game_num} ({len(df)} frames)...")
        
        # Check frames in this game
        for idx, row in df.iterrows():
            if frames_checked >= max_frames:
                print(f"\nReached max_frames limit ({max_frames})")
                return None
            
            # Get frame number
            if 'from_frame' in df.columns:
                frame_num = int(row['from_frame'])
            elif 'frame' in df.columns:
                frame_num = int(row['frame'])
            else:
                continue
            
            # Build image path
            img_path = images_dir / f"frame_{frame_num:06d}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"frame_{frame_num:06d}.png"
            if not img_path.exists():
                continue
            
            frames_checked += 1
            
            # Load and process image
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((board_size, board_size), Image.BILINEAR)
                
                # Extract squares
                squares = extract_squares_with_padding(
                    img_resized,
                    board_size=board_size,
                    square_size=square_size,
                    pad_ratio=0.5
                )
                
                # Apply transforms
                squares_tensor = torch.stack([transform(sq) for sq in squares])
                
                # Predict
                grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)
                
                # Check for unknown
                unknown_count = np.sum(grid == 'unknown')
                
                if unknown_count > 0:
                    print(f"\n{'='*60}")
                    print(f"FOUND FRAME WITH {unknown_count} UNKNOWN CELL(S)!")
                    print(f"{'='*60}")
                    print(f"Game: {game_num}")
                    print(f"Frame: {frame_num}")
                    print(f"Image: {img_path}")
                    print(f"Threshold: {threshold}")
                    print(f"\nPredicted FEN: {fen}")
                    
                    # Show which cells are unknown
                    print(f"\nUnknown cells (row, col):")
                    for r in range(8):
                        for c in range(8):
                            if grid[r, c] == 'unknown':
                                conf = confidences[r, c]
                                square_name = f"{chr(ord('a')+c)}{8-r}"
                                print(f"  {square_name} (row={r}, col={c}): confidence={conf:.4f}")
                    
                    # Get ground truth FEN if available
                    if 'fen' in row:
                        true_fen = row['fen']
                        print(f"\nGround Truth FEN: {true_fen}")
                        
                        # Compare
                        true_grid = fen_to_labels(true_fen, flatten=False)
                        print(f"\nComparison:")
                        for r in range(8):
                            for c in range(8):
                                if grid[r, c] == 'unknown':
                                    true_val = true_grid[r, c]
                                    from data.data_loader import ID_TO_PIECE
                                    true_piece = ID_TO_PIECE.get(true_val, '?')
                                    print(f"  {chr(ord('a')+c)}{8-r}: Predicted=unknown, True={true_piece if true_piece else 'empty'}")
                    
                    # Visualize
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, 
                        f"game{game_num}_frame{frame_num:06d}_unknown_visualization.png"
                    )
                    
                    visualize_prediction(
                        grid=grid,
                        fen=fen,
                        confidences=confidences,
                        output_path=output_path,
                        title=f"Game {game_num}, Frame {frame_num} (threshold={threshold})"
                    )
                    
                    print(f"\nVisualization saved to: {output_path}")
                    print(f"{'='*60}\n")
                    
                    return {
                        'game': game_num,
                        'frame': frame_num,
                        'image_path': str(img_path),
                        'grid': grid,
                        'fen': fen,
                        'confidences': confidences,
                        'unknown_count': unknown_count
                    }
                
                # Progress indicator
                if frames_checked % 10 == 0:
                    print(f"  Checked {frames_checked} frames...", end='\r')
                    
            except Exception as e:
                print(f"  Error processing frame {frame_num}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Checked {frames_checked} frames but found no 'unknown' classifications.")
    print(f"Try lowering the threshold (current: {threshold}) to find more uncertain predictions.")
    print(f"{'='*60}\n")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find a frame where model classifies cells as 'unknown'"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model weights (.pth)")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing game folders")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="OOD confidence threshold (lower = more unknowns)")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum frames to check")
    parser.add_argument("--board_size", type=int, default=512,
                       help="Board resize size")
    parser.add_argument("--square_size", type=int, default=224,
                       help="Square size")
    
    args = parser.parse_args()
    
    result = find_frame_with_unknown(
        model_path=args.model,
        data_root=args.data_root,
        threshold=args.threshold,
        max_frames=args.max_frames,
        board_size=args.board_size,
        square_size=args.square_size
    )
    
    if result is None:
        print("\nNo frame with 'unknown' classifications found.")
        print("Suggestions:")
        print("  1. Lower the threshold (e.g., --threshold 0.3)")
        print("  2. Increase max_frames (e.g., --max_frames 500)")
        print("  3. Check if model is trained and threshold is appropriate")


if __name__ == "__main__":
    main()
