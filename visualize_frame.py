#!/usr/bin/env python3
"""
Create a side-by-side visualization of original image and prediction.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.data_loader import grid_to_fen, fen_to_labels, ID_TO_PIECE
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device
from utils.visualization import visualize_prediction


def create_side_by_side_visualization(
    original_image_path: str,
    pred_grid: np.ndarray,
    true_grid: np.ndarray,
    confidences: np.ndarray,
    pred_fen: str,
    true_fen: str,
    output_path: str,
    threshold: float
):
    """Create side-by-side visualization of original image and predictions."""
    # Load original image
    original_img = Image.open(original_image_path).convert("RGB")
    
    # Map class IDs to display characters
    id_to_char = {
        0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔',
        6: '♟', 7: '♞', 8: '♝', 9: '♜', 10: '♛', 11: '♚',
        12: '·', 'unknown': '?'
    }
    
    fig = plt.figure(figsize=(20, 10))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Image\nGame 6, Frame 636', fontsize=18, weight='bold')
    ax1.axis('off')
    
    # Predicted board
    ax2 = plt.subplot(1, 3, 2)
    for row in range(8):
        for col in range(8):
            # Square color
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            rect = plt.Rectangle((col, 7-row), 1, 1, facecolor=color)
            ax2.add_patch(rect)
            
            # Get piece character
            cell = pred_grid[row, col]
            char = id_to_char.get(cell, '?')
            
            # Color for unknown
            text_color = 'red' if cell == 'unknown' else 'black'
            
            # Draw piece
            ax2.text(col + 0.5, 7 - row + 0.5, char,
                   fontsize=56, ha='center', va='center',
                   color=text_color, weight='bold')
            
            # Draw confidence if available
            if confidences is not None:
                conf = confidences[row, col]
                conf_color = 'green' if conf >= 0.7 else 'orange' if conf >= 0.4 else 'red'
                ax2.text(col + 0.85, 7 - row + 0.15, f'{conf:.2f}',
                       fontsize=14, ha='right', va='bottom',
                       color=conf_color, weight='bold')
    
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')
    
    # File labels (a-h)
    for i, label in enumerate('abcdefgh'):
        ax2.text(i + 0.5, -0.3, label, fontsize=18, ha='center', weight='bold')
    
    # Rank labels (1-8)
    for i in range(8):
        ax2.text(-0.3, i + 0.5, str(i + 1), fontsize=18, ha='center', va='center', weight='bold')
    
    ax2.axis('off')
    ax2.set_title(f'Predicted Board\nFEN: {pred_fen}\nThreshold: {threshold}', 
                 fontsize=16, weight='bold')
    
    # Ground truth board
    ax3 = plt.subplot(1, 3, 3)
    for row in range(8):
        for col in range(8):
            # Square color
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            rect = plt.Rectangle((col, 7-row), 1, 1, facecolor=color)
            ax3.add_patch(rect)
            
            # Get piece character
            cell = true_grid[row, col]
            char = id_to_char.get(cell, '?')
            
            # Draw piece
            ax3.text(col + 0.5, 7 - row + 0.5, char,
                   fontsize=56, ha='center', va='center',
                   color='black', weight='bold')
    
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 8)
    ax3.set_aspect('equal')
    
    # File labels (a-h)
    for i, label in enumerate('abcdefgh'):
        ax3.text(i + 0.5, -0.3, label, fontsize=18, ha='center', weight='bold')
    
    # Rank labels (1-8)
    for i in range(8):
        ax3.text(-0.3, i + 0.5, str(i + 1), fontsize=18, ha='center', va='center', weight='bold')
    
    ax3.axis('off')
    ax3.set_title(f'Ground Truth\nFEN: {true_fen}', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Side-by-side visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize model prediction on a frame")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model weights (.pth)")
    parser.add_argument("--game", type=int, required=True,
                       help="Game number")
    parser.add_argument("--frame", type=int, required=True,
                       help="Frame number")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="OOD confidence threshold")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    device = get_device()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = create_model(backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Create predictor
    predictor = BoardPredictor(model, device, threshold=args.threshold)
    
    # Find image path
    game_folder = Path(args.data_root) / f"game{args.game}_per_frame"
    if not game_folder.exists():
        # Try alternative naming
        data_path = Path(args.data_root)
        game_folders = [f for f in data_path.iterdir() if f.is_dir() and f"game{args.game}" in f.name.lower()]
        if game_folders:
            game_folder = game_folders[0]
        else:
            raise FileNotFoundError(f"Game {args.game} not found in {args.data_root}")
    
    images_dir = game_folder / "tagged_images"
    if not images_dir.exists():
        images_dir = game_folder / "images"
    
    img_path = images_dir / f"frame_{args.frame:06d}.jpg"
    if not img_path.exists():
        img_path = images_dir / f"frame_{args.frame:06d}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Frame {args.frame} not found in {game_folder}")
    
    print(f"Processing {img_path}...")
    
    # Load and process image
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((512, 512), Image.BILINEAR)
    
    # Extract squares
    squares = extract_squares_with_padding(
        img_resized,
        board_size=512,
        square_size=224,
        pad_ratio=0.5
    )
    
    # Apply transforms
    transform = get_default_transforms(is_training=False)
    squares_tensor = torch.stack([transform(sq) for sq in squares])
    
    # Predict
    grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)
    
    # Get ground truth
    csv_path = game_folder / f"game{args.game}.csv"
    df = pd.read_csv(csv_path)
    
    if 'from_frame' in df.columns:
        true_row = df[df['from_frame'] == args.frame]
    elif 'frame' in df.columns:
        true_row = df[df['frame'] == args.frame]
    else:
        true_row = pd.DataFrame()
    
    if len(true_row) > 0:
        true_fen = true_row.iloc[0]['fen']
        true_grid = fen_to_labels(true_fen, flatten=False)
    else:
        true_fen = "Not found in CSV"
        true_grid = None
    
    # Calculate accuracy
    if true_grid is not None:
        correct = 0
        total = 0
        for r in range(8):
            for c in range(8):
                total += 1
                pred_val = grid[r, c]
                true_val = true_grid[r, c]
                
                # Handle 'unknown' - treat as wrong for accuracy calculation
                if pred_val == 'unknown':
                    continue
                
                if pred_val == true_val:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nSquare-level Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    # Create visualization
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"game{args.game}_frame{args.frame:06d}_visualization.png"
    )
    
    if true_grid is not None:
        create_side_by_side_visualization(
            original_image_path=str(img_path),
            pred_grid=grid,
            true_grid=true_grid,
            confidences=confidences,
            pred_fen=fen,
            true_fen=true_fen,
            output_path=output_path,
            threshold=args.threshold
        )
    else:
        # Just show prediction if no ground truth
        visualize_prediction(
            grid=grid,
            fen=fen,
            confidences=confidences,
            output_path=output_path,
            title=f"Game {args.game}, Frame {args.frame} (threshold={args.threshold})"
        )
    
    print(f"\nPredicted FEN: {fen}")
    if true_fen != "Not found in CSV":
        print(f"Ground Truth FEN: {true_fen}")
        print(f"FEN Match: {fen == true_fen}")


if __name__ == "__main__":
    main()
