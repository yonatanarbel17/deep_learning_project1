#!/usr/bin/env python3
"""
Run inference on tr1.jpg and create comprehensive visualization.
"""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.board_detection import get_chess_board_upper_look
from data.data_loader import grid_to_fen, ID_TO_PIECE
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device
from utils.visualization import visualize_prediction


def create_comprehensive_visualization(
    original_image_path: str,
    pred_grid: np.ndarray,
    confidences: np.ndarray,
    pred_fen: str,
    output_path: str,
    threshold: float
):
    """Create visualization with original, warped, and prediction."""
    # Load original image
    original_img = Image.open(original_image_path).convert("RGB")
    
    # Get warped board
    warped_board = get_chess_board_upper_look(original_img, output_size=800, use_perspective=True)
    
    # Map class IDs to display characters
    id_to_char = {
        0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔',
        6: '♟', 7: '♞', 8: '♝', 9: '♜', 10: '♛', 11: '♚',
        12: '·', 'unknown': '?'
    }
    
    fig = plt.figure(figsize=(24, 8))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title('1. Original Image', fontsize=18, weight='bold')
    ax1.axis('off')
    
    # Warped board
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(warped_board)
    ax2.set_title('2. Top-Down View (Warped)', fontsize=18, weight='bold')
    ax2.axis('off')
    
    # Predicted board
    ax3 = plt.subplot(1, 3, 3)
    for row in range(8):
        for col in range(8):
            # Square color
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            rect = plt.Rectangle((col, 7-row), 1, 1, facecolor=color)
            ax3.add_patch(rect)
            
            # Get piece character
            cell = pred_grid[row, col]
            char = id_to_char.get(cell, '?')
            
            # Color for unknown
            text_color = 'red' if cell == 'unknown' else 'black'
            
            # Draw piece
            ax3.text(col + 0.5, 7 - row + 0.5, char,
                   fontsize=56, ha='center', va='center',
                   color=text_color, weight='bold')
            
            # Draw confidence if available
            if confidences is not None:
                conf = confidences[row, col]
                conf_color = 'green' if conf >= 0.7 else 'orange' if conf >= 0.4 else 'red'
                ax3.text(col + 0.85, 7 - row + 0.15, f'{conf:.2f}',
                       fontsize=12, ha='right', va='bottom',
                       color=conf_color, weight='bold')
    
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
    ax3.set_title(f'3. Predicted Board\nFEN: {pred_fen}\nThreshold: {threshold}', 
                 fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--model", type=str, default="outputs/best_model.pth",
                       help="Path to model weights")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="OOD confidence threshold")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = create_model(backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Create predictor
    predictor = BoardPredictor(model, device, threshold=args.threshold)
    
    # Load and process image
    image_path = Path(args.image).expanduser()
    print(f"Processing {image_path}...")
    
    img = Image.open(image_path).convert("RGB")
    print(f"Original image size: {img.size}")
    
    # Apply perspective transformation
    img_warped = get_chess_board_upper_look(img, output_size=512, use_perspective=True)
    print(f"Warped board size: {img_warped.size}")
    
    # Extract squares
    squares = extract_squares_with_padding(
        img_warped,
        board_size=512,
        square_size=224,
        pad_ratio=0.5
    )
    
    # Apply transforms
    transform = get_default_transforms(is_training=False)
    squares_tensor = torch.stack([transform(sq) for sq in squares])
    
    # Predict
    print("\nRunning inference...")
    grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Predicted FEN: {fen}")
    
    # Count pieces
    unique, counts = np.unique(grid, return_counts=True)
    piece_counts = dict(zip(unique, counts))
    
    print(f"\nSquare classifications:")
    for piece, count in sorted(piece_counts.items()):
        piece_name = ID_TO_PIECE.get(piece, str(piece)) if isinstance(piece, (int, np.integer)) else piece
        print(f"  {piece_name}: {count}")
    
    # Confidence statistics
    if confidences is not None:
        print(f"\nConfidence statistics:")
        print(f"  Mean: {confidences.mean():.3f}")
        print(f"  Min: {confidences.min():.3f}")
        print(f"  Max: {confidences.max():.3f}")
        print(f"  Squares with confidence < 0.5: {np.sum(confidences < 0.5)}")
        print(f"  Squares with confidence < 0.3: {np.sum(confidences < 0.3)}")
    
    # Create visualization
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "tr1_inference_result.png")
    
    create_comprehensive_visualization(
        original_image_path=str(image_path),
        pred_grid=grid,
        confidences=confidences,
        pred_fen=fen,
        output_path=output_path,
        threshold=args.threshold
    )
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
