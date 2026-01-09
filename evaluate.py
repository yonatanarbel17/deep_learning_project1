#!/usr/bin/env python3
"""
Evaluation script for Chess Square Classifier.

Usage:
    python evaluate.py --model outputs/best_model.pth --image path/to/board.jpg

This script:
1. Loads a trained model
2. Processes a single board image
3. Outputs the predicted FEN and board visualization
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import extract_squares_with_padding, get_default_transforms
from data.data_loader import grid_to_fen
from models.classifier import create_model
from inference.predictor import BoardPredictor, get_prediction_summary
from training.trainer import get_device
from utils.visualization import visualize_prediction


def evaluate_single_image(
    model_path: str,
    image_path: str,
    threshold: float = 0.5,
    output_dir: str = "outputs",
    board_size: int = 512,
    square_size: int = 224
):
    """
    Evaluate a single board image.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to board image
        threshold: OOD confidence threshold
        output_dir: Where to save outputs
        board_size: Size to resize board
        square_size: Size of extracted squares
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
    
    # Load and process image
    print(f"Processing {image_path}...")
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((board_size, board_size), Image.BILINEAR)
    
    # Extract squares
    squares = extract_squares_with_padding(
        img_resized,
        board_size=board_size,
        square_size=square_size,
        pad_ratio=0.5
    )
    
    # Apply transforms
    transform = get_default_transforms(is_training=False)
    squares_tensor = torch.stack([transform(sq) for sq in squares])
    
    # Predict
    grid, fen, confidences = predictor.predict_board(squares_tensor, return_confidences=True)
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted FEN: {fen}")
    
    summary = get_prediction_summary(grid)
    print(f"\nBoard Summary:")
    print(f"  White pieces: {summary['white_pieces']}")
    print(f"  Black pieces: {summary['black_pieces']}")
    print(f"  Empty:        {summary['empty']}")
    print(f"  Unknown:      {summary['unknown']}")
    
    if summary['pieces']:
        print(f"\nPieces detected: {summary['pieces']}")
    
    # Visualize
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction.png")
    visualize_prediction(
        grid=grid,
        fen=fen,
        confidences=confidences,
        output_path=output_path,
        title=f"Prediction (threshold={threshold})"
    )
    
    print(f"\nVisualization saved to: {output_path}")
    print("="*60)
    
    return grid, fen


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chess Square Classifier")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model weights (.pth)")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to board image")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="OOD confidence threshold")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--board_size", type=int, default=512,
                       help="Board resize size")
    parser.add_argument("--square_size", type=int, default=224,
                       help="Square size")
    
    args = parser.parse_args()
    
    evaluate_single_image(
        model_path=args.model,
        image_path=args.image,
        threshold=args.threshold,
        output_dir=args.output_dir,
        board_size=args.board_size,
        square_size=args.square_size
    )


if __name__ == "__main__":
    main()

