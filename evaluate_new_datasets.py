#!/usr/bin/env python3
"""
Evaluate the trained model on new datasets (ChessReD and Kaggle).
Runs inference on 100 images from each dataset and calculates accuracy.
"""

import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
from data.dataset import extract_squares_with_padding, get_default_transforms
from data.board_detection import get_chess_board_upper_look
from data.data_loader import fen_to_labels, grid_to_fen, NUM_CLASSES
from models.classifier import create_model
from inference.predictor import BoardPredictor
from training.trainer import get_device


def evaluate_dataset(
    model: torch.nn.Module,
    predictor: BoardPredictor,
    dataset_path: Path,
    dataset_name: str,
    num_samples: int = 100
):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        predictor: BoardPredictor instance
        dataset_path: Path to dataset folder
        dataset_name: Name of dataset for reporting
        num_samples: Number of images to evaluate
        
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")
    
    # Find CSV file
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV file found in {dataset_path}")
        return None
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in CSV")
    
    # Find images directory
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path / "tagged_images"
    if not images_dir.exists():
        print(f"ERROR: No images directory found in {dataset_path}")
        return None
    
    # Sample images (or use first N if less than num_samples available)
    num_available = len(df)
    num_to_evaluate = min(num_samples, num_available)
    
    # Get frame column name
    frame_col = None
    for col in ['from_frame', 'frame', 'frame_num']:
        if col in df.columns:
            frame_col = col
            break
    
    if frame_col is None:
        print(f"ERROR: No frame column found in CSV")
        return None
    
    # Sample indices
    if num_available > num_samples:
        sample_indices = np.random.choice(num_available, num_samples, replace=False)
    else:
        sample_indices = np.arange(num_available)
    
    print(f"Evaluating {num_to_evaluate} images...")
    
    # Evaluation metrics
    correct_squares = 0
    total_squares = 0
    correct_boards = 0
    total_boards = 0
    fen_matches = 0
    
    # Per-class accuracy
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}
    
    # Confidence statistics
    confidences_list = []
    
    # Transform
    transform = get_default_transforms(is_training=False)
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx in tqdm(sample_indices, desc=f"  Processing {dataset_name}"):
        try:
            row = df.iloc[idx]
            frame_num = int(row[frame_col])
            true_fen = row['fen']
            
            # Build image path
            img_path = images_dir / f"frame_{frame_num:06d}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"frame_{frame_num:06d}.png"
            if not img_path.exists():
                # Try alternative naming
                img_stem = Path(row.get('image_path', '')).stem if 'image_path' in row else None
                if img_stem:
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        alt_path = images_dir / f"{img_stem}{ext}"
                        if alt_path.exists():
                            img_path = alt_path
                            break
                
            if not img_path.exists():
                failed += 1
                continue
            
            # Load and process image
            img = Image.open(img_path).convert("RGB")
            
            # Apply perspective transformation
            img_warped = get_chess_board_upper_look(img, output_size=512, use_perspective=True)
            
            # Extract squares
            squares = extract_squares_with_padding(
                img_warped,
                board_size=512,
                square_size=224,
                pad_ratio=0.5
            )
            
            # Apply transforms
            squares_tensor = torch.stack([transform(sq) for sq in squares])
            
            # Predict
            pred_grid, pred_fen, confidences = predictor.predict_board(
                squares_tensor, 
                return_confidences=True
            )
            
            # Get ground truth grid
            true_grid = fen_to_labels(true_fen, flatten=False)
            
            # Calculate square-level accuracy
            board_correct = 0
            for r in range(8):
                for c in range(8):
                    total_squares += 1
                    pred_val = pred_grid[r, c]
                    true_val = true_grid[r, c]
                    
                    # Skip unknown squares in accuracy calculation
                    if pred_val == 'unknown':
                        continue
                    
                    # Count per class
                    if isinstance(true_val, (int, np.integer)):
                        class_total[true_val] = class_total.get(true_val, 0) + 1
                    
                    if pred_val == true_val:
                        correct_squares += 1
                        board_correct += 1
                        if isinstance(true_val, (int, np.integer)):
                            class_correct[true_val] = class_correct.get(true_val, 0) + 1
            
            # Board-level accuracy (all 64 squares correct)
            if board_correct == 64:
                correct_boards += 1
            total_boards += 1
            
            # FEN match
            if pred_fen == true_fen:
                fen_matches += 1
            
            # Confidence statistics
            if confidences is not None:
                confidences_list.extend(confidences.flatten().tolist())
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing image {idx}: {e}")
            failed += 1
            continue
    
    # Calculate metrics
    square_accuracy = correct_squares / total_squares if total_squares > 0 else 0
    board_accuracy = correct_boards / total_boards if total_boards > 0 else 0
    fen_accuracy = fen_matches / total_boards if total_boards > 0 else 0
    
    # Per-class accuracy
    per_class_acc = {}
    for class_id in range(NUM_CLASSES):
        if class_total[class_id] > 0:
            per_class_acc[class_id] = class_correct[class_id] / class_total[class_id]
    
    # Confidence statistics
    conf_mean = np.mean(confidences_list) if confidences_list else 0
    conf_std = np.std(confidences_list) if confidences_list else 0
    conf_min = np.min(confidences_list) if confidences_list else 0
    conf_max = np.max(confidences_list) if confidences_list else 0
    
    results = {
        'dataset': dataset_name,
        'samples_evaluated': successful,
        'samples_failed': failed,
        'square_accuracy': square_accuracy,
        'board_accuracy': board_accuracy,
        'fen_accuracy': fen_accuracy,
        'correct_squares': correct_squares,
        'total_squares': total_squares,
        'correct_boards': correct_boards,
        'total_boards': total_boards,
        'fen_matches': fen_matches,
        'per_class_accuracy': per_class_acc,
        'confidence_mean': conf_mean,
        'confidence_std': conf_std,
        'confidence_min': conf_min,
        'confidence_max': conf_max
    }
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    print(f"  Samples evaluated: {successful}/{num_to_evaluate}")
    print(f"  Samples failed: {failed}")
    print(f"  Square-level accuracy: {square_accuracy*100:.2f}% ({correct_squares}/{total_squares})")
    print(f"  Board-level accuracy: {board_accuracy*100:.2f}% ({correct_boards}/{total_boards})")
    print(f"  FEN match accuracy: {fen_accuracy*100:.2f}% ({fen_matches}/{total_boards})")
    print(f"  Mean confidence: {conf_mean:.3f} ± {conf_std:.3f}")
    print(f"  Confidence range: [{conf_min:.3f}, {conf_max:.3f}]")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate model on new datasets (ChessReD and Kaggle)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/best_model.pth",
        help="Path to model weights"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing dataset folders"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of images to evaluate per dataset (default: 100)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="OOD confidence threshold (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = create_model(backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Create predictor
    predictor = BoardPredictor(model, device, threshold=args.threshold)
    
    # Find datasets
    data_path = Path(args.data_root)
    datasets = []
    
    # Look for ChessReD dataset
    chessred_path = data_path / "game0_chessred"
    if chessred_path.exists():
        datasets.append((chessred_path, "ChessReD"))
    else:
        print(f"\nWarning: ChessReD dataset not found at {chessred_path}")
        print("  Run: python download_chessred.py")
    
    # Look for Kaggle dataset
    kaggle_path = data_path / "game1_kaggle"
    if kaggle_path.exists():
        datasets.append((kaggle_path, "Kaggle Chess Positions"))
    else:
        print(f"\nWarning: Kaggle dataset not found at {kaggle_path}")
        print("  Run: python download_kaggle_chess.py")
    
    if not datasets:
        print("\nERROR: No datasets found!")
        print("Please download at least one dataset first:")
        print("  python download_chessred.py")
        print("  python download_kaggle_chess.py")
        return
    
    # Evaluate each dataset
    all_results = []
    for dataset_path, dataset_name in datasets:
        results = evaluate_dataset(
            model=model,
            predictor=predictor,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            num_samples=args.num_samples
        )
        if results:
            all_results.append(results)
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for results in all_results:
            print(f"\n{results['dataset']}:")
            print(f"  Square Accuracy: {results['square_accuracy']*100:.2f}%")
            print(f"  Board Accuracy:  {results['board_accuracy']*100:.2f}%")
            print(f"  FEN Accuracy:    {results['fen_accuracy']*100:.2f}%")
        
        # Overall average
        avg_square = np.mean([r['square_accuracy'] for r in all_results])
        avg_board = np.mean([r['board_accuracy'] for r in all_results])
        avg_fen = np.mean([r['fen_accuracy'] for r in all_results])
        
        print(f"\n{'='*60}")
        print("OVERALL AVERAGE:")
        print(f"  Square Accuracy: {avg_square*100:.2f}%")
        print(f"  Board Accuracy:  {avg_board*100:.2f}%")
        print(f"  FEN Accuracy:    {avg_fen*100:.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
