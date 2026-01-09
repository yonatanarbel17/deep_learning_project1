#!/usr/bin/env python3
"""
Main training script for Chess Square Classifier.

Usage:
    python train.py --data_root /path/to/data --epochs 15

This script:
1. Loads game data from the specified directory
2. Splits data by game for train/val (prevents data leakage)
3. Trains a ResNet-18 classifier on 64 squares per board
4. Saves model weights and training summary
5. Optimizes OOD threshold on validation set
6. Generates training curves and report
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import ChessboardDataset, get_default_transforms, create_dataloaders
from models.classifier import create_model
from training.trainer import Trainer, get_device
from inference.predictor import find_optimal_threshold
from utils.visualization import plot_training_curves, create_training_report


def load_all_games(data_root: str, game_numbers: list = None) -> pd.DataFrame:
    """
    Load data from all games in the data directory.
    
    Args:
        data_root: Root directory containing game folders
        game_numbers: Optional list of game numbers to load (default: all)
        
    Returns:
        DataFrame with columns: image_path, fen, game_id
    """
    all_data = []
    
    # Find all game folders
    data_path = Path(data_root)
    game_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and "game" in f.name.lower()])
    
    for game_folder in game_folders:
        # Extract game number
        game_name = game_folder.name
        try:
            game_num = int(''.join(filter(str.isdigit, game_name)))
        except ValueError:
            continue
            
        if game_numbers and game_num not in game_numbers:
            continue
        
        # Find CSV file
        csv_files = list(game_folder.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV found in {game_folder}")
            continue
            
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        
        # Find images directory
        images_dir = game_folder / "tagged_images"
        if not images_dir.exists():
            images_dir = game_folder / "images"
        if not images_dir.exists():
            print(f"Warning: No images directory in {game_folder}")
            continue
        
        # Process each row
        for _, row in df.iterrows():
            # Get frame number (handle both 'frame' and 'from_frame' columns)
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
            
            all_data.append({
                "image_path": str(img_path),
                "fen": row['fen'],
                "game_id": game_num
            })
    
    result_df = pd.DataFrame(all_data)
    print(f"Loaded {len(result_df)} samples from {result_df['game_id'].nunique()} games")
    return result_df


def split_by_game(df: pd.DataFrame, val_ratio: float = 0.2) -> tuple:
    """
    Split data by game to prevent data leakage.
    
    Args:
        df: DataFrame with 'game_id' column
        val_ratio: Fraction of games to use for validation
        
    Returns:
        train_df, val_df
    """
    games = df['game_id'].unique()
    n_val_games = max(1, int(len(games) * val_ratio))
    
    # Use last games for validation (more recent/different conditions)
    val_games = set(sorted(games)[-n_val_games:])
    
    train_df = df[~df['game_id'].isin(val_games)].copy()
    val_df = df[df['game_id'].isin(val_games)].copy()
    
    print(f"Train: {len(train_df)} samples from games {sorted(set(games) - val_games)}")
    print(f"Val:   {len(val_df)} samples from games {sorted(val_games)}")
    
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Train Chess Square Classifier")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing game folders")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (boards per batch)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="Model backbone")
    parser.add_argument("--board_size", type=int, default=512,
                       help="Size to resize board images")
    parser.add_argument("--square_size", type=int, default=224,
                       help="Size of extracted squares")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="DataLoader workers (0 for MacBook)")
    parser.add_argument("--games", type=str, default=None,
                       help="Comma-separated game numbers to use (default: all)")
    
    args = parser.parse_args()
    
    # Parse game numbers
    game_numbers = None
    if args.games:
        game_numbers = [int(g.strip()) for g in args.games.split(",")]
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"\n{'='*60}")
    print("CHESS SQUARE CLASSIFIER TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    df = load_all_games(args.data_root, game_numbers)
    
    if len(df) == 0:
        print("ERROR: No data loaded. Check your data_root path.")
        return
    
    # Split by game
    print("\nSplitting by game...")
    train_df, val_df = split_by_game(df, val_ratio=0.2)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        square_size=args.square_size,
        board_size=args.board_size
    )
    
    # Create model
    print(f"\nCreating model ({args.backbone})...")
    model = create_model(
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr
    )
    
    history = trainer.train(num_epochs=args.epochs)
    
    # Generate visualizations
    print("\nGenerating training curves...")
    plot_training_curves(
        csv_path=os.path.join(args.output_dir, "training_summary.csv"),
        output_path=os.path.join(args.output_dir, "training_curves.png")
    )
    
    # Generate report
    print("\nGenerating training report...")
    report = create_training_report(
        csv_path=os.path.join(args.output_dir, "training_summary.csv"),
        output_dir=args.output_dir
    )
    print(report)
    
    # Find optimal threshold
    print("\nOptimizing OOD threshold...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    best_threshold, threshold_results = find_optimal_threshold(
        model=model,
        val_loader=val_loader,
        device=device
    )
    
    # Save threshold
    with open(os.path.join(args.output_dir, "optimal_threshold.txt"), "w") as f:
        f.write(f"optimal_threshold={best_threshold}\n")
        for t, res in threshold_results.items():
            f.write(f"threshold={t}: acc={res['accuracy']:.4f}, coverage={res['coverage']:.4f}\n")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best model saved to: {args.output_dir}/best_model.pth")
    print(f"Training summary:    {args.output_dir}/training_summary.csv")
    print(f"Training curves:     {args.output_dir}/training_curves.png")
    print(f"Training report:     {args.output_dir}/training_report.txt")
    print(f"Optimal threshold:   {best_threshold}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

