#!/usr/bin/env python3
"""
Evaluate trained model on specific game(s).

Usage:
    python evaluate_game.py --data_root data --model outputs/best_model.pth --games 2
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import ChessboardDataset, get_default_transforms
from data.data_loader import fen_to_labels, apply_occlusion_to_fen
from models.classifier import create_model
from training.trainer import get_device


def load_game_data(data_root: str, game_numbers: list) -> pd.DataFrame:
    """Load data for specific game(s)."""
    all_data = []
    data_path = Path(data_root)
    game_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and "game" in f.name.lower()])

    for game_folder in game_folders:
        game_name = game_folder.name
        try:
            game_num = int(''.join(filter(str.isdigit, game_name)))
        except ValueError:
            continue

        if game_num not in game_numbers:
            continue

        csv_files = list(game_folder.glob("*.csv"))
        if not csv_files:
            continue

        df = pd.read_csv(csv_files[0])
        images_dir = game_folder / "tagged_images"
        if not images_dir.exists():
            images_dir = game_folder / "images"
        if not images_dir.exists():
            continue

        for _, row in df.iterrows():
            if 'from_frame' in df.columns:
                frame_num = int(row['from_frame'])
            elif 'frame' in df.columns:
                frame_num = int(row['frame'])
            else:
                continue

            img_path = images_dir / f"frame_{frame_num:06d}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"frame_{frame_num:06d}.png"
            if not img_path.exists():
                continue

            fen = row['fen']
            if 'occluded' in df.columns and pd.notna(row.get('occluded')) and str(row['occluded']).strip():
                fen = apply_occlusion_to_fen(fen, row['occluded'])

            all_data.append({
                "image_path": str(img_path),
                "fen": fen,
                "game_id": game_num
            })

    result_df = pd.DataFrame(all_data)
    if len(result_df) > 0:
        valid_rows = []
        for idx, row in result_df.iterrows():
            try:
                fen_to_labels(row['fen'], flatten=True)
                valid_rows.append(idx)
            except (ValueError, KeyError):
                pass
        result_df = result_df.loc[valid_rows].reset_index(drop=True)
        result_df = result_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)

    print(f"Loaded {len(result_df)} samples from game(s) {game_numbers}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on specific games")
    parser.add_argument("--data_root", type=str, default="data", help="Data directory")
    parser.add_argument("--model", type=str, default="outputs/best_model.pth", help="Model path")
    parser.add_argument("--games", type=str, required=True, help="Comma-separated game numbers, e.g. '2' or '2,4'")
    parser.add_argument("--backbone", type=str, default="efficientnet_b2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--board_context", action="store_true")
    args = parser.parse_args()

    game_numbers = [int(g.strip()) for g in args.games.split(",")]
    device = get_device()
    print(f"Device: {device}")

    # Load data
    df = load_game_data(args.data_root, game_numbers)
    if len(df) == 0:
        print("ERROR: No data loaded.")
        return

    # Create dataset (val transforms only - no augmentation)
    _, val_transform = get_default_transforms()
    dataset = ChessboardDataset(df, transform=val_transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = create_model(
        backbone=args.backbone, pretrained=False,
        freeze_ratio=0.0, dropout=0.3,
        use_board_context=args.board_context
    )
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Class names
    class_names = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'empty', 'occluded']

    # Evaluate
    print(f"\nEvaluating on game(s) {game_numbers}...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for squares, labels in loader:
            squares = squares.to(device)
            labels = labels.to(device)
            logits = model(squares)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            preds = logits_flat.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels_flat.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n{'='*60}")
    print(f"RESULTS: Game(s) {game_numbers}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Total Squares: {len(all_labels)}")
    print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 42)

    for c in range(14):
        mask = all_labels == c
        total_c = mask.sum()
        if total_c > 0:
            correct_c = (all_preds[mask] == c).sum()
            acc_c = correct_c / total_c
            print(f"{class_names[c]:<12} {correct_c:>8} {total_c:>8} {acc_c:>10.4f}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    confusion = np.zeros((14, 14), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    header = "          " + " ".join(f"{n:>5}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = confusion[i]
        if row.sum() > 0:
            row_str = " ".join(f"{v:>5}" for v in row)
            print(f"{name:<10}{row_str}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
