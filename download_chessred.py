#!/usr/bin/env python3
"""
Download and integrate ChessReD dataset from Hugging Face.

ChessReD (Chess Recognition Dataset):
- 10,800 real-world images
- Fully annotated with FEN strings and board corner coordinates
- Captured from various angles (0° to 60°)
- Perfect for improving model robustness

Dataset link: https://huggingface.co/datasets/georgemavrilis/ChessReD
"""

import os
import pandas as pd
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable


def integrate_chessred(output_root="./data", split_name="game0_chessred", max_images=100):
    """
    Download ChessReD dataset and format it to match project structure.
    
    Args:
        output_root: Root directory for data (default: ./data)
        split_name: Name for the game folder (default: game_chessred)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Please install it with: pip install datasets tqdm")
        return
    
    # 1. Create a specific "game" folder for ChessReD
    game_dir = Path(output_root) / split_name
    images_dir = game_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading ChessReD Dataset from Hugging Face")
    print("="*60)
    print(f"Output directory: {game_dir}")
    print()
    
    # Load the dataset (images are downloaded automatically)
    print("Loading dataset from Hugging Face...")
    print("NOTE: ChessReD is actually hosted on 4TU.ResearchData/Zenodo, not Hugging Face.")
    print("The dataset requires manual download from:")
    print("  - 4TU.ResearchData: https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f")
    print("  - Zenodo: (search for 'ChessReD' or 'Chess Recognition Dataset')")
    print()
    print("Attempting to find alternative chess datasets on Hugging Face...")
    
    # Try alternative datasets
    alternative_datasets = [
        'jalFaizy/detect_chess_pieces',
        'dopaul/chess-pieces-merged',
        'Waterhorse/chess_data'
    ]
    
    dataset = None
    for ds_name in alternative_datasets:
        try:
            print(f"Trying: {ds_name}")
            dataset = load_dataset(ds_name)
            print(f"Success! Using alternative dataset: {ds_name}")
            break
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
            continue
    
    if dataset is None:
        print("\nERROR: Could not access ChessReD or alternative datasets.")
        print("\nTo download ChessReD manually:")
        print("1. Visit: https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f")
        print("2. Create an account and accept the CC BY-NC-SA 4.0 license")
        print("3. Download the dataset")
        print("4. Extract and place in data/game0_chessred/")
        return
    
    print(f"Dataset splits available: {list(dataset.keys())}")
    print()
    
    all_rows = []
    total_images = 0
    
    # Process all splits (train/val/test) for maximum data
    # Limit to max_images total
    images_downloaded = 0
    
    for split in dataset.keys():
        if images_downloaded >= max_images:
            break
            
        print(f"Processing {split} split ({len(dataset[split])} images)...")
        remaining = max_images - images_downloaded
        
        for i, item in enumerate(tqdm(dataset[split][:remaining], desc=f"  {split}")):
            if images_downloaded >= max_images:
                break
            try:
                # Define image name and path
                # Use format: frame_XXXXXX.jpg to match project convention
                frame_num = total_images  # Use sequential numbering across all splits
                image_filename = f"frame_{frame_num:06d}.jpg"
                image_path = images_dir / image_filename
                
                # Save the PIL Image object to disk
                if 'image' in item:
                    item['image'].save(image_path)
                else:
                    print(f"Warning: No image found in item {i}")
                    continue
                
                # Get FEN string
                fen = item.get('fen', '')
                if not fen:
                    print(f"Warning: No FEN found in item {i}")
                    continue
                
                # Prepare row for your game.csv format
                row_data = {
                    "from_frame": total_images,
                    "to_frame": total_images,
                    "fen": fen,
                    "split": split  # Optional metadata
                }
                
                # Add corner coordinates if available
                if 'corners' in item and item['corners'] is not None:
                    corners = item['corners']
                    if len(corners) == 4:
                        # Store corners as string representation
                        row_data["corner_0_x"] = corners[0][0] if isinstance(corners[0], (list, tuple)) else corners[0]
                        row_data["corner_0_y"] = corners[0][1] if isinstance(corners[0], (list, tuple)) else corners[0]
                        row_data["corner_1_x"] = corners[1][0] if isinstance(corners[1], (list, tuple)) else corners[1]
                        row_data["corner_1_y"] = corners[1][1] if isinstance(corners[1], (list, tuple)) else corners[1]
                        row_data["corner_2_x"] = corners[2][0] if isinstance(corners[2], (list, tuple)) else corners[2]
                        row_data["corner_2_y"] = corners[2][1] if isinstance(corners[2], (list, tuple)) else corners[2]
                        row_data["corner_3_x"] = corners[3][0] if isinstance(corners[3], (list, tuple)) else corners[3]
                        row_data["corner_3_y"] = corners[3][1] if isinstance(corners[3], (list, tuple)) else corners[3]
                
                all_rows.append(row_data)
                total_images += 1
                images_downloaded += 1
                
            except Exception as e:
                print(f"Error processing item {i} in {split}: {e}")
                continue
    
    # 2. Save the CSV in your expected format
    if all_rows:
        df = pd.DataFrame(all_rows)
        # Use the game folder name for CSV (e.g., game0_chessred.csv)
        csv_filename = f"{split_name}.csv"
        csv_path = game_dir / csv_filename
        df.to_csv(csv_path, index=False)
        
        print()
        print("="*60)
        print("Integration Complete!")
        print("="*60)
        print(f"Total images saved: {total_images}")
        print(f"Total rows in CSV: {len(df)}")
        print(f"Data location: {game_dir}")
        print(f"CSV file: {csv_path}")
        print(f"Images directory: {images_dir}")
        print()
        print("Dataset splits breakdown:")
        if 'split' in df.columns:
            print(df['split'].value_counts().to_string())
        print()
        print("To use this data in training, run:")
        print(f"  python train.py --data_root {output_root} --epochs 15")
        print("="*60)
    else:
        print("ERROR: No data was successfully downloaded.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and integrate ChessReD dataset from Hugging Face"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data",
        help="Root directory for data (default: ./data)"
    )
    parser.add_argument(
        "--game_name",
        type=str,
        default="game0_chessred",
        help="Name for the game folder (default: game0_chessred - must contain a number)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=100,
        help="Maximum number of images to download (default: 100)"
    )
    
    args = parser.parse_args()
    
    integrate_chessred(
        output_root=args.output_root,
        split_name=args.game_name,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
