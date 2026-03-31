#!/usr/bin/env python3
"""
Download and integrate Kaggle Chess Positions dataset.

Dataset: koryakinp/chess-positions
- Synthetic chess board images
- Generated using chess-generator
- Includes FEN annotations

Kaggle link: https://www.kaggle.com/datasets/koryakinp/chess-positions
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import json
import shutil

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable


def integrate_kaggle_chess(output_root="./data", game_name="game1_kaggle", max_images=100):
    """
    Download Kaggle chess positions dataset and format it to match project structure.
    
    Args:
        output_root: Root directory for data (default: ./data)
        game_name: Name for the game folder (default: game1_kaggle)
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: 'kagglehub' library not installed.")
        print("Please install it with: pip install kagglehub")
        return
    
    # 1. Create game folder
    game_dir = Path(output_root) / game_name
    images_dir = game_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Kaggle Chess Positions Dataset")
    print("="*60)
    print(f"Output directory: {game_dir}")
    print()
    
    # Download the dataset
    print("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("koryakinp/chess-positions")
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("Make sure you have:")
        print("  1. Kaggle account")
        print("  2. Kaggle API credentials set up (kaggle.json)")
        print("  3. Internet connection")
        return
    
    dataset_path = Path(path)
    print(f"Dataset path: {dataset_path}")
    print()
    
    # Find all image files and JSON annotations
    print("Scanning dataset files...")
    image_files = []
    json_files = []
    
    # Look for common image extensions
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(dataset_path.rglob(ext)))
    
    # Look for JSON files (annotations)
    json_files = list(dataset_path.rglob("*.json"))
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(json_files)} JSON files")
    print()
    
    # Process images
    all_rows = []
    frame_counter = 0
    
    # Limit to max_images
    images_processed = 0
    
    # If we have JSON files, use them for annotations
    if json_files:
        print(f"Processing images with JSON annotations (max {max_images})...")
        
        # Create a mapping from image name to JSON data
        json_map = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # JSON might contain image filename or we match by name
                    json_map[json_file.stem] = data
            except Exception as e:
                print(f"Warning: Could not read {json_file}: {e}")
        
        # Process each image
        for img_path in tqdm(image_files[:max_images], desc="Processing images"):
            if images_processed >= max_images:
                break
            try:
                # Get image name without extension
                img_stem = img_path.stem
                
                # Try to find matching JSON
                json_data = json_map.get(img_stem, None)
                
                # Get FEN from JSON if available
                fen = None
                if json_data:
                    # Try different possible keys for FEN
                    fen = json_data.get('fen') or json_data.get('FEN') or json_data.get('board')
                    if isinstance(fen, dict):
                        fen = fen.get('fen') or fen.get('FEN')
                
                # If no FEN in JSON, try to extract from filename or use placeholder
                if not fen:
                    # Some datasets encode FEN in filename
                    if 'fen' in img_stem.lower():
                        # Try to extract FEN from filename
                        parts = img_stem.split('_')
                        for part in parts:
                            if '/' in part and len(part) > 10:  # Looks like FEN
                                fen = part
                                break
                
                # If still no FEN, we'll skip or use a placeholder
                if not fen:
                    # For now, skip images without FEN
                    # You might want to generate FEN from image analysis later
                    continue
                
                # Copy image to our structure
                image_filename = f"frame_{frame_counter:06d}.jpg"
                dest_path = images_dir / image_filename
                
                # Convert to JPG if needed
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                img.save(dest_path, "JPEG", quality=95)
                
                # Add to CSV
                all_rows.append({
                    "from_frame": frame_counter,
                    "to_frame": frame_counter,
                    "fen": fen
                })
                
                frame_counter += 1
                images_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    else:
        # No JSON files - try to extract FEN from filenames
        # Kaggle dataset uses format like: "1B1B1K2-3p1N2-6k1-R7-5P2-4q3-7R-1B6.jpeg"
        # Where dashes separate ranks and need to be converted to FEN format
        print(f"No JSON files found. Attempting to extract FEN from filenames (max {max_images})...")
        
        for img_path in tqdm(image_files[:max_images], desc="Processing images"):
            if images_processed >= max_images:
                break
            try:
                # Extract FEN from filename
                img_stem = img_path.stem
                fen = None
                
                # Kaggle format: ranks separated by dashes
                # Example: "1B1B1K2-3p1N2-6k1-R7-5P2-4q3-7R-1B6"
                if '-' in img_stem:
                    # Split by dashes to get 8 ranks
                    ranks = img_stem.split('-')
                    if len(ranks) == 8:
                        # Convert to standard FEN format (slashes between ranks)
                        # Note: This format might need adjustment based on actual encoding
                        fen = '/'.join(ranks)
                    elif len(ranks) > 8:
                        # Might have additional info, take first 8
                        fen = '/'.join(ranks[:8])
                
                # Also try standard FEN format (with slashes)
                elif '/' in img_stem and len(img_stem) > 20:
                    fen = img_stem
                
                # If we can't extract FEN, skip for now
                if not fen:
                    continue
                
                # Copy image
                image_filename = f"frame_{frame_counter:06d}.jpg"
                dest_path = images_dir / image_filename
                
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                img.save(dest_path, "JPEG", quality=95)
                
                all_rows.append({
                    "from_frame": frame_counter,
                    "to_frame": frame_counter,
                    "fen": fen
                })
                
                frame_counter += 1
                images_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Save CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = game_dir / f"{game_name}.csv"
        df.to_csv(csv_path, index=False)
        
        print()
        print("="*60)
        print("Integration Complete!")
        print("="*60)
        print(f"Total images processed: {len(df)}")
        print(f"Data location: {game_dir}")
        print(f"CSV file: {csv_path}")
        print(f"Images directory: {images_dir}")
        print()
        print("To use this data in training, run:")
        print(f"  python train.py --data_root {output_root} --epochs 15")
        print("="*60)
    else:
        print("ERROR: No data was successfully processed.")
        print("This might be because:")
        print("  1. No FEN annotations found in JSON files")
        print("  2. FEN could not be extracted from filenames")
        print("  3. Dataset structure is different than expected")
        print()
        print("You may need to manually inspect the dataset structure:")
        print(f"  {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and integrate Kaggle chess positions dataset"
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
        default="game1_kaggle",
        help="Name for the game folder (default: game1_kaggle - must contain a number)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=100,
        help="Maximum number of images to download (default: 100)"
    )
    
    args = parser.parse_args()
    
    integrate_kaggle_chess(
        output_root=args.output_root,
        game_name=args.game_name,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
